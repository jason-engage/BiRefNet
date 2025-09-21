#!/usr/bin/env python
"""
Local test script for BiRefNet model_handler.py
Uses the SAME preprocess and inference code as the production handler
"""

import io
import os
import time
import logging
import base64
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision  # CRITICAL: Must import before loading TorchScript model with deform_conv2d
import requests
from pathlib import Path

# -----------------------------------------------------------------------------
# Logging configuration (stdout-friendly for Docker)
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class MockContext:
    """Mock context for local testing"""
    def __init__(self, model_path):
        self.system_properties = {"model_dir": os.path.dirname(os.path.abspath(model_path))}
        self.manifest = {"model": {"serializedFile": os.path.basename(model_path)}}


class ModelHandler:
    """
    EXACT COPY of model_handler.py with minimal modifications for local testing
    The preprocess() and inference() methods are UNCHANGED from production
    """

    # -----------------------------------------------------------------------------
    # Initialization: device, torchvision op registration, model load, pre-warm
    # -----------------------------------------------------------------------------
    def initialize(self, context):
        # Worker identity
        self.worker_pid = os.getpid()
        logger.info(f"[INIT] PID={self.worker_pid} starting initialization")

        # Backend knobs (enable Tensor Cores paths where applicable)
        try:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            # Optional: allow higher-precision matmul on Ampere+
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        except Exception as e:
            logger.warning(f"[INIT] Could not tune backend flags: {e}")

        # Device selection & diagnostics
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            logger.info(f"[INIT] device=cuda:{idx} name={props.name} vram={props.total_memory/1024/1024:.0f}MiB")
        except Exception as e:
            logger.warning(f"[INIT] Could not query CUDA device properties: {e}")

        # Resolve model path from context
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        model_file = context.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, model_file)
        logger.info(f"[INIT] Model path: {model_pt_path}")

        # -------------------------------------------------------------------------
        # CRITICAL: Register TorchVision C++ ops *before* torch.jit.load(...)
        # This fixes "Unknown builtin op: torchvision::deform_conv2d".
        # -------------------------------------------------------------------------
        try:
            import torchvision  # defer import so each worker registers its ops
            _ = torchvision.ops.deform_conv2d  # touch to force extension load/registration

            has_tv_ns = hasattr(torch.ops, "torchvision")
            has_dcn = has_tv_ns and ("deform_conv2d" in dir(torch.ops.torchvision))
            logger.info(f"[INIT] torch={torch.__version__} tv={torchvision.__version__} cuda={torch.version.cuda}")
            logger.info(f"[INIT] torchvision ns present={has_tv_ns} deform_conv2d registered={has_dcn}")
            if not has_dcn:
                logger.warning("[INIT] torchvision::deform_conv2d NOT visible after import — version/build mismatch likely.")
        except Exception as e:
            logger.exception(f"[INIT][ERROR] Failed to load/register torchvision ops: {e}")

        # Load TorchScript model and set eval
        self.model = torch.jit.load(model_pt_path, map_location=self.device)
        try:
            self.model.to(self.device)  # safe for TorchScript; no-op if already there
        except Exception as e:
            logger.warning(f"[INIT] model.to(device) failed (non-fatal for JIT): {e}")
        self.model.eval()
        logger.info("[INIT] TorchScript loaded and set to eval().")

        # Inference parameters - NO NORMALIZATION for this model!
        self.batch_size = 1
        # The TorchScript model expects input in [0,1] range, no normalization
        # self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # NOT USED
        # self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)   # NOT USED

        # Optional pre-warm to surface missing ops early and cache kernels
        try:
            with torch.inference_mode():
                dummy = torch.zeros(1, 3, 32, 32, dtype=torch.float32, device=self.device)
                _ = self.model(dummy)
            logger.info("[INIT] Pre-warm forward succeeded.")
        except Exception as e:
            logger.warning(f"[INIT] Pre-warm forward failed (non-fatal): {e}")

        logger.info(f"BiRefNet - [PID: {self.worker_pid}] Initialized successfully.")

    # -----------------------------------------------------------------------------
    # Preprocess: EXACT COPY FROM model_handler.py
    # -----------------------------------------------------------------------------
    def preprocess(self, data):
        logger.info(f"BiRefNet - [PID: {self.worker_pid}] Preprocess start: {time.strftime('%H:%M:%S')}")

        # Extract payload (bytes or base64 string)
        raw = data[0].get("input_image")
        if raw is None:
            raise ValueError("Missing 'input_image' input")

        # Accept base64 strings for convenience
        if isinstance(raw, str):
            try:
                # tolerate data URI prefix
                if raw.startswith("data:"):
                    raw = raw.split(",", 1)[1]
                raw = base64.b64decode(raw, validate=False)
            except Exception as e:
                raise ValueError(f"Failed to base64-decode input_image: {e}")

        if not isinstance(raw, (bytes, bytearray)):
            raise TypeError(f"'input_image' must be bytes or base64 string, got {type(raw)}")

        # Decode using PIL and convert to RGB
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        orig_w, orig_h = image.size
        logger.info(f"BiRefNet - [PID: {self.worker_pid}] Decoded image size: {orig_w}x{orig_h}")

        # Ensure dimensions divisible by 32 (BiRefNet-friendly)
        new_w = ((orig_w + 31) // 32) * 32
        new_h = ((orig_h + 31) // 32) * 32
        if (new_w, new_h) != (orig_w, orig_h):
            image = image.resize((new_w, new_h), Image.BILINEAR)
            logger.info(f"BiRefNet - [PID: {self.worker_pid}] Resized to {new_w}x{new_h} for 32-divisibility")

        # Convert to float32 array in [0,1] - NO NORMALIZATION!
        arr = np.asarray(image, dtype=np.float32) / 255.0
        logger.info(f"BiRefNet - Array range [0,1]: min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")

        # NO ImageNet normalization - model expects [0,1] range
        # arr = (arr - self.mean) / self.std  # REMOVED - model doesn't need this!

        return {
            "image_norm": arr,                # HWC float32 in [0,1] range (NOT normalized!)
            "original_size": (orig_h, orig_w) # (H, W) for later resize-back
        }

    # -----------------------------------------------------------------------------
    # Inference: EXACT COPY FROM model_handler.py
    # -----------------------------------------------------------------------------
    def inference(self, data):
        img = data["image_norm"]  # HWC float32
        orig_h, orig_w = data["original_size"]
        logger.info(f"BiRefNet - [PID: {self.worker_pid}] Inference input HWC: {img.shape}")

        # HWC -> NCHW
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous()  # [1,3,H,W]
        logger.info(f"BiRefNet - Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
        tensor = tensor.to(self.device, non_blocking=True)
        logger.info(f"BiRefNet - Tensor on device: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}")

        # Debug: Check each channel separately
        for i in range(3):
            ch_min = tensor[0, i].min().item()
            ch_max = tensor[0, i].max().item()
            ch_mean = tensor[0, i].mean().item()
            logger.info(f"BiRefNet - Channel {i}: min={ch_min:.4f}, max={ch_max:.4f}, mean={ch_mean:.4f}")

        start = time.time()
        with torch.inference_mode():
            # Use CUDA stream if available
            if torch.cuda.is_available():
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    out = self.model(tensor)
                    if isinstance(out, (list, tuple)):
                        logger.info(f"BiRefNet - Model returned {len(out)} outputs, using last one")
                        # Debug: Check all outputs
                        for idx, o in enumerate(out):
                            if isinstance(o, torch.Tensor):
                                logger.info(f"  Output {idx}: shape={o.shape}, min={o.min().item():.4f}, max={o.max().item():.4f}, mean={o.mean().item():.4f}")
                        out = out[-1]

                    logger.info(f"BiRefNet - Pre-sigmoid output shape: {out.shape}")
                    logger.info(f"BiRefNet - Pre-sigmoid output: min={out.min().item():.4f}, max={out.max().item():.4f}, mean={out.mean().item():.4f}")

                    # Debug: Check a few pixel values before sigmoid
                    h, w = out.shape[-2:]
                    center_val = out[0, 0, h//2, w//2].item()
                    corner_val = out[0, 0, 0, 0].item()
                    logger.info(f"BiRefNet - Sample values pre-sigmoid: center={center_val:.4f}, corner={corner_val:.4f}")

                    out = torch.sigmoid(out)
                    # Debug: Check output values
                    logger.info(f"BiRefNet - Post-sigmoid output: min={out.min().item():.4f}, max={out.max().item():.4f}, mean={out.mean().item():.4f}")

                    # Check same pixels after sigmoid
                    center_val_post = out[0, 0, h//2, w//2].item()
                    corner_val_post = out[0, 0, 0, 0].item()
                    logger.info(f"BiRefNet - Sample values post-sigmoid: center={center_val_post:.4f}, corner={corner_val_post:.4f}")
                    torch.cuda.synchronize()
            else:
                out = self.model(tensor)
                if isinstance(out, (list, tuple)):
                    logger.info(f"BiRefNet - Model returned {len(out)} outputs, using last one")
                    out = out[-1]
                logger.info(f"BiRefNet - Pre-sigmoid output: min={out.min().item():.4f}, max={out.max().item():.4f}, mean={out.mean().item():.4f}")
                out = torch.sigmoid(out)
                logger.info(f"BiRefNet - Post-sigmoid output: min={out.min().item():.4f}, max={out.max().item():.4f}, mean={out.mean().item():.4f}")

        elapsed = time.time() - start
        logger.info(f"BiRefNet - [PID: {self.worker_pid}] Forward time: {elapsed:.3f}s")

        # IMPORTANT: The TorchScript model with ResizeWrapper always outputs 1024x1024
        # We need to resize back to the original dimensions
        oh, ow = out.shape[-2], out.shape[-1]
        logger.info(f"BiRefNet - Model output size: {oh}x{ow}, Original size: {orig_h}x{orig_w}")
        if (oh, ow) != (orig_h, orig_w):
            out = F.interpolate(out, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            logger.info(f"BiRefNet - Resized output back to original dimensions")

        # Squeeze to [H,W], clamp, convert to uint8 mask
        mask = out.squeeze(0).squeeze(0).detach().cpu().clamp_(0, 1).numpy()
        logger.info(f"BiRefNet - Final mask: min={mask.min():.4f}, max={mask.max():.4f}, mean={mask.mean():.4f}")
        mask_u8 = (mask * 255.0).astype(np.uint8)
        logger.info(f"BiRefNet - Mask uint8: min={mask_u8.min()}, max={mask_u8.max()}, mean={mask_u8.mean():.1f}, shape={mask_u8.shape}")

        # Encode PNG (grayscale "L")
        pil_out = Image.fromarray(mask_u8, mode="L")
        buf = io.BytesIO()
        # Light compression for speed
        pil_out.save(buf, format="PNG", compress_level=1)
        png_bytes = buf.getvalue()

        # Free some GPU memory between requests
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"output_image": png_bytes}

    # -----------------------------------------------------------------------------
    # Postprocess: EXACT COPY FROM model_handler.py
    # -----------------------------------------------------------------------------
    def postprocess(self, data):
        png_bytes = data["output_image"]
        logger.info(f"BiRefNet - [PID: {self.worker_pid}] Postprocess PNG bytes: {len(png_bytes)}")
        return [png_bytes]

    # -----------------------------------------------------------------------------
    # Handle method for local testing (simplified)
    # -----------------------------------------------------------------------------
    def handle(self, data, context):
        logger.info(f"BiRefNet - 🔥 Inference START at {time.time():.6f}")
        try:
            preprocessed = self.preprocess(data)
            infer_out = self.inference(preprocessed)
            return self.postprocess(infer_out)
        except Exception as e:
            logger.exception(f"BiRefNet - 🚨 Inference failed: {e}")
            return [{"error": str(e)}]


def test_simple_inference(model_path):
    """Test with simple inference without handler wrapper"""
    logger.info("\n" + "=" * 60)
    logger.info("Running SIMPLE inference test (no normalization)")
    logger.info("=" * 60)

    # CRITICAL: Register torchvision ops before loading model
    try:
        import torchvision
        _ = torchvision.ops.deform_conv2d  # Force registration
        logger.info(f"Registered torchvision ops (version {torchvision.__version__})")
    except Exception as e:
        logger.error(f"Failed to register torchvision ops: {e}")

    # Load model directly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    # Create a simple test tensor (white square on black background)
    test_tensor = torch.zeros(1, 3, 1024, 1024, device=device)
    test_tensor[:, :, 256:768, 256:768] = 1.0  # White square in center

    with torch.no_grad():
        output = model(test_tensor)
        if isinstance(output, (list, tuple)):
            output = output[-1]

        logger.info(f"Simple test - Pre-sigmoid: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
        output_sig = torch.sigmoid(output)
        logger.info(f"Simple test - Post-sigmoid: min={output_sig.min():.4f}, max={output_sig.max():.4f}, mean={output_sig.mean():.4f}")

    # Test with normalized input
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    normalized_tensor = (test_tensor - mean) / std

    with torch.no_grad():
        output_norm = model(normalized_tensor)
        if isinstance(output_norm, (list, tuple)):
            output_norm = output_norm[-1]

        logger.info(f"Normalized test - Pre-sigmoid: min={output_norm.min():.4f}, max={output_norm.max():.4f}, mean={output_norm.mean():.4f}")
        output_norm_sig = torch.sigmoid(output_norm)
        logger.info(f"Normalized test - Post-sigmoid: min={output_norm_sig.min():.4f}, max={output_norm_sig.max():.4f}, mean={output_norm_sig.mean():.4f}")

    # Test what the model expects - no normalization, values in [0,1]
    logger.info("\n" + "=" * 60)
    logger.info("Testing with REAL IMAGE (no normalization, [0,1] range)")

    # Load and prepare a real image without normalization
    test_url = "https://cdn.shopify.com/s/files/1/0255/3657/files/123456_1245802d-5b04-40e1-b896-a5661c6058d9.webp"
    response = requests.get(test_url)
    test_img = Image.open(io.BytesIO(response.content)).convert("RGB")
    test_img = test_img.resize((1024, 1024), Image.BILINEAR)  # Resize to model's expected size

    # Convert to tensor in [0,1] range WITHOUT normalization
    test_array = np.array(test_img, dtype=np.float32) / 255.0
    test_tensor_real = torch.from_numpy(test_array).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output_real = model(test_tensor_real)
        if isinstance(output_real, (list, tuple)):
            output_real = output_real[-1]

        logger.info(f"Real image test - Input range: [{test_tensor_real.min():.4f}, {test_tensor_real.max():.4f}]")
        logger.info(f"Real image test - Pre-sigmoid: min={output_real.min():.4f}, max={output_real.max():.4f}, mean={output_real.mean():.4f}")
        output_real_sig = torch.sigmoid(output_real)
        logger.info(f"Real image test - Post-sigmoid: min={output_real_sig.min():.4f}, max={output_real_sig.max():.4f}, mean={output_real_sig.mean():.4f}")

        # Just log the stats, don't save
        logger.info("Simple test completed successfully")


def test_local():
    """Test the model with the specified image"""

    # Configuration - UPDATE THIS PATH
    MODEL_PATH = "pretrained/central_rattlesnake_epoch_12.ts"
    TEST_IMAGE_URL = "https://cdn.shopify.com/s/files/1/0255/3657/files/123456_1245802d-5b04-40e1-b896-a5661c6058d9.webp"

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at: {MODEL_PATH}")
        logger.info("Please update MODEL_PATH to point to your .ts model file")
        return

    logger.info("=" * 60)
    logger.info("Starting BiRefNet local test")
    logger.info("=" * 60)

    # First run simple test to validate model
    test_simple_inference(MODEL_PATH)

    # Initialize handler with mock context
    handler = ModelHandler()
    context = MockContext(MODEL_PATH)

    try:
        handler.initialize(context)
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return

    # Download test image
    logger.info(f"\nDownloading test image from: {TEST_IMAGE_URL}")
    response = requests.get(TEST_IMAGE_URL)
    if response.status_code != 200:
        logger.error(f"Failed to download image: {response.status_code}")
        return

    image_bytes = response.content
    logger.info(f"Downloaded {len(image_bytes)} bytes")

    # Save original image for reference
    os.makedirs("masks", exist_ok=True)
    original_img = Image.open(io.BytesIO(image_bytes))
    original_img.save("masks/original.png")
    logger.info("Saved original image to masks/original.png")

    # Run DIRECT inference (without normalization) - THIS WORKS!
    logger.info("\n" + "=" * 60)
    logger.info("Running direct inference (correct method)")
    logger.info("=" * 60)

    # Prepare image for inference - resize to model's expected size
    test_img = original_img.resize((1024, 1024), Image.BILINEAR)

    # Convert to tensor in [0,1] range WITHOUT normalization
    test_array = np.array(test_img, dtype=np.float32) / 255.0
    test_tensor = torch.from_numpy(test_array).permute(2, 0, 1).unsqueeze(0).to(handler.device)

    with torch.no_grad():
        output = handler.model(test_tensor)
        if isinstance(output, (list, tuple)):
            output = output[-1]

        logger.info(f"Pre-sigmoid: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
        output_sig = torch.sigmoid(output)
        logger.info(f"Post-sigmoid: min={output_sig.min():.4f}, max={output_sig.max():.4f}, mean={output_sig.mean():.4f}")

    # Convert to mask and resize back to original size
    mask = (output_sig.squeeze().cpu().numpy() * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask).resize(original_img.size, Image.BILINEAR)

    # Save the correct mask
    mask_img.save("masks/output_mask.png")
    logger.info(f"✅ Saved correct mask to masks/output_mask.png")

    # Create composite with white background
    composite = Image.new("RGB", original_img.size, (255, 255, 255))  # White background

    # Convert mask to binary mask for cleaner edges
    mask_binary = mask_img.point(lambda x: 255 if x > 127 else 0)

    # Paste original image using mask
    composite.paste(original_img, (0, 0), mask_binary)
    composite.save("masks/composite.png")
    logger.info(f"✅ Saved composite with white background to masks/composite.png")

    logger.info("\n" + "=" * 60)
    logger.info("Test complete! Check the 'masks' directory for:")
    logger.info("  - original.png (input image)")
    logger.info("  - output_mask.png (model output)")
    logger.info("  - composite.png (removed background with white fill)")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_local()