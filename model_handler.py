# model_handler.py
# -----------------------------------------------------------------------------
# TorchServe handler for BiRefNet (background removal) using a TorchScript model.
# - Registers TorchVision C++ ops (fixes: Unknown builtin op torchvision::deform_conv2d)
# - Robust logging for Docker (stdout)
# - Accepts input bytes (PNG/JPG/WEBP) under key "input_image"
# - Ensures H/W divisible by 32 (BiRefNet-friendly)
# - Returns a single-channel PNG mask as bytes in a one-item list
# -----------------------------------------------------------------------------

import io
import os
import time
import logging
import base64
from typing import Any, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler

# -----------------------------------------------------------------------------
# Logging configuration (stdout-friendly for Docker)
# -----------------------------------------------------------------------------
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(message)s",
	handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class ModelHandler(BaseHandler):
	# -----------------------------------------------------------------------------
	# Handler entrypoint for inference requests
	# -----------------------------------------------------------------------------
	def handle(self, data, context):
		logger.info(f"BiRefNet - Inference start")

		try:
			preprocessed = self.preprocess(data)
			infer_out = self.inference(preprocessed)
			return self.postprocess(infer_out)
		except Exception as e:
			logger.exception(f"BiRefNet - 🚨 Inference failed: {e}")
			return [{"error": str(e)}]

	# -----------------------------------------------------------------------------
	# Initialization: device, torchvision op registration, model load, pre-warm
	# -----------------------------------------------------------------------------
	def initialize(self, context):
		# Worker identity
		self.worker_pid = os.getpid()
		logger.info(f"[INIT] PID={self.worker_pid} starting initialization")

		# Enable Tensor Cores
		torch.backends.cudnn.enabled = True
		torch.backends.cudnn.benchmark = True

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
		self.model.eval()
		logger.info("[INIT] TorchScript model loaded and set to eval()")

		# Verify model is on correct device
		try:
			# Try to get a parameter to check device
			first_param = next(self.model.parameters(), None)
			if first_param is not None:
				logger.info(f"[INIT] Model device: {first_param.device}")
		except:
			# TorchScript models might not expose parameters
			pass

		# Model expects input in [0,1] range, no normalization needed
		self.batch_size = 1

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
	# Preprocess: decode image, ensure size divisible by 32, convert to [0,1]
	# -----------------------------------------------------------------------------
	def preprocess(self, data) -> Dict[str, Any]:

		# Extract image data from request
		raw = data[0].get("input_image")
		if raw is None:
			raise ValueError("Missing 'input_image' input")

		logger.info(f"BiRefNet - Input type: {type(raw)}, size: {len(raw) if hasattr(raw, '__len__') else 'N/A'} bytes")

		# Accept base64 strings for convenience
		if isinstance(raw, str):
			try:
				# tolerate data URI prefix
				if raw.startswith("data:"):
					raw = raw.split(",", 1)[1]
				raw = base64.b64decode(raw, validate=False)
			except Exception as e:
				raise ValueError(f"Failed to base64-decode input: {e}")

		if not isinstance(raw, (bytes, bytearray)):
			raise TypeError(f"Image data must be bytes or base64 string, got {type(raw)}")

		# Decode using PIL and convert to RGB
		image = Image.open(io.BytesIO(raw)).convert("RGB")
		orig_w, orig_h = image.size
		logger.info(f"BiRefNet - Decoded size: {orig_w}x{orig_h}")

		# NOTE: The TorchScript model has ResizeWrapper that handles resizing to 1024x1024
		# We don't need to resize here - just pass the original image

		# Convert to float32 array in [0,1] range
		arr = np.asarray(image, dtype=np.float32) / 255.0

		# Debug log to verify input range
		logger.info(f"BiRefNet - Input array stats: min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")

		return {
			"image_norm": arr,                # HWC float32 in [0,1] range
			"original_size": (orig_h, orig_w) # (H, W) for later resize-back
		}

	# -----------------------------------------------------------------------------
	# Inference: tensorize, forward, sigmoid, resize back to original
	# -----------------------------------------------------------------------------
	def inference(self, data) -> Dict[str, Any]:
		img = data["image_norm"]  # HWC float32
		orig_h, orig_w = data["original_size"]

		# HWC -> NCHW
		tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous()  # [1,3,H,W]
		tensor = tensor.to(self.device, non_blocking=True)

		start = time.time()
		with torch.inference_mode():
			out = self.model(tensor)
			if isinstance(out, (list, tuple)):
				out = out[-1]

			# Debug: log raw output stats
			logger.info(f"BiRefNet - Raw output: min={out.min().item():.4f}, max={out.max().item():.4f}, mean={out.mean().item():.4f}")

			out = torch.sigmoid(out)
			logger.info(f"BiRefNet - After sigmoid: min={out.min().item():.4f}, max={out.max().item():.4f}, mean={out.mean().item():.4f}")

		elapsed = time.time() - start
		logger.info(f"BiRefNet - Inference time: {elapsed:.3f}s")

		# The model with ResizeWrapper always outputs 1024x1024, resize back to original
		out = F.interpolate(out, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

		# Convert to uint8 mask
		mask = out.squeeze(0).squeeze(0).detach().cpu().clamp_(0, 1).numpy()
		mask_u8 = (mask * 255.0).astype(np.uint8)

		# Debug: check if mask is all black
		unique_vals = np.unique(mask_u8)
		logger.info(f"BiRefNet - Mask unique values: {unique_vals[:10] if len(unique_vals) > 10 else unique_vals}, total unique: {len(unique_vals)}")
		logger.info(f"BiRefNet - Mask stats: min={mask_u8.min()}, max={mask_u8.max()}, mean={mask_u8.mean():.1f}")

		# Encode as PNG
		pil_out = Image.fromarray(mask_u8, mode="L")
		buf = io.BytesIO()
		pil_out.save(buf, format="PNG", compress_level=1)
		png_bytes = buf.getvalue()

		# Free GPU memory if available
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

		return {"output_image": png_bytes}

	# -----------------------------------------------------------------------------
	# Postprocess: package bytes for TorchServe
	# -----------------------------------------------------------------------------
	def postprocess(self, data):
		return [data["output_image"]]