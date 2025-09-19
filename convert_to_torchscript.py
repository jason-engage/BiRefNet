#!/usr/bin/env python
"""
Convert BiRefNet model to TorchScript format for optimized inference.
This version wraps the model so traced export can accept *any* input size.
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models.birefnet import BiRefNet, BiRefNetC2F
from utils import check_state_dict


# ------------------------
# Parse command-line args
# ------------------------
parser = argparse.ArgumentParser(description="Convert BiRefNet model to TorchScript format")
parser.add_argument("-m", "--model", type=str, required=True, help="Path to the model checkpoint (.pth or .bin)")
parser.add_argument("-s", "--size", type=int, nargs=2, default=[1024, 1024], help="Input size (width height). Default: 1024 1024")
parser.add_argument("-t", "--model_type", type=str, default="BiRefNet", choices=["BiRefNet", "BiRefNetC2F"], help="Model type. Default: BiRefNet")
parser.add_argument("-o", "--output", type=str, default=None, help="Output path for TorchScript model. Default: input_path.ts")
args = parser.parse_args()


def convert_to_torchscript(model_path, input_size, model_type="BiRefNet", output_path=None):
    """
    Convert BiRefNet model to TorchScript format with flexible input support.
    """

    # Init config (not directly used but may be needed for BiRefNet)
    config = Config()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------
    # Load checkpoint
    # ------------------------
    print(f"Loading checkpoint from: {model_path}")
    if model_path.endswith(".bin"):
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    else:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

    state_dict = check_state_dict(state_dict)

    # ------------------------
    # Init model
    # ------------------------
    print(f"Initializing {model_type} model...")
    if model_type == "BiRefNet":
        model = BiRefNet(bb_pretrained=False)
    elif model_type == "BiRefNetC2F":
        model = BiRefNetC2F(bb_pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("Model loaded successfully")

    # ------------------------
    # Create wrapper for flexible inputs
    # ------------------------
    width, height = input_size

    class ResizeWrapper(torch.nn.Module):
        def __init__(self, model, size):
            super().__init__()
            self.model = model
            self.size = size  # (H, W)

        def forward(self, x):
            # Always resize input to fixed size
            x = torch.nn.functional.interpolate(
                x, size=self.size, mode="bilinear", align_corners=False
            )
            out = self.model(x)
            if isinstance(out, (list, tuple)):
                out = out[-1]  # take highest resolution
            return out

    wrapped_model = ResizeWrapper(model, (height, width))
    wrapped_model.eval()

    # Example input for tracing
    example_input = torch.randn(1, 3, height, width).to(device)
    print(f"Example input shape for trace: {example_input.shape}")

    # ------------------------
    # Trace
    # ------------------------
    with torch.no_grad():
        scripted_model = torch.jit.trace(wrapped_model, example_input)

    # ------------------------
    # Save
    # ------------------------
    if output_path is None:
        output_path = Path(model_path).with_suffix(".ts")
    else:
        output_path = Path(output_path)

    scripted_model.save(str(output_path))
    print(f"✅ TorchScript model saved to: {output_path}")

    # ------------------------
    # Test
    # ------------------------
    print("Testing scripted model...")
    with torch.no_grad():
        test_input = torch.randn(1, 3, 720, 1280).to(device)  # arbitrary size
        test_output = scripted_model(test_input)
        print(f"Test output shape: {test_output.shape}")

    # Size stats
    original_size = Path(model_path).stat().st_size / (1024 * 1024)
    scripted_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Original checkpoint size: {original_size:.2f} MB")
    print(f"TorchScript model size: {scripted_size:.2f} MB")
    print(f"Size ratio: {scripted_size/original_size:.2%}")

    return output_path


if __name__ == "__main__":
    convert_to_torchscript(
        model_path=args.model,
        input_size=tuple(args.size),
        model_type=args.model_type,
        output_path=args.output,
    )