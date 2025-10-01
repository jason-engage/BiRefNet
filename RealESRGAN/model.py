import os
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import cv2

from .rrdbnet_arch import RRDBNet
from .utils import pad_reflect, split_image_into_overlapping_patches, stich_together, \
	unpad_image

class RealESRGAN:
	def __init__(self, device, scale=4):
		self.device = device
		self.scale = scale
		self.model = RRDBNet(
			num_in_ch=3, num_out_ch=3, num_feat=64,
			num_block=23, num_grow_ch=32, scale=scale
		)

	def load_weights(self, model_path, download=False):
		if not os.path.exists(model_path):
			raise FileNotFoundError(f"Model weights not found at: {model_path}")

		loadnet = torch.load(model_path)
		if 'params' in loadnet:
			self.model.load_state_dict(loadnet['params'], strict=True)
		elif 'params_ema' in loadnet:
			self.model.load_state_dict(loadnet['params_ema'], strict=True)
		else:
			self.model.load_state_dict(loadnet, strict=True)
		self.model.eval()
		self.model.to(self.device)

	# @torch.cuda.amp.autocast()
	def predictFromPil(self, lr_image, batch_size=4, patches_size=192,
				padding=24, pad_size=15):
		"""Process a PIL image through the RealESRGAN model for upscaling.
		
		Args:
			lr_image: PIL Image in RGB format
			batch_size: Number of patches to process at once (for memory efficiency)
			patches_size: Size of each square patch to process
			padding: Overlap size between patches to avoid boundary artifacts
			pad_size: Extra padding around the entire image to handle edges
		"""
		# Enable autocast for mixed precision inference if available
		torch.autocast(device_type=self.device.type)
		scale = self.scale
		device = self.device

		# Convert PIL image to numpy array (shape: H x W x C)
		np_image = np.array(lr_image)
		print(f"Image shape pre-padding: Type: {np_image.dtype}, Shape: {np_image.shape}") # Type: uint8, Shape: (819, 819, 3)

		# Add reflection padding around the image to better handle edges
		np_image = pad_reflect(np_image, pad_size)
		print(f"Image shape pre-patching: Type: {np_image.dtype}, Shape: {np_image.shape}") # Type: uint8, Shape: (819, 819, 3)

		# Split image into overlapping patches to process in batches
		# This helps with memory usage and maintains consistency across the image
		patches, p_shape = split_image_into_overlapping_patches(
			np_image, patch_size=patches_size, padding_size=padding
		)
		
		# Convert patches to torch tensor and prepare for model:
		# 1. Normalize to [0, 1] range by dividing by 255
		# 2. Permute from (N, H, W, C) to (N, C, H, W) format for PyTorch
		# 3. Move to target device (CPU/GPU)
		img = torch.FloatTensor(patches / 255).permute((0, 3, 1, 2)).to(device).detach()

		# Process patches through the model in batches
		with torch.no_grad():
			print(f"Predicting... Type: {img[0:0 + batch_size].dtype}, Shape: {img[0:0 + batch_size].shape}")  # Type: torch.float32, Shape: torch.Size([4, 3, 240, 240])
			# Process each batch and concatenate results
			prediction = torch.cat([self.model(img[i:i + batch_size]) for i in range(0, img.shape[0], batch_size)], dim=0)

		print(f"Image shape post-prediction: Type: {prediction.dtype}, Shape: {prediction.shape}") #Type: torch.float32, Shape: torch.Size([25, 3, 480, 480])

		# Convert prediction back to numpy format:
		# 1. Permute from (N, C, H, W) to (N, H, W, C)
		# 2. Clamp values to [0, 1] range
		# 3. Convert to numpy array
		np_sr_image = prediction.permute((0, 2, 3, 1)).cpu().clamp_(0, 1).numpy()
		print(f"Image shape post-prediction permute: Type: {np_sr_image.dtype}, Shape: {np_sr_image.shape}") #Type: float32, Shape: (25, 480, 480, 3)

		# Calculate target shapes for the padded and final images
		# Scale dimensions by the upscaling factor
		padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
		scaled_image_shape = tuple(np.multiply(np_image.shape[0:2], scale)) + (3,)

		# Stitch the patches back together, handling overlapping regions
		np_sr_image = stich_together(
			np_sr_image, padded_image_shape=padded_size_scaled,
			target_shape=scaled_image_shape, padding_size=padding * scale
		)

		# Convert back to uint8 format (0-255 range)
		np_sr_image = (np_sr_image * 255).astype(np.uint8)
		# Remove the padding that was added at the start
		np_sr_image = unpad_image(np_sr_image, pad_size * scale)

		print(f"Image shape pre-pil: Type: {np_sr_image.dtype}, Shape: {np_sr_image.shape}") #Type: uint8, Shape: (1578, 1578, 3)

		# Convert back to PIL image format
		return Image.fromarray(np_sr_image)