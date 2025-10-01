import os
import argparse
from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from contextlib import nullcontext
import shutil
import yaml
import tempfile

from models.birefnet import BiRefNet
from utils import check_state_dict, wandComposite
from config import Config


config = Config()

# Load config variables for upscale model path
try:
    with open('config_vars.yml', 'r') as f:
        config_vars = yaml.safe_load(f)
except FileNotFoundError:
    print("Warning: config_vars.yml not found")
    config_vars = {}

def upscale_image(img_pil, device):
    """Upscale image using RealESRGAN"""
    from RealESRGAN.model import RealESRGAN

    # Get upscale model path from config - hardcoded scale factor
    upscale_model_path = config_vars.get('upscale_model_path', 'weights/RealESRGAN_x2.pth')
    # Extract scale factor from the model path (e.g., RealESRGAN_x2.pth -> 2)
    scale = 2  # Default to 2x, but could extract from path if needed
    if '_x4' in upscale_model_path:
        scale = 4

    # Initialize and load the upscaler model
    upscaler = RealESRGAN(device, scale=scale)
    upscaler.load_weights(upscale_model_path, download=False)

    # Upscale the image
    result = upscaler.predictFromPil(img_pil.convert('RGB'))
    return result

def process_image(model, image_path, output_dir, size, device, copy_original=False, should_upscale=False):
    """Process a single image and save mask + composite"""

    from dataset import MyData
    import cv2

    print(f"\nProcessing: {os.path.basename(image_path)}")

    # Load and preprocess image
    img_pil = Image.open(image_path).convert('RGB')
    original_size = img_pil.size
    print(f"  Original size: {original_size}")

    if should_upscale:
        # Check if img_pil width or height are both less than size, otherwise skip
        #    if img_pil.width < size[0] or img_pil.height < size[1]:

        file_size = os.path.getsize(image_path)

        if img_pil.width <= 800 and img_pil.height <= 800 and file_size < 85000:
            print(f"  Upscaling image...")
            # Create temporary file for upscaled image
            upscale_save_path = 'temp_upscaled_image.png'
            try:
                # Upscale the image
                upscaled_img = upscale_image(img_pil, device)
                # Save upscaled_image
                # upscaled_img.save(upscale_save_path)
                # Reload as the main image
                img_pil = upscaled_img
            except Exception as e:
                print(f"  Warning: Upscaling failed: {e}")
           
    img_pil = img_pil.resize((size[0], size[1]), Image.BILINEAR)
    print(f"  Resized to: {size}")

    # Convert to tensor and normalize
    img_np = np.array(img_pil) / 255.0
    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1)
    img_tensor = (img_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    print(f"  Input tensor shape: {img_tensor.shape}")

    # Run inference
    with torch.no_grad():
        pred = model(img_tensor)[-1].sigmoid()
    print(f"  Prediction shape: {pred.shape}")

    # Resize mask back to original size
    mask = torch.nn.functional.interpolate(
        pred,
        size=original_size[::-1],  # PIL uses (W,H), torch uses (H,W)
        mode='bilinear',
        align_corners=True
    ).squeeze().cpu().numpy()
    print(f"  Mask shape after resize: {mask.shape}, min={mask.min():.3f}, max={mask.max():.3f}")

    # Save mask
    basename = os.path.splitext(os.path.basename(image_path))[0]
    # mask_path = os.path.join(output_dir, f"{basename}_mask.png")
    mask_path = os.path.join(output_dir, f"{basename}.png")
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')  # Ensure it's grayscale
    print(f"  Mask PIL image: mode={mask_img.mode}, size={mask_img.size}")
    mask_img.save(mask_path)
    print(f"  Saved mask to: {mask_path}")

    # Create composite using wandComposite
    original_img = Image.open(image_path).convert('RGB')
    print(f"  Original image for composite: mode={original_img.mode}, size={original_img.size}")
    print(f"  Mask image for composite: mode={mask_img.mode}, size={mask_img.size}")

    try:
        composite = wandComposite(original_img, mask_img)
        print(f"  Composite created: mode={composite.mode}, size={composite.size}")
    except Exception as e:
        print(f"  ERROR in wandComposite: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Save as WebP
    composite_path = os.path.join(output_dir, f"{basename}.webp")
    composite.save(composite_path, 'WEBP', quality=95)
    print(f"  Saved composite to: {composite_path}")

    # Copy original if requested
    if copy_original:
        original_ext = os.path.splitext(image_path)[1]
        original_copy_path = os.path.join(output_dir, f"{basename}_original{original_ext}")
        shutil.copy2(image_path, original_copy_path)
        print(f"  Copied original to: {original_copy_path}")

    return mask_path, composite_path


def main(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load model
    print(f'Loading model from {args.model}')
    model = BiRefNet(bb_pretrained=False)
    state_dict = torch.load(args.model, map_location='cpu', weights_only=True)
    state_dict = check_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Parse size
    if args.size and args.size not in ['None', 'none', '0']:
        size = [int(x) for x in args.size.split('x')]
        print(f'Using size: {size[0]}x{size[1]}')
    else:
        size = list(config.size)  # Convert tuple to list
        print(f'Using default config size: {size[0]}x{size[1]}')

    # Determine output directory
    if args.local:
        # Extract model name from path
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        # Append size to directory name
        dir_name = f"{model_name}_{size[0]}x{size[1]}"
        # Append _upscaled if upscaling is enabled
        if args.upscale:
            dir_name += "_upscaled"

        # add images_dir name to output dir
        images_dir_name = os.path.basename(os.path.normpath(args.images))
        dir_name = os.path.join(images_dir_name, dir_name)
                
        output_dir = os.path.join('tests', dir_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f'Local mode: Output directory set to {output_dir}')
        copy_original = True
    else:
        # Use input directory as output
        output_dir = args.images
        copy_original = False

    # Get image list
    image_extensions = ['*.jpg'] #, '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp'
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(args.images, ext)))
        image_files.extend(glob(os.path.join(args.images, ext.upper())))

    if not image_files:
        print(f'No images found in {args.images}')
        return

    print(f'Found {len(image_files)} images to process')

    # Process each image
    for image_path in tqdm(image_files, desc='Processing images'):
        try:
            mask_path, composite_path = process_image(model, image_path, output_dir, size, device, copy_original, args.upscale)
        except Exception as e:
            print(f'Error processing {image_path}: {e}')
            continue

    print(f'Done! Results saved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple BiRefNet inference on directory of images')
    parser.add_argument('--images', required=True, type=str, help='Directory containing images to process')
    parser.add_argument('--model', required=True, type=str, help='Path to model checkpoint (.pth file)')
    parser.add_argument('--size', type=str, help='Inference size as WIDTHxHEIGHT (e.g., 1024x1024). Leave empty to use config.size')
    parser.add_argument('--upscale', action='store_true', help='Apply upscaling before background removal')
    parser.add_argument('--local', action='store_true', help='Save outputs to tests/[model_name]/ and copy originals')

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f'Model file not found: {args.model}')
        exit(1)

    if not os.path.exists(args.images):
        print(f'Image directory not found: {args.images}')
        exit(1)

    main(args)