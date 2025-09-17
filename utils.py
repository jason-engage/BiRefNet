import logging
import os
import torch
from torchvision import transforms
import numpy as np
import random
import cv2
from PIL import Image


def path_to_image(path, size=(1024, 1024), color_type=['rgb', 'gray'][0]):
    if color_type.lower() == 'rgb':
        image = cv2.imread(path)
    elif color_type.lower() == 'gray':
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        print('Select the color_type to return, either to RGB or gray image.')
        return
    if size:
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    if color_type.lower() == 'rgb':
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGB')
    else:
        image = Image.fromarray(image).convert('L')
    return image



def check_state_dict(state_dict, unwanted_prefixes=['module.', '_orig_mod.']):
    for k, v in list(state_dict.items()):
        prefix_length = 0
        for unwanted_prefix in unwanted_prefixes:
            if k[prefix_length:].startswith(unwanted_prefix):
                prefix_length += len(unwanted_prefix)
        state_dict[k[prefix_length:]] = state_dict.pop(k)
    return state_dict


def generate_smoothed_gt(gts):
    epsilon = 0.001
    new_gts = (1-epsilon)*gts+epsilon/2
    return new_gts


class Logger():
    def __init__(self, path="log.txt"):
        self.logger = logging.getLogger('BiRefNet')
        self.file_handler = logging.FileHandler(path, "w")
        self.stdout_handler = logging.StreamHandler()
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
    
    def info(self, txt):
        self.logger.info(txt)
    
    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, path, filename="latest.pth"):
    torch.save(state, os.path.join(path, filename))


def save_tensor_img(tenor_im, path):
    im = tenor_im.cpu().clone()
    im = im.squeeze(0)
    tensor2pil = transforms.ToPILImage()
    im = tensor2pil(im)
    im.save(path)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class SimpleMovingAverage:
    """Simple moving average for smoothing metrics"""
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.values = []

    def update(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return sum(self.values) / len(self.values)




########### LOGGING ###########

def checkGradientNorm(model):
    """Calculate the L2 norm of all gradients in the model."""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

from pathlib import Path
# List of validation images to send to comet ml
def is_sample_image(file_path):
    file_list = [
        '0ac4c733-e803-42ca-aa76-edc3aa0b108c1675064255462-ETC-Women-Night-suits-2691675064254924-5',
        '0AFDB91F-87C2-4C84-820D-5B3ACF4EC4DD',
        '0c8ec50f2b6a47d35a6a29643c9c3efe',
        '1_46bd718f-da4b-4b47-9266-760ccbc6fa5d',
        '2nfkX2',
        '9c270d46b1df98e677e9dd70a6899674',
        '86EFA7C0-7E46-4997-A301-7F337FCAA0AA',
        '404fcfe83e27c2b73b8024cb2d666a3b',
        '3352BLACK_1_8dcff332-5f70-4028-99f6-6c406d4d0c2d',
        '88924',
        'auto-watermark-test-black',
        'bague-tank-noeud-en-or-jaune-et-diamants-859747',
        'blue-triplet-opal-gemstone-handmade-wedding-jewelry-pendant-208-892810',
        'coussin-de-lecture-avec-accoudoirs-182',
        'dolce-gabbana-teen-girls-white-poppy-dress-486805-88440f42ca2ae53575237eeb2bfe084a92f5e283_nrunab_23b11f4e-95ed-4a5f-8b71-5be6f1cdd954',
        'EFE72F39-F54A-4636-B763-593D29EEFE95',
        'free-fly-icon-cap-ICCP-413-washed-navy-front_65086c596541e',
        'GT08',
        'gund-xavier-kitten-9in-6067463-front',
        'H47751bad62754aaca36ad5f1be48c15dn',
        'Hfc26b0880a4f40d49ae5745f7dcabf35L',
        'HTB1EiktXrj1gK0jSZFuq6ArHpXaE',
        'HTB1KyhbEv9TBuNjy1zbq6xpepXaU',
        'IMG_2201-5eff5ff1eff82',
        'IMG_10722',
        'IMG_20210125_093811_resized_20210125_094015285',
        'mc_BPJtWbJzi',
        'mc_Jig1lwHrJ',
        'mc_nqyoVAYt2',
        'new-era-food-burger-white-t-shirt-60416411-center',
        'top-focoss-fitness-cod089-ref089-focoss-747470',
        '2ceea7e7af5b4329823c0d967288dd8b-max-6738cb9642c28',
        '3b5f3cad39e336da560651a419f1e5c4',
        'HS_Charme4_Champagne_Rooted_2017',
        'h36e087bbc2e8499bb7322928bd2ee076p-66c3dbcc3bdb0',
        'Hdb05ea9bc5c14c559732575d30e71326o',
        'HTB1kAs3aEzrK1RjSspmq6AOdFXaW',
        'HTB1KtlOeBaE3KVjSZLeq6xsSFXau',
        'HTB1MjuKNhjaK1RjSZKzq6xVwXXaY',
        'HTB1rS8IeBWD3KVjSZFsq6AqkpXaC',
        'mc_af546738-253b-4627-bc03-5aec187ffdff',
        'mc_e3ce50a0-1bf6-4c94-b1bf-ed086da27e0a',
        'product-image-1712332648',
        'SweetTalk',
        'Zappelin-4',
        'mc__lvEkSwWx', # veryhard
        'S37acc11f67424e99a089818fa2b5d12bq', # veryhard
        'gxu3exsdtbq5-kgnxlkqhw-bmasmkw5eeii-x7sruokzg-90485-672c6d724d3fc', # veryhard
        'H45a9945126874501b99b16283bd66fc9r', # veryhard
        'gxu3exsdtbq5-kgnxlkqhw-iku18kuoeei45pmdm-vxlg-68457-672c6c3eba5b6', # veryhard
        '001-20201103-121614_121629', # perfect (prevent pizelation)
        '0-main-verao-casual-sem-mangas-oco-para-fora-macacao-66c3d77562e72',
        'clown-necklace-4',
        'heavy-bulldog-silver-bracelet-4',
        'mc_3p4sorken',
        'skull-sterling-silver-pen-5',
        'tropical-bag-in-ecru-beach-bags',
        'cuban-chain-necklace-3',
        '14A8322E-11A9-4B5B-AA05-468E83FEDFAB',
        'main_images__2_61b74325-aab0-4904-96ea-889b96368c0d'
    ]

    file_name = Path(file_path).name
    return file_name in file_list


########### METRICS ###########

import torch.nn as nn
import torch.nn.functional as F

def ssim_index(pred: torch.Tensor,
               target: torch.Tensor,
               window_size: int = 11,
               size_average: bool = True,
               C1: float = 0.01**2,
               C2: float = 0.03**2) -> torch.Tensor:
    """
    Evaluation metric: Calculates the Structural Similarity Index Measure (SSIM)
    using the same logic as the provided SSIMLoss class.
    Window creation logic is integrated into this function.

    Args:
        pred (torch.Tensor): Predicted probabilities after sigmoid ([B, C, H, W]).
                             Expected range 0.0 to 1.0.
        target (torch.Tensor): Target binary mask ([B, C, H, W]).
                               Expected range 0.0 to 1.0.
        window_size (int): Size of the Gaussian window for SSIM calculation.
        size_average (bool): If True, returns the mean SSIM score over the batch.
                             If False, returns the SSIM score for each image/channel.
        C1 (float): Constant for luminance stabilization.
        C2 (float): Constant for contrast stabilization.

    Returns:
        torch.Tensor: SSIM score. Higher values indicate greater similarity (1 is perfect).
                      If size_average=True, returns a scalar tensor.
                      If size_average=False, returns a tensor of shape [B*C].
    """
    # --- Input Validation ---
    if pred.shape != target.shape:
        raise ValueError(f"Input prediction and target must have the same shape. Got {pred.shape} and {target.shape}")
    if pred.dim() != 4 or target.dim() != 4:
         raise ValueError(f"Input prediction and target must be 4D tensors ([B, C, H, W]). Got {pred.dim()}D and {target.dim()}D")

    # --- Setup ---
    device = pred.device
    channel = pred.size(1) # Get the number of channels from the input tensor
    padding = window_size // 2

    # --- Create Gaussian Window (Integrated) ---
    sigma = 1.5
    _numpy_gauss = np.exp(-(np.arange(window_size) - window_size//2)**2 / float(2*sigma**2))
    gauss = torch.from_numpy(_numpy_gauss).float().to(device) # Create directly on the target device
    gauss = gauss / gauss.sum()

    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    # Expand window to match the number of input channels and move to device
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    # --- Window creation complete ---

    # --- SSIM Calculation (exactly matches SSIMLoss._ssim) ---
    # Calculate means
    mu1 = F.conv2d(pred, window, padding=padding, groups=channel)
    mu2 = F.conv2d(target, window, padding=padding, groups=channel)

    # Calculate squares of means
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Calculate variances and covariance
    sigma1_sq = F.conv2d(pred * pred, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=padding, groups=channel) - mu1_mu2

    # Calculate SSIM map
    # Using .clamp(min=0) on variances to prevent potential negative values due to floating point errors
    sigma1_sq = sigma1_sq.clamp(min=0)
    sigma2_sq = sigma2_sq.clamp(min=0)

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # --- Averaging ---
    # Calculate the mean SSIM for each image/channel in the batch
    # ssim_val has shape [B, C]
    ssim_val = ssim_map.mean(dim=[2, 3]) # Average over height and width

    if size_average:
        return ssim_val.mean() # Average over batch and channel dimensions -> scalar
    else:
        # Return SSIM per image/channel, flattened to match typical evaluation output formats
        return ssim_val.view(-1) # Return tensor of shape [B*C]

def jaccard_index(pred, target, smooth=1e-6):
    """
    Evaluation metric: returns value between 0 (no overlap) and 1 (perfect overlap)
    Should match 1 - calc_jaccard_loss
    Should already have sigmoid applied
    Args:
        pred: Predicted probabilities after sigmoid
        target: Target binary mask
        smooth: Smoothing factor to avoid numerical instability
    Returns:
        Tensor: Dice coefficient between 0 (worse) and 1 (better)
    """
    
    # Flatten predictions and targets
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    # Calculate intersection and union exactly like in loss function
    intersection = (pred_flat * target_flat).sum(dim=1) + smooth
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection + smooth
    
    # Calculate Jaccard index
    jaccard = intersection / union
    
    return jaccard.mean()  # Returns value between 0 (worse) and 1 (better)

def dice_index(pred, target, smooth=1e-6):
    """
    Evaluation metric: returns Dice coefficient (F1 score) between 0 (no overlap) and 1 (perfect overlap)
    Should match 1 - calc_dice_loss
    Should already have sigmoid applied
    Args:
        pred: Predicted probabilities after sigmoid
        target: Target binary mask
        smooth: Smoothing factor to avoid numerical instability
    Returns:
        Tensor: Dice coefficient between 0 (worse) and 1 (better)
    """
    # Flatten predictions and targets
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    # Calculate intersection
    intersection = (pred_flat * target_flat).sum(dim=1) + smooth
    
    # Calculate denominator (sum of both individual sums)
    denominator = pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth
    
    # Calculate Dice coefficient
    dice = 2 * intersection / denominator
    
    return dice.mean()  # Returns value between 0 (worse) and 1 (better)

def bce_index(pred, target):
    """
    Calculate similarity score between predicted and target masks
    Args:
        pred: Predicted mask (0-1 range, post-sigmoid)
        target: Ground truth mask (0-1 range)
    Returns:
        Score between 0-1 where 1 means perfect match
    """
    # Method 1: Direct BCE-based score
    eps = 1e-7
    pred = torch.clamp(pred, eps, 1 - eps)
    score = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    # Convert loss to similarity score (invert and normalize)
    score = 1 - (score.mean() / 16.0)  # or use torch.exp(-score.mean())
    
    # Or Method 2: Simpler similarity metrics
    # score = 1 - torch.abs(pred - target).mean()  # L1 similarity
    # score = 1 - ((pred - target) ** 2).mean()    # L2 similarity
    
    return score

def pixel_index(pred, target):
    """
    Calculate pixel-wise accuracy between predicted and target masks.
    Handles batch processing and returns mean accuracy.
    Should already have sigmoid applied
    Args:
        pred: Predicted mask (0-1 range, post-sigmoid)
        target: Ground truth mask (0-1 range)
    Returns:
        Mean pixel accuracy across the batch
    """
    # Threshold predictions
    pred = (pred > 0.001).float()
    target = (target > 0.001).float()
    
    # Calculate correct pixels per sample
    correct = (pred == target).float().sum((1, 2))
    total = float(pred.size(1) * pred.size(2))
    
    # Calculate accuracy and return mean
    accuracy = correct / total
    return accuracy.mean()  # Returns value between 0 (worse) and 1 (better)

