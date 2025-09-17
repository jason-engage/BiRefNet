import os
import datetime
from contextlib import nullcontext
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
import heapq
import math
from pathlib import Path
from colorama import init, Back, Fore, Style
import yaml
from tqdm import tqdm
if tuple(map(int, torch.__version__.split('+')[0].split(".")[:3])) >= (2, 5, 0):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from config import Config
from loss import PixLoss, ClsLoss
from dataset import MyData
from models.birefnet import BiRefNet, BiRefNetC2F
from utils import Logger, AverageMeter, set_seed, check_state_dict, SimpleMovingAverage, bce_index, dice_index, jaccard_index, ssim_index, pixel_index, is_sample_image, checkGradientNorm

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group


# ============== Load start ==============
init() # colorama init
config = Config() # load birefnet config

parser = argparse.ArgumentParser(description='')
parser.add_argument('--experiment_name', default='BiRefNet', type=str, help='Name for this experiment/training run')
parser.add_argument('--dist', default=False, type=lambda x: x == 'True')
parser.add_argument('--use_accelerate', action='store_true', help='`accelerate launch --multi_gpu train.py --use_accelerate`. Use accelerate for training, good for FP16/BF16/...')
args = parser.parse_args()

# ============== Load config variables ==============
try:
    with open('config_vars.yml', 'r') as f:
        config_vars = yaml.safe_load(f)
except FileNotFoundError:
    print("\n" + "="*60)
    print("ERROR: Configuration file not found!")
    print("="*60)
    print("\nTo set up your configuration:")
    print("1. Copy the template file:")
    print("   cp config_vars_template.yml config_vars.yml")
    print("\n2. Edit config_vars.yml with your settings:")
    print("   - Update paths to your pretrained models")
    print("   - Set your Comet ML API key (optional)")
    print("   - Adjust training parameters as needed")
    print("\n3. Run the training script again")
    print("="*60 + "\n")

    import sys
    sys.exit(1)

# ============== Initialize DDP ==============
to_be_distributed = args.dist
if to_be_distributed:
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*10))
    device = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main_process = (rank == 0)
else:
    device = config.device
    rank = 0
    world_size = 1
    is_main_process = True

# ============== Initialize Experiment + Comet ML experiment (optional) ==============
experiment_name = args.experiment_name
comet_experiment = None

if config_vars.get('comet_ml_enable', False) and is_main_process:
    try:
        from comet_ml import Experiment
        comet_experiment = Experiment(
            api_key=config_vars.get('comet_ml_api_key', 'YOUR_API_KEY'),
            project_name=config_vars.get('comet_ml_project_name', 'birefnet'),
            workspace=config_vars.get('comet_ml_workspace', 'YOUR_WORKSPACE'),
            auto_param_logging=False,
            auto_metric_logging=False,
        )
        # Update This Later
        # comet_experiment.set_name(f"BiRefNet_{config.task}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

        # comet_experiment.log_parameters({
        #     "model": config.model,
        #     "task": config.task,
        #     "batch_size": config.batch_size,
        #     "learning_rate": config.lr,
        #     "optimizer": config.optimizer,
        #     "dynamic_size": config.dynamic_size,
        #     "compile": config.compile,
        #     "mixed_precision": config.mixed_precision if args.use_accelerate else "none",
        # })

        experiment_name = comet_experiment.get_name().rsplit('_', 1)[0] # Overwrites experiment_name
        comet_experiment_epoch = 0 # Will be incremented before training
        
        print("=" * 60)
        print(f"Comet ML Experiment Name: {experiment_name}")
        print("=" * 60)
        print()

		# Logs code from model and processing scripts
        comet_experiment.log_code(file_name="train.py")
        comet_experiment.log_code(file_name="config.py")
        comet_experiment.log_code(file_name="dataset.py")
        comet_experiment.log_code(file_name="loss.py")
        comet_experiment.log_code(file_name="config_vars.yml")

    except ImportError:

        print("=" * 60)
        print("Comet ML not installed. Proceeding without Comet logging.")
        print("=" * 60)
        print()

# ============== Set checkpoint directory ==============
# Combine base checkpoint directory with experiment name
ckpt_dir = os.path.join(config_vars.get('ckpt_dir', 'ckpt'), experiment_name)
os.makedirs(ckpt_dir, exist_ok=True)

# ============== Init log file ==============
logger = Logger(os.path.join(ckpt_dir, "log.txt")) if is_main_process else None
logger_loss_idx = 1

# ============== Initialize Accelerate (optional) ==============
if args.use_accelerate:
    from accelerate import Accelerator, utils
    mixed_precision = config.mixed_precision
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=1,
        kwargs_handlers=[
            utils.InitProcessGroupKwargs(backend="nccl", timeout=datetime.timedelta(seconds=3600*10)),
            utils.DistributedDataParallelKwargs(find_unused_parameters=False),
            utils.GradScalerKwargs(backoff_factor=0.5)],
    )
    args.dist = False

# DDP already initialized above, just update device for accelerate if needed
if not to_be_distributed and args.use_accelerate:
    device = accelerator.local_process_index


# ============== Resume Training ==============
epoch_st = config_vars.get('epoch_st', 1)  # Default starting epoch from config
epochs_end = config_vars.get('epochs_end', 1) # Total Epochs

resume_weights_path = config_vars.get('resume_weights_path', None)

if resume_weights_path:
    if os.path.isfile(resume_weights_path):
        # Try to extract epoch number from checkpoint filename (e.g., 'epoch_99.pth' -> 100)
        if 'epoch_' in resume_weights_path and resume_weights_path.endswith('.pth'):
            try:
                epoch_st = int(resume_weights_path.rstrip('.pth').split('epoch_')[-1]) + 1
                if logger:
                    logger.info("-" * 60)
                    logger.info(f"Resuming from epoch {epoch_st} based on checkpoint filename")
                    logger.info("-" * 60)
            except ValueError:
                if logger:
                    logger.info(f"Could not extract epoch from filename, using epoch_st={epoch_st} from config")
    else:
        if logger:
            logger.info("=" * 60)
            logger.info(f"Resume checkpoint '{resume_weights_path}' not found. Starting from scratch.")
            logger.info("=" * 60)
        resume_weights_path = None  # Reset to None if file doesn't exist
        

# ============== Set random seed for reproducibility ==============
if config.rand_seed:
    set_seed(config.rand_seed + device)


# ============== Log model and optimizer params ==============
if is_main_process:
    # logger.info("Model details:"); logger.info(model)
    # if args.use_accelerate and accelerator.mixed_precision != 'no':
    #     config.compile = False
    # logger.info("datasets: load_all={}, compile={}.".format(config.load_all, config.compile))
    logger.info(f"{'='*80}")
    logger.info("Hyperparameters:")
    logger.info(f"{'='*80}")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    for key, value in config_vars.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"  batch size: {config.batch_size}")
    logger.info(f"  experiment name: {experiment_name}")

from dataset import custom_collate_fn, custom_collate_resize_fn


def prepare_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, to_be_distributed=False, is_train=True):
    # Prepare dataloaders
    sampler = None

    # Handle num_samples limitation for training
    if is_train and config_vars.get('max_samples', 0) > 0:
        desired_samples = config_vars.get('max_samples', 0)
        total_size = len(dataset)

        # Round to nearest multiple of batch_size for even batches
        num_samples = (desired_samples // batch_size) * batch_size

        # If we rounded down to 0, use at least one full batch
        if num_samples == 0:
            num_samples = batch_size

        # Adjust for multi-GPU if using DDP
        if to_be_distributed:
            world_size = dist.get_world_size()
            # For DDP, we want total of 'desired_samples' across all GPUs
            # So don't divide by world_size here - DistributedSampler will handle the split
            indices = list(range(min(desired_samples, total_size)))
            # Create subset dataset
            dataset = torch.utils.data.Subset(dataset, indices)
            # Let DistributedSampler handle the per-GPU splitting
            sampler = DistributedSampler(dataset)
        else:
            # Single GPU - just limit the indices
            indices = list(range(min(num_samples, total_size)))
            sampler = torch.utils.data.SubsetRandomSampler(indices) if is_train else None
            if not is_train:
                dataset = torch.utils.data.Subset(dataset, indices)

        # Debug print to verify setup
        if is_main_process:
            logger.info(f"{'='*60}")
            logger.info("📊 SAMPLING CONFIGURATION")
            logger.info(f"{'='*60}")
            logger.info(f"  Total available samples:  {total_size:,}")
            logger.info(f"  Desired samples:          {desired_samples:,}")
            if to_be_distributed:
                samples_per_gpu = len(indices) // dist.get_world_size()
                logger.info(f"  Adjusted total samples:   {len(indices):,}")
                logger.info(f"  Samples per GPU (approx): {samples_per_gpu:,}")
            else:
                logger.info(f"  Samples to use:           {len(indices):,}")
            logger.info(f"  Batch size:               {batch_size}")
            logger.info(f"{'='*60}\n")
    elif to_be_distributed:
        sampler = DistributedSampler(dataset)

    if to_be_distributed:
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=min(config.num_workers, batch_size), pin_memory=True,
            shuffle=False, sampler=sampler, drop_last=True, collate_fn=custom_collate_resize_fn if is_train and config.dynamic_size else None
        )
    else:
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=min(config.num_workers, batch_size), pin_memory=True,
            shuffle=(is_train and sampler is None), sampler=sampler, drop_last=True, collate_fn=custom_collate_resize_fn if is_train and config.dynamic_size else None
        )


def init_data_loaders(to_be_distributed):
    # Prepare datasets
    # FIX for Accelerate + DDP: When using Accelerate, we must NOT use DistributedSampler
    # because Accelerate will handle data distribution internally via accelerator.prepare().
    # Using both would cause double distribution (dividing data by world_size twice).
    # - If using Accelerate: to_be_distributed should be False (let Accelerate handle it)
    # - If using pure DDP: to_be_distributed should be True (use DistributedSampler)
    should_use_distributed_sampler = to_be_distributed and not args.use_accelerate

    train_loader = prepare_dataloader(
        MyData(datasets=config.training_set, data_size=None if config.dynamic_size else config.size, is_train=True),
        config.batch_size, to_be_distributed=should_use_distributed_sampler, is_train=True
    )
    if is_main_process:
        logger.info("{} batches of train dataloader {} have been created.".format(len(train_loader), config.training_set))

    # Prepare validation loader if needed
    val_loader = None
    if config_vars.get('eval_each_epoch', 0) > 0 and config.testsets:
        # Validation should NEVER be distributed - each GPU processes full dataset
        # This applies regardless of whether we're using Accelerate or pure DDP
        val_loader = prepare_dataloader(
            MyData(datasets=config.testsets.replace(',', '+'), data_size=config.size, is_train=False),
            config.batch_size_valid, to_be_distributed=False, is_train=False
        )
        if is_main_process:
            logger.info("{} batches of validation dataloader {} have been created.".format(len(val_loader), config.testsets))

    return train_loader, val_loader


def init_models_optimizers(epochs, to_be_distributed):
    # Init models
    # Only use pretrained backbone if NOT resuming from checkpoint
    use_bb_pretrained = (resume_weights_path is None)

    if config.model == 'BiRefNet':
        model = BiRefNet(bb_pretrained=use_bb_pretrained)
    elif config.model == 'BiRefNetC2F':
        model = BiRefNetC2F(bb_pretrained=use_bb_pretrained)

    # Load checkpoint weights if resuming
    if resume_weights_path:
        if is_main_process:
            logger.info("Loading checkpoint '{}'".format(resume_weights_path))

        # Check if it's a .bin file (HuggingFace format)
        if resume_weights_path.endswith('.bin'):
            # .bin files are typically just the model weights without wrapper
            state_dict = torch.load(resume_weights_path, map_location='cpu', weights_only=True)
            # .bin files usually don't need the check_state_dict processing
            # They're already in the correct format
            model.load_state_dict(state_dict, strict=False)  # strict=False in case of minor mismatches
            if is_main_process:
                logger.info("Loaded .bin weights file successfully")
        else:
            # Standard .pth checkpoint
            state_dict = torch.load(resume_weights_path, map_location='cpu', weights_only=True)
            state_dict = check_state_dict(state_dict)
            model.load_state_dict(state_dict)
    if not args.use_accelerate:
        if to_be_distributed:
            model = model.to(device)
            model = DDP(model, device_ids=[device])
        else:
            model = model.to(device)
    if config.compile:
        model = torch.compile(model, mode=['default', 'reduce-overhead', 'max-autotune'][0])
    if config.precisionHigh:
        torch.set_float32_matmul_precision('high')

    # ============== OPTIMIZER ==============

    # Scale learning rate based on batch size and number of GPUs
    # Original config.py had: sqrt(batch_size/4) scaling
    batch_scale = math.sqrt(config.batch_size / 4)

    # Scale for multi-GPU training
    if to_be_distributed:
        gpu_scale = math.sqrt(world_size)
    else:
        gpu_scale = 1.0

    # Apply both scaling factors
    scaled_lr = config.lr * batch_scale * gpu_scale

    # Log the scaling factors for debugging
    if is_main_process:
        logger.info(f"LR Scaling: base_lr={config.lr:.2e}, batch_scale={batch_scale:.3f}, gpu_scale={gpu_scale:.3f}, final_lr={scaled_lr:.2e}")

    if config.optimizer == 'AdamW':
        optimizer = optim.AdamW(params=model.parameters(), lr=scaled_lr, weight_decay=1e-2)
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(params=model.parameters(), lr=scaled_lr, weight_decay=0)
    elif config.optimizer == 'Ranger':
        import torch_optimizer as optim_extra
        optimizer = optim_extra.Ranger(params=model.parameters(), lr=scaled_lr, weight_decay=1e-2)
	
	# ============== LR SCHEDULER ==============
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[lde if lde > 0 else epochs + lde + 1 for lde in config.lr_decay_epochs],
        gamma=config.lr_decay_rate
    )

    # Fast-forward scheduler if resuming training
    if resume_weights_path and epoch_st > 1 and config_vars.get('fast_forward_scheduler', True):
        if is_main_process:
            logger.info(f"Fast-forwarding LR scheduler to epoch {epoch_st-1}")
        for _ in range(epoch_st - 1):
            lr_scheduler.step()

    # logger.info("Optimizer details:"); logger.info(optimizer)

    return model, optimizer, lr_scheduler


class Trainer:
    def __init__(
        self, data_loaders, model_opt_lrsch,
    ):
        """
        Initialize the trainer with comprehensive metrics tracking
        """
        # ============== Model, Optimizer, and Scheduler ==============
        self.model, self.optimizer, self.lr_scheduler = model_opt_lrsch
        self.train_loader, self.val_loader = data_loaders

        # Accelerate integration
        if args.use_accelerate:
            if self.val_loader is not None:
                self.train_loader, self.val_loader, self.model, self.optimizer = accelerator.prepare(
                    self.train_loader, self.val_loader, self.model, self.optimizer
                )
            else:
                self.train_loader, self.model, self.optimizer = accelerator.prepare(self.train_loader, self.model, self.optimizer)

        # ============== Loss Functions ==============
        if config.out_ref:
            self.criterion_gdt = nn.BCELoss()
        self.pix_loss = PixLoss()
        self.cls_loss = ClsLoss()

        # ============== Metrics Tracking ==============
        # Loss tracking
        self.loss_log = AverageMeter()
        self.train_loss_total = 0.0
        self.train_loss_batch = 0.0

        # Individual loss components tracking
        self.train_bce_total = 0.0
        self.train_dice_total = 0.0
        self.train_jaccard_total = 0.0
        self.train_ssim_total = 0.0
        self.train_pixel_total = 0.0

        # Moving averages for smoothing
        self.bce_moving_avg = SimpleMovingAverage(window_size=8)

        # Performance counters (for colored display)
        self.red_count = 0  # Poor performance (Jaccard < 0.95)
        self.green_count = 0  # Good performance (BCE > 0.999)
        self.yellow_count = 0  # Excellent performance (BCE > 0.999 and Dice > 0.985)
        self.blue_count = 0  # Default/normal performance

        # Batch tracking
        self.logged_steps_count = 0
        self.current_epoch = 0

        # Batch composition tracking
        self.epoch_composition_totals = {}
        self.current_batch_composition = {}

        # Timing
        self.epoch_start_time = None

	# ============================================================
	# ***************** TRAIN A BATCH **********************
	# ============================================================

    def _train_batch(self, batch, batch_idx):
        """
        Train a single batch with comprehensive metrics calculation
        """
        # ============== Data Preparation ==============
        if args.use_accelerate:
            inputs = batch[0]
            gts = batch[1]
            class_labels = batch[2]
            paths = batch[3]  # Now includes paths
        else:
            inputs = batch[0].to(device)
            gts = batch[1].to(device)
            class_labels = batch[2].to(device)
            paths = batch[3]  # Now includes paths

        # ============== Get Batch Composition ==============
        batch_composition = self.train_loader.dataset.get_batch_composition(paths)
        self.current_batch_composition = batch_composition  # Store for display

        # Accumulate composition totals for the epoch
        for dataset_name, count in batch_composition.items():
            self.epoch_composition_totals[dataset_name] = self.epoch_composition_totals.get(dataset_name, 0) + count

        # ============== Forward Pass ==============
        self.optimizer.zero_grad()
        scaled_preds, class_preds_lst = self.model(inputs)

		# CLASS LABEL PREDICTIONS IS DISABLED
        # # Debug: Check what class predictions look like
        # print(f"\n[DEBUG] class_preds_lst type: {type(class_preds_lst)}")
        # if isinstance(class_preds_lst, list):
        #     print(f"[DEBUG] class_preds_lst length: {len(class_preds_lst)}")
        #     if len(class_preds_lst) > 0:
        #         print(f"[DEBUG] First element: {class_preds_lst[0]}")
        #         if class_preds_lst[0] is not None:
        #             print(f"[DEBUG] First element shape: {class_preds_lst[0].shape if hasattr(class_preds_lst[0], 'shape') else 'N/A'}")
        # print(f"[DEBUG] class_labels: {class_labels}\n")

        # Handle reference output if enabled
        if config.out_ref:
            (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds
            for _idx, (_gdt_pred, _gdt_label) in enumerate(zip(outs_gdt_pred, outs_gdt_label)):
                _gdt_pred = nn.functional.interpolate(_gdt_pred, size=_gdt_label.shape[2:], mode='bilinear', align_corners=True).sigmoid()
                _gdt_label = _gdt_label.sigmoid()
                loss_gdt = self.criterion_gdt(_gdt_pred, _gdt_label) if _idx == 0 else self.criterion_gdt(_gdt_pred, _gdt_label) + loss_gdt
            self.loss_dict['loss_gdt'] = loss_gdt.item()

        # ============== Loss Calculation ==============
        # Classification loss
        if None in class_preds_lst:
            loss_cls = 0.
        else:
            loss_cls = self.cls_loss(class_preds_lst, class_labels)
            self.loss_dict['loss_cls'] = loss_cls.item()

        # Pixel loss (main loss)
        loss_pix, loss_dict_pix = self.pix_loss(scaled_preds, torch.clamp(gts, 0, 1), pix_loss_lambda=1.0)
        self.loss_dict.update(loss_dict_pix)
        self.loss_dict['loss_pix'] = loss_pix.item()

        # Total loss
        loss = loss_pix + loss_cls
        if config.out_ref:
            loss = loss + loss_gdt * 1.0

        # Track batch loss
        self.train_loss_batch = loss.item()
        self.train_loss_total += self.train_loss_batch
        self.loss_log.update(loss.item(), inputs.size(0))

        # ============== Calculate Metrics (every log_frequency batches) ==============
        should_calculate_metrics = (batch_idx % config_vars.get('log_frequency', 100) == 0 or batch_idx == len(self.train_loader) - 1)

        if should_calculate_metrics:
            with torch.no_grad():
                # Get the main prediction (highest resolution)
                if isinstance(scaled_preds, (list, tuple)):
                    main_pred = scaled_preds[-1] if len(scaled_preds) > 0 else scaled_preds[0]
                else:
                    main_pred = scaled_preds

                # Apply sigmoid to convert logits to probabilities for metrics
                main_pred = torch.sigmoid(main_pred)

                # Calculate metrics
                current_bce = bce_index(main_pred, gts).item()
                current_dice = dice_index(main_pred, gts).item()
                current_jaccard = jaccard_index(main_pred, gts).item()
                current_ssim = ssim_index(main_pred, gts).item()
                current_pixel = pixel_index(main_pred, gts).item()

                # Update totals
                self.train_bce_total += current_bce
                self.train_dice_total += current_dice
                self.train_jaccard_total += current_jaccard
                self.train_ssim_total += current_ssim
                self.train_pixel_total += current_pixel
                self.logged_steps_count += 1

                # Determine performance color
                if current_jaccard < 0.95:
                    self.back_color = Back.RED
                    self.red_count += 1
                elif current_bce > 0.999:
                    if current_dice > 0.985:
                        self.back_color = Back.CYAN
                        self.yellow_count += 1
                        self.green_count += 1
                    else:
                        self.back_color = Back.GREEN
                        self.green_count += 1
                else:
                    self.back_color = Back.BLUE
                    self.blue_count += 1

                # Store current metrics for display
                self.current_metrics = {
                    'bce': current_bce,
                    'dice': current_dice,
                    'jaccard': current_jaccard,
                    'ssim': current_ssim,
                    'pixel': current_pixel
                }

                # Update moving average
                self.bce_moving_avg.update(current_bce)

        # ============== Backward Pass ==============
        if args.use_accelerate:
            loss = loss / accelerator.gradient_accumulation_steps
            accelerator.backward(loss)
        else:
            loss.backward()

        # Calculate gradient norm before optimizer step (only when logging)
        grad_norm = 0.0
        if should_calculate_metrics:
            grad_norm = checkGradientNorm(self.model)
            # Check for NaN and handle it
            if not torch.isnan(torch.tensor(grad_norm)):
                self.grad_norm_total += grad_norm
            else:
                grad_norm = 0.0  # Set to 0 if NaN

        self.optimizer.step()

	# ============================================================
	# ***************** TRAIN AN EPOCH **********************
	# ============================================================
    def train_epoch(self, epoch):
        """
        Train for one epoch with comprehensive metrics display
        """
        global logger_loss_idx, comet_experiment
        self.model.train()
        self.loss_dict = {}
        self.current_epoch = epoch
        self.back_color = Back.RESET
        self.current_metrics = {}

        # ============== Reset Epoch Metrics ==============
        self.train_loss_total = 0.0
        self.train_bce_total = 0.0
        self.train_dice_total = 0.0
        self.train_jaccard_total = 0.0
        self.train_ssim_total = 0.0
        self.train_pixel_total = 0.0
        self.logged_steps_count = 0
        self.red_count = 0
        self.green_count = 0
        self.yellow_count = 0
        self.blue_count = 0
        self.epoch_composition_totals = {}  # Reset composition counts
        self.grad_norm_total = 0.0  # Track gradient norm

        # ============== Set to prevent potential errors ==============
        train_bce_avg = 0
        train_dice_avg = 0
        train_jaccard_avg = 0
        train_ssim_avg = 0
        train_pixel_avg = 0
        train_grad_norm_avg = 0

        # Start timing
        self.epoch_start_time = time.time()
        total_batches = len(self.train_loader)

        # ============== Adjust Loss Weights for Fine-tuning ==============
        if epoch > epochs_end + config.finetune_last_epochs:
            if config.task == 'Matting':
                self.pix_loss.lambdas_pix_last['mae'] *= 1
                self.pix_loss.lambdas_pix_last['mse'] *= 0.9
                self.pix_loss.lambdas_pix_last['ssim'] *= 0.9
            else:
                self.pix_loss.lambdas_pix_last['bce'] *= 0
                self.pix_loss.lambdas_pix_last['ssim'] *= 1
                self.pix_loss.lambdas_pix_last['iou'] *= 0.5
                self.pix_loss.lambdas_pix_last['mae'] *= 0.9

        # ============== Training Loop ==============
        if is_main_process:
            logger.info(f"{'='*80}")
            logger.info(f"Starting Epoch {epoch}/{epochs_end} - Total batches: {total_batches}")
            logger.info(f"{'='*80}\n")

        for batch_idx, batch in enumerate(self.train_loader):
            current_batch = batch_idx + 1

            # Train batch
            self._train_batch(batch, batch_idx)

            # ============== Display Comprehensive Metrics ==============
            should_display = (batch_idx % config_vars.get('log_frequency', 100) == 0 or batch_idx == total_batches - 1)

            if should_display and self.logged_steps_count > 0:
                # Calculate averages
                train_loss_avg = self.train_loss_total / current_batch
                train_bce_avg = self.train_bce_total / self.logged_steps_count
                train_dice_avg = self.train_dice_total / self.logged_steps_count
                train_jaccard_avg = self.train_jaccard_total / self.logged_steps_count
                train_ssim_avg = self.train_ssim_total / self.logged_steps_count
                train_pixel_avg = self.train_pixel_total / self.logged_steps_count
                train_grad_norm_avg = self.grad_norm_total / self.logged_steps_count if self.logged_steps_count > 0 else 0.0

                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                current_momentum = self.optimizer.param_groups[0].get('betas', [0.9])[0]

                # Calculate timing
                time_per_iter = (time.time() - self.epoch_start_time) / current_batch
                remaining_iters = total_batches - current_batch
                eta_seconds = remaining_iters * time_per_iter
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                # Get current metrics (batch metrics)
                curr_bce = self.current_metrics.get('bce', 0)
                curr_dice = self.current_metrics.get('dice', 0)
                curr_jaccard = self.current_metrics.get('jaccard', 0)
                curr_ssim = self.current_metrics.get('ssim', 0)

                # Format batch composition for display (batch/total)
                batch_comp_str = ""
                total_comp_str = ""
                # Use original dataset order from config
                dataset_order = self.train_loader.dataset.dataset_names

                if self.current_batch_composition:
                    batch_counts = [str(self.current_batch_composition.get(k, 0)) for k in dataset_order]
                    batch_comp_str = "\\".join(batch_counts)

                if self.epoch_composition_totals:
                    total_counts = [str(self.epoch_composition_totals.get(k, 0)) for k in dataset_order]
                    total_comp_str = "\\".join(total_counts)
                    # Also add dataset names for clarity on first display
                    if batch_idx == 0 or batch_idx % 50 == 0:
                        dataset_names = "|".join(dataset_order)
                        # print(f"  Dataset order: {dataset_names}")

                # ============== Comprehensive Display Output ==============
                display_msg = (
                    f"{self.back_color}E{epoch} - {current_batch}/{total_batches} - "
                    f"L: {train_loss_avg:.3f}, LB: {self.train_loss_batch:.3f}, "
                    f"BC: {train_bce_avg:.3f}, BCB: {curr_bce:.3f}, "
                    f"DI: {train_dice_avg:.3f}, DIB: {curr_dice:.3f}, "
                    f"JI: {train_jaccard_avg:.3f}, JIB: {curr_jaccard:.3f}, "
                    f"SI: {train_ssim_avg:.3f}, SIB: {curr_ssim:.3f}, "
                    f"LR: {current_lr:.7f}, M: {current_momentum:.2f}, GN: {train_grad_norm_avg:.1f}, "
                    f"BC: {batch_comp_str}, TBC: {total_comp_str}, "
                    f"It/s: {(1/time_per_iter):.1f}, ETA: {eta}{Back.RESET}"
                )
                if is_main_process and logger:
                    logger.info(display_msg)

                # ============== Log to Comet ML ==============
                if comet_experiment is not None:
                    current_step = (epoch - 1) * total_batches + current_batch
                    comet_experiment.set_step(current_step)

                    # Log batch metrics
                    comet_experiment.log_metric("loss_batch", self.train_loss_batch)

                    # Log individual metrics
                    comet_experiment.log_metric("bce_batch", curr_bce)
                    comet_experiment.log_metric("dice_batch", curr_dice)
                    comet_experiment.log_metric("jaccard_batch", curr_jaccard)
                    comet_experiment.log_metric("ssim_batch", curr_ssim)

                    # Log training parameters
                    comet_experiment.log_metric("learning_rate", current_lr)
                    comet_experiment.log_metric("momentum", current_momentum)
                    
                    # Log loss components
                    # for loss_name, loss_value in self.loss_dict.items():
                    #     comet_experiment.log_metric(loss_name, loss_value)

            # ============== Original Logger Output (less frequent) ==============
            # if (epoch < 2 and batch_idx < 100 and batch_idx % 20 == 0) or batch_idx % max(100, len(self.train_loader) // 10) == 0:
            #     info_progress = f'Epoch[{epoch}/{epochs_end}] Iter[{batch_idx}/{len(self.train_loader)}].'
            #     info_loss = 'Training Losses:'
            #     for loss_name, loss_value in self.loss_dict.items():
            #         info_loss += f' {loss_name}: {loss_value:.5g} |'
            #     if is_main_process:
            #         logger.info(' '.join((info_progress, info_loss)))

        # ============== Epoch Summary ==============
        epoch_time = time.time() - self.epoch_start_time
        final_loss_avg = self.train_loss_total / total_batches if total_batches > 0 else 0

        # Print summary
        if is_main_process:
            logger.info(f"{'='*80}")
            logger.info(f"Epoch {epoch} Summary:")
            logger.info(f"  Final Loss: {final_loss_avg:.5g}")
            if self.logged_steps_count > 0:
                logger.info(f"  BCE: {self.train_bce_total/self.logged_steps_count:.4f}, "
                           f"Dice: {self.train_dice_total/self.logged_steps_count:.4f}, "
                           f"Jaccard: {self.train_jaccard_total/self.logged_steps_count:.4f}, "
                           f"SSIM: {self.train_ssim_total/self.logged_steps_count:.4f}")
            logger.info(f"  Performance Distribution - Red: {self.red_count}, Green: {self.green_count}, Yellow: {self.yellow_count}, Blue: {self.blue_count}")
            if self.epoch_composition_totals:
                comp_summary = ", ".join([f"{k}: {v}" for k, v in sorted(self.epoch_composition_totals.items())])
                logger.info(f"  Dataset Composition: {comp_summary}")
            logger.info(f"  Epoch Time: {datetime.timedelta(seconds=int(epoch_time))}")
            logger.info(f"{'='*80}\n")

            # Log epoch summary to Comet
            if comet_experiment is not None:
                comet_experiment.log_metric("loss", final_loss_avg)
                comet_experiment.log_metric("time", epoch_time)
                comet_experiment.log_metric("bce", train_bce_avg)
                comet_experiment.log_metric("dice", train_dice_avg)
                comet_experiment.log_metric("jaccard", train_jaccard_avg)
                comet_experiment.log_metric("ssim", train_ssim_avg)
                comet_experiment.log_metric("pixel", train_pixel_avg)
                comet_experiment.log_metric("grad_norm", train_grad_norm_avg)
                
                comet_experiment.log_metric("bce_high", self.green_count) # Log final counts
                comet_experiment.log_metric("dice_high", self.yellow_count)
                comet_experiment.log_metric("jaccard_low", self.red_count)
                
        self.lr_scheduler.step()
        return self.loss_log.avg

	# ============================================================
	# ***************** VALIDATE THE EPOCH **********************
	# ============================================================
    def validate_epoch(self, epoch):
        """
        Run validation on the validation set
        """
        if self.val_loader is None:
            return None

        global comet_experiment
        self.model.eval()

        # Validation metrics
        val_loss_total = 0.0
        val_bce_total = 0.0
        val_dice_total = 0.0
        val_jaccard_total = 0.0
        val_ssim_total = 0.0
        val_pixel_total = 0.0
        val_batch_count = 0

        # Performance tracking
        worst_items_heap = []  # Min heap for worst items (negated jaccard)
        best_items_heap = []   # Max heap for best items
        jaccard_buckets = {i/20: 0 for i in range(20)}  # Buckets: [0.0-0.05], [0.05-0.10], ..., [0.95-1.0]

        if is_main_process:
            logger.info(f"Running validation for epoch {epoch}...")

        # Initialize progress bar
        pbar = None if not is_main_process else tqdm(total=len(self.val_loader), desc=f"Validation Epoch {epoch}")

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Set Comet ML step
                if comet_experiment is not None:
                    current_batch = batch_idx + 1
                    total_batches = len(self.val_loader)
                    current_step = (epoch - 1) * total_batches + current_batch
                    comet_experiment.set_step(current_step)

                # Data preparation
                if args.use_accelerate:
                    inputs = batch[0]
                    gts = batch[1]
                    paths = batch[2]  # Validation returns paths as 3rd element
                else:
                    inputs = batch[0].to(device)
                    gts = batch[1].to(device)
                    paths = batch[2]  # Validation returns paths as 3rd element

                # Forward pass
                scaled_preds = self.model(inputs)[-1]

                # Note: config.out_ref only applies in training mode, not in eval mode

                # Calculate loss - wrap single prediction in list for pix_loss
                loss_pix, _ = self.pix_loss([scaled_preds], torch.clamp(gts, 0, 1), pix_loss_lambda=1.0)

                # Get the main prediction
                if isinstance(scaled_preds, (list, tuple)):
                    main_pred = scaled_preds[-1] if len(scaled_preds) > 0 else scaled_preds[0]
                else:
                    main_pred = scaled_preds

                # Apply sigmoid to convert logits to probabilities for metrics
                main_pred = torch.sigmoid(main_pred)

                # Calculate metrics
                val_loss_total += loss_pix.item()
                val_bce_total += bce_index(main_pred, gts).item()
                val_dice_total += dice_index(main_pred, gts).item()
                current_jaccard = jaccard_index(main_pred, gts).item()
                val_jaccard_total += current_jaccard
                val_ssim_total += ssim_index(main_pred, gts).item()
                val_pixel_total += pixel_index(main_pred, gts).item()
                val_batch_count += 1

				# ******************* Track Best/Worst Performers *********************
                file_path = paths if not isinstance(paths, list) else paths[0]
                file_name = Path(file_path).name

                # Update worst items (using negative jaccard for min heap)
                heapq.heappush(worst_items_heap, (-current_jaccard, file_name))
                if len(worst_items_heap) > 10:
                    heapq.heappop(worst_items_heap)

                # Update best items
                heapq.heappush(best_items_heap, (current_jaccard, file_name))
                if len(best_items_heap) > 10:
                    heapq.heappop(best_items_heap)

                # Update Jaccard buckets
                # Clamp to [0, 0.95] range - 1.0 goes into 0.95 bucket
                bucket = min(0.95, math.floor(current_jaccard * 20) / 20)
                jaccard_buckets[bucket] += 1

                # ******************* LOG SAMPLE IMAGES *********************
                # Log specific sample images to Comet ML for visual tracking
                if comet_experiment is not None and is_main_process:
                    # Get the file path (handle both single path and list of paths)
                    file_path = paths if not isinstance(paths, list) else paths[0]

                    # Check if this image is in our sample list
                    if is_sample_image(Path(file_path).stem):
                        # On first epoch, log the ground truth
                        if epoch == 1 and config_vars.get('resume_start_with_eval', False) is False:
                            comet_experiment.log_image(
                                gts.squeeze().cpu(),
                                name=f"{Path(file_path).stem[:70]}.png",
                                step=comet_experiment_epoch,
                                image_format="png"
                            )
                        else:
                            # Log the prediction for all other epochs
                            comet_experiment.log_image(
                                main_pred.squeeze().cpu(),
                                name=f"{Path(file_path).stem[:70]}.png",
                                step=comet_experiment_epoch,
                                image_format="png"
                            )

                # Update progress bar
                if pbar is not None:
                    # Calculate running averages
                    val_loss_running_avg = val_loss_total / val_batch_count if val_batch_count > 0 else 0
                    val_bce_running_avg = val_bce_total / val_batch_count if val_batch_count > 0 else 0
                    val_dice_running_avg = val_dice_total / val_batch_count if val_batch_count > 0 else 0
                    val_jaccard_running_avg = val_jaccard_total / val_batch_count if val_batch_count > 0 else 0
                    val_ssim_running_avg = val_ssim_total / val_batch_count if val_batch_count > 0 else 0
                    val_pixel_running_avg = val_pixel_total / val_batch_count if val_batch_count > 0 else 0

                    pbar.update(1)
                    pbar.set_postfix({
                        "LOSS": f"{val_loss_running_avg:.4f}",
                        "BCE": f"{val_bce_running_avg:.4f}",
                        "DI": f"{val_dice_running_avg:.4f}",
                        "JI": f"{val_jaccard_running_avg:.4f}",
                        "SI": f"{val_ssim_running_avg:.4f}",
                        "PA": f"{val_pixel_running_avg:.4f}"
                    })

        # Close progress bar
        if pbar is not None:
            pbar.close()

        # Calculate averages
        val_loss_avg = val_loss_total / val_batch_count
        val_bce_avg = val_bce_total / val_batch_count
        val_dice_avg = val_dice_total / val_batch_count
        val_jaccard_avg = val_jaccard_total / val_batch_count
        val_ssim_avg = val_ssim_total / val_batch_count
        val_pixel_avg = val_pixel_total / val_batch_count

        # ***************** VALIDATION RESULTS **********************
        if is_main_process:
            logger.info(f"{'='*80}")
            logger.info(f"VALIDATION RESULTS - EPOCH {epoch}")
            logger.info(f"{'='*80}\n")

            # Overall metrics
            logger.info(f"📊 OVERALL METRICS:")
            logger.info(f"{'-'*40}")
            logger.info(f"  Loss:         {val_loss_avg:.5g}")
            logger.info(f"  BCE:          {val_bce_avg:.4f}")
            logger.info(f"  Dice:         {val_dice_avg:.4f}")
            logger.info(f"  Jaccard (IoU): {val_jaccard_avg:.4f}")
            logger.info(f"  SSIM:         {val_ssim_avg:.4f}")
            logger.info(f"  Pixel Acc:    {val_pixel_avg:.4f}")
            logger.info(f"{'='*80}\n")

            # Worst performing items
            logger.info(f"❌ 10 WORST PERFORMING ITEMS (by Jaccard):")
            logger.info(f"{'-'*40}")
            worst_items_sorted = sorted(worst_items_heap, key=lambda x: x[0], reverse=True)
            for j_index, fn in worst_items_sorted[:10]:
                logger.info(f"  {fn[:50]:50} | Jaccard: {-j_index:.4f}")
            logger.info(f"{'='*80}\n")

            # Best performing items
            logger.info(f"✅ 10 BEST PERFORMING ITEMS (by Jaccard):")
            logger.info(f"{'-'*40}")
            best_items_sorted = sorted(best_items_heap, key=lambda x: x[0], reverse=True)
            for j_index, fn in best_items_sorted[:10]:
                logger.info(f"  {fn[:50]:50} | Jaccard: {j_index:.4f}")
            logger.info(f"{'='*80}\n")

            # Jaccard distribution
            logger.info(f"📈 JACCARD DISTRIBUTION:")
            logger.info(f"{'-'*40}")
            for bucket in sorted(jaccard_buckets.keys()):
                count = jaccard_buckets[bucket]
                if count > 0:  # Only show buckets with items
                    bar = '█' * min(50, int(count * 50 / max(jaccard_buckets.values())))
                    logger.info(f"  [{bucket:.2f}-{min(1.0, bucket + 0.05):.2f}]: {bar} {count}")
            logger.info(f"{'='*80}\n")

            # Log to Comet ML
            if comet_experiment is not None:
                # Log overall metrics
                comet_experiment.log_metric("loss", val_loss_avg)
                comet_experiment.log_metric("bce", val_bce_avg)
                comet_experiment.log_metric("dice", val_dice_avg)
                comet_experiment.log_metric("jaccard", val_jaccard_avg)
                comet_experiment.log_metric("ssim", val_ssim_avg)
                comet_experiment.log_metric("pixel_accuracy", val_pixel_avg)

                # Log Jaccard buckets
                for bucket, count in jaccard_buckets.items():
                    if count > 0:
                        comet_experiment.log_metric(f"ji_{bucket:.2f}_{min(1.0, bucket + 0.05):.2f}", count, step=comet_experiment_epoch)

                # Create and log Jaccard histogram
                jaccard_values = []
                for bucket, count in jaccard_buckets.items():
                    jaccard_values.extend([bucket + 0.025] * count)  # Use bucket midpoint
                if jaccard_values:
                    comet_experiment.log_histogram_3d(jaccard_values, name="jaccard_distribution", step=comet_experiment_epoch)

        return val_jaccard_avg  # Return primary metric for tracking


# ============================================================
# ***************** TRAIN + VALIDATE + SAVE **********************
# ============================================================
def main():
    global comet_experiment_epoch

    trainer = Trainer(
        data_loaders=init_data_loaders(to_be_distributed),
        model_opt_lrsch=init_models_optimizers(epochs_end, to_be_distributed)
    )

    for epoch in range(epoch_st, epochs_end+1):

        # 1) Validating at start of training if resuming from checkpoint
        if epoch == epoch_st and config_vars.get('resume_start_with_eval', False) and resume_weights_path is not None:
            # If starting with evaluation, set epoch to 0 for initial validation
            if comet_experiment is not None:
                comet_experiment.set_epoch(comet_experiment_epoch)

            validate_context = comet_experiment.validate() if (is_main_process and comet_experiment is not None) else nullcontext()
            with validate_context:
                val_metric = trainer.validate_epoch(epoch)
        

		# Set epoch only on main process
        if comet_experiment is not None:
            comet_experiment_epoch += 1
            comet_experiment.set_epoch(comet_experiment_epoch)
	
        # 2) Training
        train_context = comet_experiment.train() if (is_main_process and comet_experiment is not None) else nullcontext()
        with train_context:
            train_loss = trainer.train_epoch(epoch)

        # 3) Validating(every N epochs)
        eval_frequency = config_vars.get('eval_each_epoch', 0)
        if eval_frequency > 0 and epoch % eval_frequency == 0:
            # Validation with context
            validate_context = comet_experiment.validate() if (is_main_process and comet_experiment is not None) else nullcontext()
            with validate_context:
                val_metric = trainer.validate_epoch(epoch)

        # 4) Saving Checkpoint
        if epoch >= epochs_end - config.save_last and epoch % config.save_step == 0:
            if args.use_accelerate:
                state_dict = trainer.model.state_dict()
            else:
                state_dict = trainer.model.module.state_dict() if to_be_distributed else trainer.model.state_dict()
            checkpoint_filename = '{}_epoch_{}.pth'.format(experiment_name, epoch)
            checkpoint_path = os.path.join(ckpt_dir, checkpoint_filename)
            torch.save(state_dict, checkpoint_path)

            if is_main_process:
                logger.info(f"{'='*80}")
                logger.info(f"💾 CHECKPOINT SAVED")
                logger.info(f"  Path: {checkpoint_path}")
                logger.info(f"{'='*80}\n")
				
    if to_be_distributed:
        destroy_process_group()


if __name__ == '__main__':
    main()
