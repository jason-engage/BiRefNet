# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BiRefNet Training Enhanced is a production-ready fork of BiRefNet (Bilateral Reference for High-Resolution Dichotomous Image Segmentation) with comprehensive training infrastructure, monitoring, and experiment tracking capabilities.

### Key Enhancements in This Fork

- **Comet ML Integration**: Full experiment tracking and visualization
- **External Configuration**: `config_vars.yml` for parameter management
- **Advanced Monitoring**: Gradient norm tracking, color-coded performance indicators
- **Enhanced Validation**: tqdm progress bars, best/worst sample tracking, distribution analysis
- **Multi-Dataset Support**: Composition tracking for balanced training
- **Developer Experience**: VS Code settings, secure API management, smart resume

## Common Development Commands

### Environment Setup
```bash
# Create conda environment
conda create -n birefnet python=3.10 -y && conda activate birefnet
pip install -r requirements.txt

# Setup configuration
cp config_vars_template.yml config_vars.yml
# Edit config_vars.yml with your settings
```

### Training Commands
```bash
# Single GPU training
python train.py --experiment_name my_experiment

# Multi-GPU training with DDP
torchrun --nproc_per_node=4 train.py --dist True --experiment_name my_experiment

# Training with mixed precision (FP16/BF16)
accelerate launch train.py --use_accelerate --experiment_name my_experiment
```

### Testing Commands
```bash
# Run inference on images
python inference.py --image_path path/to/image.jpg

# Evaluate on test set
python eval_existingOnes.py

# Generate best epoch based on metrics
python gen_best_ep.py
```

## File Structure

### Core Training Files
- `train.py` - Main training script with comprehensive metrics and logging
- `config.py` - Model and training configuration
- `config_vars.yml` - External configuration (user-specific, gitignored)
- `config_vars_template.yml` - Configuration template with placeholders

### Model Architecture
- `models/birefnet.py` - Main BiRefNet model implementation
- `models/backbones/` - Various backbone networks (Swin, PVT, VGG, ResNet)
- `models/refinement/` - Refinement modules for post-processing

### Data Pipeline
- `dataset.py` - Dataset loading with multi-dataset support and dynamic sizing
- `image_proc.py` - Image preprocessing including random_rotate_zoom

### Training Infrastructure
- `loss.py` - Multiple loss functions (BCE, IoU, SSIM, MAE, Contour)
- `utils.py` - Training utilities including Logger and AverageMeter
- `scripts/utils.py` - Metric calculations and evaluation utilities

## Configuration System

### config_vars.yml Structure
```yaml
# Training Configuration
resume_weights_path: null  # Path to checkpoint to resume from
resume_start_with_eval: false  # Run validation before first epoch
log_frequency: 20  # How often to log metrics
eval_each_epoch: 1  # Run validation every N epochs (0=disabled)
max_samples: 0  # Limit training samples (0=use all)

# Comet ML Configuration
comet_ml_enable: false  # Enable experiment tracking
comet_ml_api_key: "YOUR_KEY"  # Get from comet.com
comet_ml_workspace: "YOUR_WORKSPACE"
comet_ml_project_name: "birefnet"
```

### config.py Key Settings
- `self.task`: Choose task (DIS5K, COD, HRSOD, 1024px, General-2K, Matting)
- `self.batch_size`: Batch size for training
- `self.dynamic_size`: List of sizes for dynamic resolution training
- `self.dynamic_size_batch`: Change size every N batches
- `self.bb`: Backbone network selection
- `self.optimizer`: AdamW, Adam, or Ranger
- `self.mixed_precision`: 'no', 'fp16', 'bf16', 'fp8'

## Key Features Implementation

### Gradient Norm Monitoring
```python
def checkGradientNorm(model):
    """Calculate L2 norm of all gradients."""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
```

### Dynamic Size Training
- Randomly selects resolution every N batches (default: 200)
- All sizes must be divisible by 32
- Configured via `config.dynamic_size` list

### Dataset Composition Tracking
- Tracks samples from each dataset in multi-dataset training
- Displays as "BC: 2\0\1\0\0\0" showing batch composition
- "TBC: 200\0\100\0\0\0" shows total epoch composition

### Validation Enhancements
- tqdm progress bar with real-time metrics
- Tracks 10 best/worst performing samples
- Creates Jaccard distribution histogram
- Logs sample images to Comet ML

## Training Output Format

```
E50 - 100/200 - L: 0.234, LB: 0.198, BC: 0.998, BCB: 0.999, DI: 0.985, DIB: 0.991,
JI: 0.971, JIB: 0.983, SI: 0.945, SIB: 0.962, LR: 0.0000071, M: 0.90, GN: 4.2,
BC: 2\0\1\0\0\0, TBC: 200\0\100\0\0\0, It/s: 3.2, ETA: 0:00:31
```

### Metrics Legend
- **E**: Epoch number
- **L/LB**: Loss (epoch average/batch)
- **BC/BCB**: BCE (average/batch)
- **DI/DIB**: Dice (average/batch)
- **JI/JIB**: Jaccard IoU (average/batch)
- **SI/SIB**: SSIM (average/batch)
- **LR**: Learning rate
- **M**: Momentum
- **GN**: Gradient norm
- **BC/TBC**: Batch/Total composition
- **It/s**: Iterations per second
- **ETA**: Estimated time remaining

### Color Coding
- 🔴 **Red** (Back.RED): Poor performance (Jaccard < 0.95)
- 🟢 **Green** (Back.GREEN): Good performance (BCE > 0.999)
- 🔵 **Cyan** (Back.CYAN): Excellent (BCE > 0.999 AND Dice > 0.985)
- 🔷 **Blue** (Back.BLUE): Normal performance

## Important Development Notes

### API Key Security
- **NEVER** commit `config_vars.yml` - it's gitignored
- Use `config_vars_template.yml` as reference
- API keys should only exist in local `config_vars.yml`

### Checkpoint Management
- Checkpoints saved as `{experiment_name}_epoch_{num}.pth`
- Automatic epoch detection when resuming
- Controlled by `save_last` and `save_step` in config.py

### Memory Optimization
- `load_all`: Load entire dataset into RAM (faster but memory intensive)
- `compile`: PyTorch 2.0 compilation (may cause memory leaks)
- `dynamic_size`: Requires disabling compile for compatibility

### Multi-GPU Training
- DDP: Use `torchrun --nproc_per_node=N`
- Accelerate: Better for mixed precision training
- Both supported, Accelerate recommended for FP16/BF16

## Common Tasks

### Adding a New Loss Function
1. Add to `loss.py` with proper weighting
2. Update `config.py` lambdas_pix_last dictionary
3. Add to loss calculation in `train.py`

### Adding a New Optimizer
1. Import in `train.py` (around line 325)
2. Add elif clause in `init_models_optimizers()`
3. Update config.py optimizer options

### Modifying Training Display
1. Edit display_msg in `train.py` (around line 671)
2. Update metric calculations if needed
3. Consider color coding for new metrics

### Adding Comet ML Metrics
1. Add calculation in training loop
2. Log with `comet_experiment.log_metric()`
3. Use appropriate step/epoch for consistency

## Troubleshooting

### High Gradient Norms (>100)
- Learning rate too high
- Bad initialization
- Incompatible checkpoint
- Add gradient clipping if needed

### Dynamic Size Slowdown
- Increase `dynamic_size_batch` (default: 200)
- Reduce number of size options
- Disable if performance critical

### Memory Issues
- Disable `load_all`
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision

### Checkpoint Loading Errors
- Check if .bin (HuggingFace) or .pth format
- Verify task compatibility (matting vs segmentation)
- Use strict=False for minor mismatches

## Best Practices

1. **Always use config_vars.yml** for experiment-specific settings
2. **Monitor gradient norms** - should be < 10 for stable training
3. **Use Comet ML** for experiment tracking and comparison
4. **Run validation frequently** early in training (every epoch)
5. **Save checkpoints regularly** with save_last and save_step
6. **Use mixed precision** for faster training with minimal quality loss
7. **Track dataset composition** for balanced multi-dataset training
8. **Document changes** in commit messages and comments