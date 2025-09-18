#!/bin/bash
# validate.sh - Run validation on a checkpoint

# Settings - following same convention as train.sh
experiment_name="$1"
checkpoint_path="$2"

# Usage info
if [ -z "$experiment_name" ] || [ -z "$checkpoint_path" ]; then
    echo ""
    echo "Usage: ./validate.sh <experiment_name> <checkpoint_path>"
    echo ""
    echo "Examples:"
    echo "  ./validate.sh val_run checkpoints/model_epoch_50.pth"
    echo "  ./validate.sh validation_best checkpoints/best_model.pth"
    echo "  ./validate.sh val_exp1 /path/to/pytorch_model.bin"
    echo ""
    exit 1
fi

# Check if checkpoint file exists
if [ ! -f "$checkpoint_path" ]; then
    echo "Error: Checkpoint file not found: $checkpoint_path"
    exit 1
fi

echo "=========================================="
echo "Running validation"
echo "Experiment name: $experiment_name"
echo "Checkpoint: $checkpoint_path"
echo "Device: GPU 0 (default)"
echo "=========================================="
echo ""

# Single GPU validation only (following your requirement)
echo "Starting validation..."
CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --mode validate \
    --validate_checkpoint "${checkpoint_path}" \
    --experiment_name "${experiment_name}" \
    --dist False

echo ""
echo "=========================================="
echo "Validation completed"
echo "=========================================="