#!/bin/bash
# validate.sh - Run validation on a checkpoint

checkpoint_path="$1"
enable_comet="${2:-0}"  # Default to 0 (disabled)

# Usage info
if [ -z "$checkpoint_path" ]; then
    echo ""
    echo "Usage: ./validate.sh <checkpoint_path> [enable_comet]"
    echo ""
    echo "Examples:"
    echo "  ./validate.sh checkpoints/model_epoch_50.pth      # No Comet ML"
    echo "  ./validate.sh checkpoints/model.pth 0             # No Comet ML (explicit)"
    echo "  ./validate.sh checkpoints/model.pth 1             # With Comet ML"
    echo ""
    exit 1
fi

# Check if checkpoint file exists
if [ ! -f "$checkpoint_path" ]; then
    echo "Error: Checkpoint file not found: $checkpoint_path"
    exit 1
fi

# Extract filename without extension for experiment name
# This handles both .pth and .bin extensions
experiment_name=$(basename "${checkpoint_path}")
experiment_name="${experiment_name%.*}"

# Build comet flag based on second argument
comet_flag=""
comet_status="Disabled"
if [ "${enable_comet}" == "1" ]; then
    comet_flag="--comet"
    comet_status="Enabled"
fi

echo "=========================================="
echo "Running validation"
echo "Experiment name: ${experiment_name}"
echo "Checkpoint: $checkpoint_path"
echo "Device: GPU 0 (default)"
echo "Comet ML: ${comet_status}"
echo "=========================================="
echo ""

# Single GPU validation only
echo "Starting validation..."
CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --mode validate \
    --validate_checkpoint "${checkpoint_path}" \
    --experiment_name "val_${experiment_name}" \
    --dist False \
    ${comet_flag}

echo ""
echo "=========================================="
echo "Validation completed"
echo "=========================================="