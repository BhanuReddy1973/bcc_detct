#!/bin/bash

# Exit on error
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Load required modules
module load cuda/11.8
module load python/3.9
module load gcc/9.3.0

# Create and activate virtual environment
if [ ! -d "venv" ]; then
    python -m venv venv
fi
source venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install torch torchvision numpy pillow matplotlib seaborn scikit-learn tqdm pyyaml

# Create necessary directories
mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$SCRIPT_DIR/visualizations"
mkdir -p "$SCRIPT_DIR/models"
mkdir -p "$SCRIPT_DIR/results"
mkdir -p "$SCRIPT_DIR/reports"

# Check if dataset directory exists
if [ ! -d "../dataset" ]; then
    echo "Error: dataset directory not found in parent directory"
    exit 1
fi

# Check available GPU memory
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "Available GPU memory: $GPU_MEM MB"

# Adjust batch size based on GPU memory
if [ $GPU_MEM -lt 16000 ]; then
    echo "Warning: Low GPU memory detected. Adjusting batch size..."
    sed -i 's/batch_size: 32/batch_size: 16/' configs/training_config.yaml
fi

# Run the training script with memory monitoring
echo "Starting training..."
python run_training.py

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully"
    
    # Compress logs to save space
    echo "Compressing logs..."
    tar -czf "logs/training_$(date +%Y%m%d_%H%M%S).tar.gz" logs/*.log
    rm logs/*.log
    
    # Clean up old model checkpoints (keep only the best)
    echo "Cleaning up old checkpoints..."
    find models/ -name "best_model_iteration_*.pth" -not -name "best_model_iteration_10.pth" -delete
else
    echo "Training failed"
    exit 1
fi

# Deactivate virtual environment
deactivate

echo "All done!" 