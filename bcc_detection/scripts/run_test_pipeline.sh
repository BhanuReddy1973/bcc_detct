#!/bin/bash
#SBATCH --job-name=bcc_test_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=test_pipeline_%j.out
#SBATCH --error=test_pipeline_%j.err

# Print start time
echo "Job started at $(date)"

# Load required modules
echo "Loading modules..."
module load python/3.8
module load cuda/11.7
module load cudnn/8.4

# Check if modules loaded successfully
if [ $? -ne 0 ]; then
    echo "Error: Failed to load modules"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source ~/venv/bin/activate

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

# Set environment variables
echo "Setting environment variables..."
export PYTHONPATH=$PYTHONPATH:$(pwd)/..

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p models
mkdir -p reports
mkdir -p visualizations

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi

# Run the test pipeline
echo "Starting test pipeline..."
python test_pipeline.py \
    --num-samples 100 \
    --batch-size 32 \
    --epochs 20 \
    --num-workers 4

# Check if pipeline ran successfully
if [ $? -ne 0 ]; then
    echo "Error: Test pipeline failed"
    exit 1
fi

# Compress logs and results
echo "Compressing logs and results..."
timestamp=$(date +%Y%m%d_%H%M%S)
tar -czf results_${timestamp}.tar.gz logs/ models/ reports/ visualizations/

if [ $? -ne 0 ]; then
    echo "Error: Failed to compress results"
    exit 1
fi

echo "Test pipeline completed successfully at $(date)" 