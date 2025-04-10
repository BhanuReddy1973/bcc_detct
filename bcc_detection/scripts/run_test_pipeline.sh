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

# Load required modules
module load python/3.8
module load cuda/11.7
module load cudnn/8.4

# Activate virtual environment
source ~/venv/bin/activate

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)/..

# Create necessary directories
mkdir -p logs
mkdir -p models
mkdir -p reports
mkdir -p visualizations

# Run the test pipeline
echo "Starting test pipeline..."
python test_pipeline.py \
    --num-samples 100 \
    --batch-size 32 \
    --epochs 20 \
    --num-workers 4

# Compress logs and results
echo "Compressing logs and results..."
timestamp=$(date +%Y%m%d_%H%M%S)
tar -czf results_${timestamp}.tar.gz logs/ models/ reports/ visualizations/

echo "Test pipeline completed!" 