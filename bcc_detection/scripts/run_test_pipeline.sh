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

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_SCRIPT="${PROJECT_DIR}/test_small_dataset.py"

# Get the current user's home directory
if [ "$SUDO_USER" ]; then
    USER_HOME=$(eval echo ~$SUDO_USER)
else
    USER_HOME=$HOME
fi

echo "Script directory: $SCRIPT_DIR"
echo "Project directory: $PROJECT_DIR"
echo "Python script: $PYTHON_SCRIPT"
echo "User home: $USER_HOME"

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Install system dependencies for OpenSlide
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y openslide-tools libopenslide-dev

# Activate virtual environment
echo "Activating virtual environment..."
VENV_PATH="${PROJECT_DIR}/venv"
if [ -d "$VENV_PATH" ]; then
    source "${VENV_PATH}/bin/activate"
else
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Install Python packages
echo "Installing Python packages..."
pip install --upgrade pip
pip install openslide-python
pip install openslide-bin
pip install -r "${PROJECT_DIR}/requirements.txt"

# Set environment variables
echo "Setting environment variables..."
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/:${LD_LIBRARY_PATH}"

# Create necessary directories
echo "Creating directories..."
cd "${PROJECT_DIR}"
mkdir -p logs
mkdir -p models
mkdir -p reports
mkdir -p visualizations

# Check GPU availability
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    GPU_FLAG="--gpu 0"
else
    echo "No GPU available, running on CPU"
    GPU_FLAG=""
fi

# Run the test pipeline
echo "Starting test pipeline..."
python "$PYTHON_SCRIPT" \
    --num-samples 100 \
    --batch-size 32 \
    --epochs 20 \
    --num-workers 4 \
    ${GPU_FLAG}

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