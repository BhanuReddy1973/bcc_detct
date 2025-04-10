# BCC Detection Pipeline

This repository contains a pipeline for Basal Cell Carcinoma (BCC) detection using deep learning.

## Directory Structure

```
bcc_detection/
├── configs/               # Configuration files
├── data/                 # Data directory
├── evaluation/           # Evaluation scripts
├── feature_extraction/   # Feature extraction modules
├── logs/                # Training logs
├── models/              # Saved models
├── optimization/        # Optimization modules
├── preprocessing/       # Preprocessing modules
├── reports/            # Generated reports
├── results/            # Training results
├── scripts/            # Utility scripts
├── tests/              # Test files
├── utils/              # Utility functions
├── visualizations/     # Generated visualizations
├── requirements.txt    # Python dependencies
├── run_training.py     # Main training script
└── run_training.sh     # Training script for HPC
```

## Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running on HPC

1. Transfer the code to HPC:
```bash
scp -r bcc_detection user@hpc:/path/to/destination
```

2. SSH into HPC:
```bash
ssh user@hpc
```

3. Navigate to the directory:
```bash
cd /path/to/bcc_detection
```

4. Make the script executable:
```bash
chmod +x run_training.sh
```

5. Submit the job:
```bash
sbatch -N 1 -n 1 --gres=gpu:1 --time=24:00:00 run_training.sh
```

## Configuration

Edit `configs/training_config.yaml` to modify:
- Training parameters
- Model architecture
- Data preprocessing
- Hyperparameter tuning

## Testing

Run the test suite:
```bash
pytest tests/
```

## Outputs

The pipeline generates:
- Model checkpoints in `models/`
- Training logs in `logs/`
- Visualizations in `visualizations/`
- Results in `results/`
- Reports in `reports/`

## Monitoring

1. Check job status:
```bash
squeue -u <username>
```

2. Monitor GPU usage:
```bash
nvidia-smi
```

3. View logs:
```bash
tail -f logs/training_*.log
```

## Troubleshooting

Common issues and solutions:

1. **CUDA Out of Memory**
   - Reduce batch size in config
   - Use gradient accumulation

2. **Dataset Loading Issues**
   - Verify dataset structure
   - Check file permissions

3. **Module Not Found**
   - Reinstall requirements
   - Check Python version

4. **Permission Issues**
   - Check directory permissions
   - Use `chmod` to fix

## Contributing

1. Create a new branch
2. Make changes
3. Run tests
4. Submit pull request

## License

This project is licensed under the MIT License.

# Basal Cell Carcinoma (BCC) Detection

An automated deep learning pipeline for detecting Basal Cell Carcinoma in Whole Slide Images (WSIs).

## Project Overview

This project implements a comprehensive pipeline for automated BCC detection using deep learning techniques. The pipeline includes tissue segmentation, patch extraction, feature learning, and classification components.

### Key Features

- Automated tissue segmentation using Otsu's thresholding
- Intelligent patch extraction with tissue content filtering
- Deep feature extraction using EfficientNet-B7
- Fuzzy clustering for tissue pattern analysis
- Mixed precision training for improved performance
- Comprehensive evaluation metrics and visualization tools

## Project Structure

```
bcc_detection/
├── configs/            # Configuration files
├── data/              # Data directory
│   ├── raw/           # Raw WSI files
│   ├── processed/     # Processed patches
│   ├── annotations/   # Ground truth annotations
│   └── splits/        # Dataset splits
├── preprocessing/     # Preprocessing modules
├── feature_extraction/# Feature extraction modules
├── models/           # Model architectures
├── evaluation/       # Evaluation metrics
├── optimization/     # Training optimization
├── utils/           # Utility functions
└── scripts/         # Training scripts
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bcc_detection.git
cd bcc_detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your WSI dataset:
   - Place WSI files in `data/raw/`
   - Add annotations to `data/annotations/`

2. Configure the pipeline:
   - Adjust parameters in `configs/config.py`
   - Modify model architecture if needed

3. Run the training pipeline:
```bash
python main.py
```

## Model Architecture

The model uses a hierarchical approach:
1. EfficientNet-B7 backbone for feature extraction
2. Fuzzy clustering for tissue pattern analysis
3. Fully connected layers for final classification

## Training

The training process includes:
- Mixed precision training for efficiency
- Progressive resolution analysis
- Early stopping and model checkpointing
- Comprehensive metric tracking

## Evaluation

The system provides various evaluation metrics:
- Accuracy, Sensitivity, Specificity
- ROC curves and AUC scores
- Uncertainty estimation
- Visualization of predictions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors
- Special thanks to the medical professionals who provided annotations
- Powered by PyTorch and OpenSlide 

# BCC Detection Pipeline Test Report

## Dataset
- Total samples: 1000 (50% of total data)
- Training: 400 images (40%)
- Validation: 50 images (5%)
- Testing: 50 images (5%)
- Image size: 224x224
- Classes: Grid pattern vs Circular pattern

## Model Architecture
- Backbone: EfficientNet-B7
- Classification head: Custom with dropout
- Parameters: ~66M (pretrained)

## Training Setup
- Optimizer: Adam (lr=1e-4, weight_decay=1e-5)
- Loss: CrossEntropyLoss
- Batch size: 32
- Epochs: 20 (with early stopping)
- Device: CPU

## Initial Results (Epoch 1)
- Training accuracy: 87.25%
- Training loss: 0.5087
- Validation: In progress

## Data Augmentation
- Random horizontal flip
- Random vertical flip
- Random rotation (15 degrees)
- Color jitter (brightness, contrast)

## Next Steps
1. Complete training to see final accuracy
2. Analyze validation and test performance
3. Fine-tune hyperparameters if needed
4. Implement on real BCC dataset

The initial results show promising performance with 87.25% training accuracy in the first epoch, suggesting the model is effectively learning to distinguish between the patterns. 

# BCC Detection Pipeline - Large Dataset Setup

This guide explains how to set up and run the BCC detection pipeline with a large dataset (660GB) on HPC.

## Directory Structure

```
your_workspace/
├── bcc_detection/          # Git repository
│   ├── configs/           # Configuration files
│   ├── evaluation/        # Evaluation scripts
│   ├── feature_extraction/# Feature extraction modules
│   ├── logs/             # Training logs
│   ├── models/           # Saved models
│   ├── optimization/     # Optimization modules
│   ├── preprocessing/    # Preprocessing modules
│   ├── reports/         # Generated reports
│   ├── results/         # Training results
│   ├── scripts/         # Utility scripts
│   ├── tests/           # Test files
│   ├── utils/           # Utility functions
│   ├── visualizations/  # Generated visualizations
│   ├── requirements.txt # Python dependencies
│   ├── run_training.py  # Main training script
│   └── run_training.sh  # Training script for HPC
│
└── dataset/              # Large dataset (660GB)
    ├── package/
    │   ├── bcc/
    │   │   └── data/
    │   │       └── images/
    │   └── non-malignant/
    │       └── data/
    │           └── images/
```

## Setup Instructions

### 1. Local Machine Setup

1. Create workspace directory:
```bash
mkdir -p ~/bcc_project
cd ~/bcc_project
```

2. Clone the repository:
```bash
git clone <repository_url> bcc_detection
```

3. Create dataset directory:
```bash
mkdir dataset
```

4. Copy your dataset to the dataset directory:
```bash
# If dataset is on external drive
cp -r /path/to/external/drive/dataset/* dataset/

# If dataset is on network drive
rsync -avz /path/to/network/drive/dataset/* dataset/
```

### 2. HPC Setup

1. Transfer the code to HPC:
```bash
# From your local machine
scp -r ~/bcc_project/bcc_detection user@hpc:/path/to/workspace/
```

2. Transfer the dataset to HPC:
```bash
# Using rsync for large dataset (recommended)
rsync -avz --progress ~/bcc_project/dataset user@hpc:/path/to/workspace/

# Or using scp (slower for large files)
scp -r ~/bcc_project/dataset user@hpc:/path/to/workspace/
```

3. SSH into HPC:
```bash
ssh user@hpc
```

4. Navigate to workspace:
```bash
cd /path/to/workspace
```

5. Verify directory structure:
```bash
ls -R bcc_detection/
ls -R dataset/
```

### 3. Environment Setup on HPC

1. Load required modules:
```bash
module load cuda/11.8
module load python/3.9
```

2. Create and activate virtual environment:
```bash
cd bcc_detection
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Make training script executable:
```bash
chmod +x run_training.sh
```

### 4. Running the Pipeline

1. Submit the job to SLURM:
```bash
sbatch -N 1 -n 1 --gres=gpu:1 --time=72:00:00 --mem=128G run_training.sh
```

2. Monitor the job:
```bash
# Check job status
squeue -u <username>

# Monitor GPU usage
nvidia-smi

# View logs
tail -f logs/training_*.log
```

### 5. Dataset Management

1. To add new images:
```bash
# On HPC
rsync -avz --progress /path/to/new/images/* dataset/package/bcc/data/images/
# or
rsync -avz --progress /path/to/new/images/* dataset/package/non-malignant/data/images/
```

2. To verify dataset integrity:
```bash
# Count total images
find dataset/ -type f -name "*.tif" | wc -l

# Check file sizes
du -sh dataset/
```

### 6. Troubleshooting

1. **Dataset Access Issues**
```bash
# Check permissions
ls -la dataset/

# Fix permissions if needed
chmod -R 755 dataset/
```

2. **Storage Issues**
```bash
# Check available space
df -h

# Clean up old files if needed
rm -rf models/old_checkpoints/
rm -rf logs/old_logs/
```

3. **Memory Issues**
```bash
# Monitor memory usage
free -h

# Adjust batch size in config if needed
nano configs/training_config.yaml
```

4. **GPU Issues**
```bash
# Check GPU status
nvidia-smi

# Check CUDA version
nvcc --version
```

### 7. Best Practices

1. **Data Organization**
   - Keep dataset structure consistent
   - Use meaningful filenames
   - Maintain backup copies

2. **Resource Management**
   - Monitor disk usage
   - Clean up old checkpoints
   - Use compression for logs

3. **Performance Optimization**
   - Use appropriate batch size
   - Enable mixed precision training
   - Use data prefetching

4. **Backup Strategy**
   - Regular dataset backups
   - Model checkpoint backups
   - Log file backups

## Support

For issues or questions:
1. Check the logs in `logs/`
2. Review the troubleshooting section
3. Contact the development team

## License

This project is licensed under the MIT License. 