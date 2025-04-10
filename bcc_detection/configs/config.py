from typing import Dict, List
import os
from pathlib import Path
import torch

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = Path("D:/bhanu/dataset/package")  # Update with your actual dataset path

# Dataset paths
BCC_DIR = DATASET_ROOT / "bcc"
NON_MALIGNANT_DIR = DATASET_ROOT / "non-malignant"

# Data subdirectories
IMAGES_DIR = "data/images"
TISSUE_MASKS_DIR = "data/tissue_masks"
LABELS_DIR = "data/labels"

# Model configuration
MODEL_CONFIG = {
    "backbone": "efficientnet_b7",
    "num_classes": 2,  # BCC vs non-BCC
    "dropout_rate": 0.3,
    "feature_dim": 256,
    "pretrained": True
}

# Data preprocessing configuration
DATA_CONFIG = {
    "patch_size": 256,  # Size of patches to extract from TIF images
    "patch_overlap": 0.7,  # Overlap between patches
    "input_size": 224,  # Input size for the model
    "batch_size": 32,
    "num_workers": 4,
    "min_tissue_percentage": 0.3  # Minimum tissue percentage in a patch
}

# Training configuration
TRAIN_CONFIG = {
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 50,
    "early_stopping_patience": 10,
    "save_dir": "models",
    "log_dir": "logs"
}

# Paths
PATHS = {
    "bcc_images": str(BCC_DIR / IMAGES_DIR),
    "bcc_masks": str(BCC_DIR / TISSUE_MASKS_DIR),
    "bcc_labels": str(BCC_DIR / LABELS_DIR),
    "non_malignant_images": str(NON_MALIGNANT_DIR / IMAGES_DIR),
    "non_malignant_masks": str(NON_MALIGNANT_DIR / TISSUE_MASKS_DIR),
    "non_malignant_labels": str(NON_MALIGNANT_DIR / LABELS_DIR)
}

# Data preprocessing parameters
PREPROCESSING_CONFIG = {
    "normalization": "minmax",
    "augmentation": {
        "rotation": 30,
        "horizontal_flip": True,
        "vertical_flip": True,
        "brightness_contrast": True,
    }
}

# Feature extraction parameters
FEATURE_EXTRACTION_CONFIG = {
    "fcm_clusters": 5,
    "fcm_m": 2.0,
    "fcm_max_iter": 100,
    "fcm_error": 0.005,
    "patch_size": 64,
    "stride": 32,
}

# Training parameters
TRAINING_CONFIG = {
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_seed": 42,
    "pin_memory": True,
}

# Evaluation parameters
EVALUATION_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
    "confusion_matrix": True,
    "roc_curve": True,
    "pr_curve": True,
}

# Optimization parameters
OPTIMIZATION_CONFIG = {
    "optimizer": "adam",
    "scheduler": "reduce_lr_on_plateau",
    "scheduler_params": {
        "factor": 0.1,
        "patience": 5,
        "min_lr": 1e-6,
    },
    "early_stopping": {
        "patience": 10,
        "min_delta": 0.001,
    }
}

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
    
    # Dataset parameters
    NUM_SAMPLES = 100  # Number of random images to use
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Image parameters
    IMAGE_SIZE = 224  # Resize images to this size
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    
    # Model parameters
    MODEL_NAME = 'efficientnet_b4'
    PRETRAINED = True
    NUM_CLASSES = 2  # BCC vs Non-BCC
    
    # Training parameters
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 5
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Augmentation parameters
    AUGMENTATION_PROB = 0.5
    
    # Metrics
    METRICS = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    
    # Random seed
    RANDOM_SEED = 42

def get_config() -> Dict:
    """Get configuration dictionary"""
    config = {
        # Model configuration
        "backbone": "efficientnet_b7",
        "num_classes": 2,
        "dropout_rate": 0.3,
        "feature_dim": 256,
        
        # Data configuration
        "patch_size": 224,
        "min_tissue_percentage": 0.7,
        "overlap": 0.5,
        "batch_size": 32,
        "num_workers": 4,
        
        # Training configuration
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "num_epochs": 50,
        "early_stopping_patience": 10,
        
        # Paths
        "data_dir": "data",
        "checkpoint_dir": "checkpoints",
        
        # Data paths (to be filled with actual paths)
        "bcc_images": str(BCC_DIR / IMAGES_DIR),
        "bcc_masks": str(BCC_DIR / TISSUE_MASKS_DIR),
        "bcc_labels": str(BCC_DIR / LABELS_DIR),
        "non_malignant_images": str(NON_MALIGNANT_DIR / IMAGES_DIR),
        "non_malignant_masks": str(NON_MALIGNANT_DIR / TISSUE_MASKS_DIR),
        "non_malignant_labels": str(NON_MALIGNANT_DIR / LABELS_DIR)
    }
    
    # Create directories if they don't exist
    os.makedirs(config["data_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    
    return config

def update_data_paths(config: Dict, 
                     train_images: List[str],
                     train_labels: List[int],
                     val_images: List[str],
                     val_labels: List[int],
                     test_images: List[str],
                     test_labels: List[int]) -> Dict:
    """Update configuration with data paths and labels"""
    config["train_images"] = train_images
    config["train_labels"] = train_labels
    config["val_images"] = val_images
    config["val_labels"] = val_labels
    config["test_images"] = test_images
    config["test_labels"] = test_labels
    
    return config 