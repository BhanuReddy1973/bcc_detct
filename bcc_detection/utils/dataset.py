import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from ..configs.config import Config

class BCCDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

def get_transforms(is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=Config.AUGMENTATION_PROB),
            transforms.RandomVerticalFlip(p=Config.AUGMENTATION_PROB),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

def load_dataset(data_dir):
    """Load and prepare the dataset"""
    # Get all TIF files
    bcc_path = os.path.join(data_dir, 'package/bcc/data/biopsies/images')
    non_malignant_path = os.path.join(data_dir, 'package/non-malignant/data/biopsies/images')
    
    bcc_files = [os.path.join(bcc_path, f) for f in os.listdir(bcc_path) if f.endswith('.tif')]
    non_malignant_files = [os.path.join(non_malignant_path, f) for f in os.listdir(non_malignant_path) if f.endswith('.tif')]
    
    # Randomly sample images
    random.seed(Config.RANDOM_SEED)
    bcc_files = random.sample(bcc_files, min(len(bcc_files), Config.NUM_SAMPLES // 2))
    non_malignant_files = random.sample(non_malignant_files, min(len(non_malignant_files), Config.NUM_SAMPLES // 2))
    
    # Prepare data and labels
    image_paths = bcc_files + non_malignant_files
    labels = [1] * len(bcc_files) + [0] * len(non_malignant_files)
    
    # Split dataset
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, 
        train_size=Config.TRAIN_RATIO,
        random_state=Config.RANDOM_SEED,
        stratify=labels
    )
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=Config.TEST_RATIO/(1-Config.TRAIN_RATIO),
        random_state=Config.RANDOM_SEED,
        stratify=temp_labels
    )
    
    # Create datasets
    train_dataset = BCCDataset(train_paths, train_labels, get_transforms(True))
    val_dataset = BCCDataset(val_paths, val_labels, get_transforms(False))
    test_dataset = BCCDataset(test_paths, test_labels, get_transforms(False))
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )
    
    return train_loader, val_loader, test_loader 