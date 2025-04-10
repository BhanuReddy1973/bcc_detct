#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
from pathlib import Path

# Force CPU usage and configure PyTorch
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.set_num_threads(1)  # Limit CPU threads

# Ensure we're using CPU
if hasattr(torch, 'cuda'):
    torch.cuda.is_available = lambda: False
device = torch.device('cpu')

# Disable CUDA support in PyTorch
def _dummy_cuda_device(*args, **kwargs):
    raise RuntimeError("CUDA is not available")
torch.cuda.device = _dummy_cuda_device
torch.cuda.device_count = lambda: 0
torch.cuda.current_device = lambda: -1
torch.cuda.set_device = lambda x: None

# Disable pin memory in DataLoader
def _dummy_pin_memory(*args, **kwargs):
    return args[0]
torch._utils.pin_memory.pin_memory = _dummy_pin_memory

class CustomDataset(Dataset):
    def __init__(self, bcc_path, non_malignant_path, transform=None, num_samples=None):
        self.transform = transform
        self.bcc_files = []
        self.non_malignant_files = []
        
        # Get all BCC image files
        for root, _, files in os.walk(bcc_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    self.bcc_files.append(os.path.join(root, file))
        
        # Get all non-malignant image files
        for root, _, files in os.walk(non_malignant_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    self.non_malignant_files.append(os.path.join(root, file))
        
        print(f"Found {len(self.bcc_files)} BCC files and {len(self.non_malignant_files)} non-malignant files")
        
        if num_samples is not None:
            num_samples_per_class = num_samples // 2
            self.bcc_files = np.random.choice(self.bcc_files, num_samples_per_class, replace=False)
            self.non_malignant_files = np.random.choice(self.non_malignant_files, num_samples_per_class, replace=False)
            print(f"Selected {num_samples_per_class} BCC images and {num_samples_per_class} non-malignant images")
        
        self.all_files = list(self.bcc_files) + list(self.non_malignant_files)
        self.labels = [1] * len(self.bcc_files) + [0] * len(self.non_malignant_files)
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx):
        img_path = self.all_files[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a black image and the label if there's an error
            if self.transform:
                return torch.zeros((3, 224, 224)), label
            return Image.new('RGB', (224, 224), 'black'), label

def main():
    parser = argparse.ArgumentParser(description='Test pipeline with small dataset')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to use')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loading workers')
    args = parser.parse_args()

    print("Starting test with small dataset...")
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Set up paths
    dataset_root = Path("/home/bhanu/bcc_detection/dataset/package")
    print(f"Looking for dataset in: {dataset_root}")
    
    bcc_path = dataset_root / "bcc/data/images"
    non_malignant_path = dataset_root / "non-malignant/data/images"
    print(f"BCC path: {bcc_path}")
    print(f"Non-malignant path: {non_malignant_path}")

    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset
    dataset = CustomDataset(
        str(bcc_path),
        str(non_malignant_path),
        transform=transform,
        num_samples=args.num_samples
    )

    # Create data loader with CPU-only settings
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
        generator=torch.Generator(device='cpu')
    )

    print("\nTesting batch loading...")
    try:
        for batch_idx, (images, labels) in enumerate(test_loader):
            print(f"\nBatch {batch_idx + 1}:")
            print(f"Images shape: {images.shape}")
            print(f"Labels: {labels}")
            
            if batch_idx >= 2:  # Only test first 3 batches
                break
                
        print("\nTest completed successfully!")
                
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main() 