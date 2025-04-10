import os
import torch
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random

class BCCDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train', num_samples=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        
        # Load data from the specified dataset path
        self.samples = self._load_samples(num_samples)
    
    def _load_samples(self, num_samples=None):
        """Load data from the dataset directory"""
        samples = []
        
        # Load BCC (positive) samples
        bcc_dir = self.data_dir / "package" / "bcc" / "data" / "images"
        for img_path in bcc_dir.glob("*.tif"):
            samples.append({
                'image_path': str(img_path),
                'label': 1  # BCC is positive class
            })
        
        # Load non-malignant (negative) samples
        normal_dir = self.data_dir / "package" / "non-malignant" / "data" / "images"
        for img_path in normal_dir.glob("*.tif"):
            samples.append({
                'image_path': str(img_path),
                'label': 0  # Non-malignant is negative class
            })
        
        # Randomly sample if num_samples is specified
        if num_samples is not None:
            # Ensure equal number of samples from each class
            bcc_samples = [s for s in samples if s['label'] == 1]
            normal_samples = [s for s in samples if s['label'] == 0]
            
            # Take equal number of samples from each class
            num_samples_per_class = num_samples // 2
            bcc_samples = random.sample(bcc_samples, min(num_samples_per_class, len(bcc_samples)))
            normal_samples = random.sample(normal_samples, min(num_samples_per_class, len(normal_samples)))
            
            samples = bcc_samples + normal_samples
        
        # Split into train/val/test
        random.shuffle(samples)
        n = len(samples)
        if self.split == 'train':
            return samples[:int(0.7*n)]
        elif self.split == 'val':
            return samples[int(0.7*n):int(0.85*n)]
        else:  # test
            return samples[int(0.85*n):]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        label = sample['label']
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'input': image,
            'label': label
        }

def create_data_loaders(data_dir, batch_size=32, num_samples=None, train_transform=None, val_transform=None):
    """Create data loaders for train, validation, and test sets"""
    if train_transform is None:
        from torchvision import transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    if val_transform is None:
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    # Create datasets
    train_dataset = BCCDataset(data_dir, train_transform, 'train', num_samples)
    val_dataset = BCCDataset(data_dir, val_transform, 'val', num_samples)
    test_dataset = BCCDataset(data_dir, val_transform, 'test', num_samples)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test data loading
    data_dir = Path("D:/bhanu/dataset")
    train_loader, val_loader, test_loader = create_data_loaders(data_dir, num_samples=100)  # Load 100 samples (50 from each class)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Print first batch
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch['input'].shape}")
    print(f"Labels: {batch['label']}") 