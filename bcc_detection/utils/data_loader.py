import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

class TIFDataset(Dataset):
    """Dataset class for TIF images."""
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
        is_train: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the TIF images
            transform: Optional transforms to apply to the images
            is_train: Whether this is the training set
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_train = is_train
        
        # Get all TIF files
        self.image_files = list(self.data_dir.glob('*.tif'))
        
        if not self.image_files:
            raise ValueError(f"No TIF files found in {data_dir}")
            
        print(f"Found {len(self.image_files)} TIF files")
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get an image and its label.
        
        Args:
            idx: Index of the image to get
            
        Returns:
            Tuple of (image, label)
        """
        img_path = self.image_files[idx]
        
        # Load image
        image = Image.open(img_path)
        
        # Convert to RGB if grayscale
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        # Get label from filename (assuming format: label_image.tif)
        label = int(img_path.stem.split('_')[0])
        
        return image, torch.tensor(label, dtype=torch.long)

def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation and testing.
    
    Args:
        data_dir: Directory containing the TIF images
        batch_size: Batch size for the data loaders
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = TIFDataset(
        data_dir=os.path.join(data_dir, 'train'),
        transform=train_transform,
        is_train=True
    )
    
    val_dataset = TIFDataset(
        data_dir=os.path.join(data_dir, 'val'),
        transform=val_transform,
        is_train=False
    )
    
    test_dataset = TIFDataset(
        data_dir=os.path.join(data_dir, 'test'),
        transform=val_transform,
        is_train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader 