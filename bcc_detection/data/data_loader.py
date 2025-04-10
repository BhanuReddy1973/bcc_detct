import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
from pathlib import Path
from ..preprocessing.tissue_segmentation import TissueSegmentation
from ..preprocessing.tissue_packing import TissuePacking
import os
import random

# Disable PIL image size limit
Image.MAX_IMAGE_PIXELS = None

class TIFDataset(Dataset):
    """Dataset for handling TIF format biopsy images"""
    
    def __init__(self,
                 image_paths: List[str],
                 labels: List[int],
                 patch_size: int = 224,
                 min_tissue_percentage: float = 0.7,
                 overlap: float = 0.5,
                 transform: Optional[callable] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.patch_size = patch_size
        self.transform = transform
        
        # Initialize preprocessing modules
        self.tissue_segmenter = TissueSegmentation()
        self.tissue_packer = TissuePacking(
            patch_size=patch_size,
            min_tissue_percentage=min_tissue_percentage,
            overlap=overlap
        )
        
        # Preprocess all images
        self.patches = []
        self.patch_labels = []
        self._preprocess_images()
    
    def _preprocess_images(self):
        """Preprocess all images and extract patches"""
        for image_path, label in zip(self.image_paths, self.labels):
            # Open image
            img = np.array(Image.open(image_path))
            
            # Segment tissue
            mask, _ = self.tissue_segmenter.segment_tissue(img)
            
            # Extract patches
            patches = self.tissue_packer.extract_patches(img, mask)
            
            # Add to dataset
            self.patches.extend(patches)
            self.patch_labels.extend([label] * len(patches))
    
    def __len__(self) -> int:
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        patch = self.patches[idx]
        label = self.patch_labels[idx]
        
        # Convert to tensor
        image = torch.from_numpy(patch.image).permute(2, 0, 1).float() / 255.0
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_data_loaders(
    train_images: List[str],
    train_labels: List[int],
    val_images: List[str],
    val_labels: List[int],
    test_images: List[str],
    test_labels: List[int],
    batch_size: int = 32,
    num_workers: int = 4,
    patch_size: int = 224,
    min_tissue_percentage: float = 0.7,
    overlap: float = 0.5,
    transform: Optional[callable] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_images: List of paths to training images
        train_labels: List of labels for training images
        val_images: List of paths to validation images
        val_labels: List of labels for validation images
        test_images: List of paths to test images
        test_labels: List of labels for test images
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        patch_size: Size of patches to extract
        min_tissue_percentage: Minimum tissue percentage for patches
        overlap: Overlap between patches
        transform: Optional transform to apply to patches
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = TIFDataset(
        train_images,
        train_labels,
        patch_size=patch_size,
        min_tissue_percentage=min_tissue_percentage,
        overlap=overlap,
        transform=transform
    )
    
    val_dataset = TIFDataset(
        val_images,
        val_labels,
        patch_size=patch_size,
        min_tissue_percentage=min_tissue_percentage,
        overlap=overlap,
        transform=transform
    )
    
    test_dataset = TIFDataset(
        test_images,
        test_labels,
        patch_size=patch_size,
        min_tissue_percentage=min_tissue_percentage,
        overlap=overlap,
        transform=transform
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

def load_patches(class_name: str, num_samples: Optional[int] = None) -> np.ndarray:
    """
    Load preprocessed image patches for a specific class.
    
    Args:
        class_name (str): Class name ('bcc' or 'non-malignant')
        num_samples (int, optional): Number of samples to load. If None, load all.
    
    Returns:
        np.ndarray: Array of image patches
    """
    # Define the base path for the dataset
    base_path = Path('dataset/package')
    
    # Get the class-specific path
    class_path = base_path / class_name / 'data/images'
    
    # Get list of all TIF files
    image_files = list(class_path.glob('*.tif'))
    
    if not image_files:
        raise ValueError(f"No TIF files found in {class_path}")
    
    # Randomly sample if num_samples is specified
    if num_samples is not None:
        image_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Initialize preprocessing modules with optimized parameters
    tissue_segmenter = TissueSegmentation()
    tissue_packer = TissuePacking(
        patch_size=256,  # Optimized patch size
        min_tissue_percentage=0.3,  # Optimized tissue threshold
        overlap=0.7  # Optimized overlap
    )
    
    # Load and preprocess images
    all_patches = []
    for img_path in image_files:
        try:
            # Load and resize image
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # Resize if necessary
            h, w = img_array.shape[:2]
            if h > 4096 or w > 4096:
                scale = min(4096/h, 4096/w)
                new_h, new_w = int(h * scale), int(w * scale)
                print(f"Resizing {img_path.name} from {h}x{w} to {new_h}x{new_w}")
                img_array = np.array(Image.fromarray(img_array).resize((new_w, new_h)))
            
            # Ensure image is RGB
            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = img_array[:, :, :3]
            
            # Segment tissue
            mask, _ = tissue_segmenter.segment_tissue(img_array)
            
            # Extract patches
            patches = tissue_packer.extract_patches(img_array, mask)
            all_patches.extend(patches)
            
            print(f"Extracted {len(patches)} patches from {img_path.name}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    if not all_patches:
        raise ValueError(f"No valid patches extracted from {class_name} images")
    
    return np.array([patch.image for patch in all_patches])

def get_patch_dataset(
    num_samples_per_class: Optional[int] = None,
    train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get training and validation datasets of preprocessed patches.
    
    Args:
        num_samples_per_class (int, optional): Number of samples per class
        train_ratio (float): Ratio of training samples
    
    Returns:
        Tuple containing:
        - Training patches
        - Training labels
        - Validation patches
        - Validation labels
    """
    print("Loading BCC patches...")
    bcc_patches = load_patches('bcc', num_samples_per_class)
    print(f"Loaded {len(bcc_patches)} BCC patches")
    
    print("\nLoading non-malignant patches...")
    normal_patches = load_patches('non-malignant', num_samples_per_class)
    print(f"Loaded {len(normal_patches)} non-malignant patches")
    
    # Create labels
    bcc_labels = np.ones(len(bcc_patches))
    normal_labels = np.zeros(len(normal_patches))
    
    # Combine patches and labels
    all_patches = np.concatenate([bcc_patches, normal_patches])
    all_labels = np.concatenate([bcc_labels, normal_labels])
    
    # Shuffle the data
    indices = np.arange(len(all_patches))
    np.random.shuffle(indices)
    all_patches = all_patches[indices]
    all_labels = all_labels[indices]
    
    # Split into training and validation sets
    split_idx = int(len(all_patches) * train_ratio)
    train_patches = all_patches[:split_idx]
    train_labels = all_labels[:split_idx]
    val_patches = all_patches[split_idx:]
    val_labels = all_labels[split_idx:]
    
    print(f"\nDataset split:")
    print(f"Training: {len(train_patches)} patches")
    print(f"Validation: {len(val_patches)} patches")
    
    return train_patches, train_labels, val_patches, val_labels 