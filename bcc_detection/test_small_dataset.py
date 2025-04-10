import os
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from PIL import Image

# Increase PIL image size limit
Image.MAX_IMAGE_PIXELS = None

class SimpleTIFDataset(Dataset):
    """Simple dataset for handling TIF format biopsy images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # Load image
            img = Image.open(self.image_paths[idx])
            img = np.array(img)
            
            # Convert to tensor and normalize
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            
            # Apply transform if specified
            if self.transform:
                img = self.transform(img)
            
            return img, self.labels[idx]
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {str(e)}")
            raise

def get_small_dataset(num_samples_per_class=5):
    """
    Create a small dataset for testing with a limited number of samples per class.
    """
    try:
        # Base path to the dataset - using absolute path
        base_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / 'dataset' / 'package'
        print(f"Looking for dataset in: {base_path}")
        
        # Get paths for both classes
        bcc_path = base_path / 'bcc' / 'data' / 'images'
        non_malignant_path = base_path / 'non-malignant' / 'data' / 'images'
        
        print(f"BCC path: {bcc_path}")
        print(f"Non-malignant path: {non_malignant_path}")
        
        # Get list of all TIF files
        bcc_files = list(bcc_path.glob('*.tif'))
        non_malignant_files = list(non_malignant_path.glob('*.tif'))
        
        print(f"Found {len(bcc_files)} BCC files and {len(non_malignant_files)} non-malignant files")
        
        if not bcc_files or not non_malignant_files:
            raise ValueError("No image files found in one or both directories")
        
        # Randomly sample the specified number of files
        bcc_files = random.sample(bcc_files, min(num_samples_per_class, len(bcc_files)))
        non_malignant_files = random.sample(non_malignant_files, min(num_samples_per_class, len(non_malignant_files)))
        
        # Create lists of paths and labels
        image_paths = bcc_files + non_malignant_files
        labels = [1] * len(bcc_files) + [0] * len(non_malignant_files)
        
        print(f"Selected {len(bcc_files)} BCC images and {len(non_malignant_files)} non-malignant images")
        
        # Create dataset
        dataset = SimpleTIFDataset(
            image_paths=image_paths,
            labels=labels
        )
        
        # Create data loader
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        return loader
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        raise

def main():
    print("Starting test with small dataset...")
    
    try:
        # Create small dataset
        test_loader = get_small_dataset(num_samples_per_class=5)
        
        # Print dataset information
        print(f"\nDataset Information:")
        print(f"Number of batches: {len(test_loader)}")
        print(f"Batch size: {test_loader.batch_size}")
        
        # Test loading a batch
        print("\nTesting batch loading...")
        for batch_idx, (images, labels) in enumerate(test_loader):
            print(f"\nBatch {batch_idx + 1}:")
            print(f"Images shape: {images.shape}")
            print(f"Labels: {labels}")
            
            # Only process first batch for testing
            break
        
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 