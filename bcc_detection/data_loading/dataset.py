import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import logging
from PIL import Image
import openslide
from .preprocessing import TissueSegmenter, TissuePacker

class WSIDataset(Dataset):
    def __init__(self, data_dir, transform=None, patch_size=224, min_tissue_ratio=0.7):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.patch_size = patch_size
        self.min_tissue_ratio = min_tissue_ratio
        
        # Initialize preprocessing modules
        self.segmenter = TissueSegmenter()
        self.packer = TissuePacker(patch_size=patch_size, min_tissue_ratio=min_tissue_ratio)
        
        # Load slide paths
        self.slide_paths = self._load_slide_paths()
        self.patches = []
        self.coordinates = []
        
        # Process slides
        self._process_slides()
    
    def _load_slide_paths(self):
        """Load paths of all WSI files"""
        try:
            slide_paths = []
            for ext in ['.svs', '.ndpi', '.tif', '.tiff']:
                slide_paths.extend(list(self.data_dir.glob(f'*{ext}')))
            return slide_paths
            
        except Exception as e:
            logging.error(f"Error loading slide paths: {str(e)}")
            raise
    
    def _process_slides(self):
        """Process all slides and extract patches"""
        try:
            for slide_path in self.slide_paths:
                # Open slide
                slide = openslide.OpenSlide(str(slide_path))
                
                # Get slide dimensions
                width, height = slide.dimensions
                
                # Read region at level 0 (highest resolution)
                region = slide.read_region((0, 0), 0, (width, height))
                region = np.array(region)[:, :, :3]  # Convert to numpy array and remove alpha channel
                
                # Segment tissue
                mask = self.segmenter.segment_tissue(region)
                
                # Extract patches
                patches, coordinates = self.packer.extract_patches(region, mask)
                
                # Add to dataset
                self.patches.extend(patches)
                self.coordinates.extend(coordinates)
                
                # Close slide
                slide.close()
                
        except Exception as e:
            logging.error(f"Error processing slides: {str(e)}")
            raise
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        try:
            patch = self.patches[idx]
            coord = self.coordinates[idx]
            
            # Convert to PIL Image
            patch = Image.fromarray(patch)
            
            # Apply transforms
            if self.transform:
                patch = self.transform(patch)
            
            return patch, coord
            
        except Exception as e:
            logging.error(f"Error getting item {idx}: {str(e)}")
            raise

class PatchDataset(Dataset):
    def __init__(self, patches, coordinates, transform=None):
        self.patches = patches
        self.coordinates = coordinates
        self.transform = transform
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        try:
            patch = self.patches[idx]
            coord = self.coordinates[idx]
            
            # Convert to PIL Image
            patch = Image.fromarray(patch)
            
            # Apply transforms
            if self.transform:
                patch = self.transform(patch)
            
            return patch, coord
            
        except Exception as e:
            logging.error(f"Error getting item {idx}: {str(e)}")
            raise

def create_data_loaders(data_dir, batch_size=32, num_workers=4, transform=None):
    """Create data loaders for training and validation"""
    try:
        # Create dataset
        dataset = WSIDataset(data_dir, transform=transform)
        
        # Split into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
        
    except Exception as e:
        logging.error(f"Error creating data loaders: {str(e)}")
        raise 