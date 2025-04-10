import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import cv2

@dataclass
class PatchInfo:
    """Information about a tissue patch"""
    image: np.ndarray
    coordinates: Tuple[int, int]
    tissue_percentage: float

class TissuePacking:
    """Tissue packing module for organizing tissue regions into patches"""
    
    def __init__(self, 
                 patch_size: int = 224,
                 min_tissue_percentage: float = 0.7,
                 overlap: float = 0.5):
        self.patch_size = patch_size
        self.min_tissue_percentage = min_tissue_percentage
        self.overlap = overlap
        self.stride = int(patch_size * (1 - overlap))
    
    def extract_patches(self, 
                       image: np.ndarray, 
                       mask: np.ndarray) -> List[PatchInfo]:
        """Extract patches from the image based on tissue mask"""
        patches = []
        height, width = image.shape[:2]
        
        # Calculate number of patches
        n_patches_h = (height - self.patch_size) // self.stride + 1
        n_patches_w = (width - self.patch_size) // self.stride + 1
        
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                # Calculate patch coordinates
                y = i * self.stride
                x = j * self.stride
                
                # Extract patch and mask
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                patch_mask = mask[y:y+self.patch_size, x:x+self.patch_size]
                
                # Calculate tissue percentage
                tissue_percentage = np.sum(patch_mask > 0) / patch_mask.size
                
                # Keep patch if it meets tissue percentage threshold
                if tissue_percentage >= self.min_tissue_percentage:
                    patches.append(PatchInfo(
                        image=patch,
                        coordinates=(x, y),
                        tissue_percentage=tissue_percentage
                    ))
        
        return patches
    
    def resize_patches(self, patches: List[PatchInfo]) -> List[PatchInfo]:
        """Resize patches to the target size"""
        resized_patches = []
        for patch in patches:
            resized_image = cv2.resize(patch.image, (self.patch_size, self.patch_size))
            resized_patches.append(PatchInfo(
                image=resized_image,
                coordinates=patch.coordinates,
                tissue_percentage=patch.tissue_percentage
            ))
        return resized_patches 