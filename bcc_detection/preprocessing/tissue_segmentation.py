import cv2
import numpy as np
from skimage import morphology
from typing import Tuple, Optional
from PIL import Image

class TissueSegmentation:
    """Tissue segmentation module for TIF image preprocessing"""
    
    def __init__(self, min_tissue_area: int = 1000):
        self.min_tissue_area = min_tissue_area
        
    def color_deconvolution(self, image: np.ndarray) -> np.ndarray:
        """Separate hematoxylin channel from H&E stained image"""
        # Convert to optical density space
        od = -np.log(image / 255.0 + 1e-6)
        
        # H&E color vectors
        he_vectors = np.array([
            [0.65, 0.70, 0.29],  # Hematoxylin
            [0.07, 0.99, 0.11]   # Eosin
        ])
        
        # Normalize vectors
        he_vectors = he_vectors / np.linalg.norm(he_vectors, axis=1)[:, np.newaxis]
        
        # Project onto hematoxylin vector
        hematoxylin = np.dot(od, he_vectors[0])
        
        return hematoxylin
    
    def otsu_thresholding(self, image: np.ndarray) -> np.ndarray:
        """Apply Otsu's thresholding"""
        # Normalize to 0-255
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def morphological_operations(self, binary: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean the mask"""
        # Remove small objects
        cleaned = morphology.remove_small_objects(binary.astype(bool), min_size=self.min_tissue_area)
        
        # Fill holes
        filled = morphology.remove_small_holes(cleaned, area_threshold=self.min_tissue_area)
        
        return filled.astype(np.uint8) * 255
    
    def segment_tissue(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Segment tissue from background"""
        # Color deconvolution
        hematoxylin = self.color_deconvolution(image)
        
        # Otsu's thresholding
        binary = self.otsu_thresholding(hematoxylin)
        
        # Morphological operations
        mask = self.morphological_operations(binary)
        
        # Calculate tissue percentage
        tissue_percentage = np.sum(mask > 0) / mask.size
        
        return mask, tissue_percentage

def get_tissue_mask(image_path: str, otsu_threshold: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    Generate a binary mask for tissue regions in a TIF image.
    
    Args:
        image_path: Path to the TIF image
        otsu_threshold: Optional manual threshold value
        
    Returns:
        Tuple of (binary mask, computed threshold)
    """
    # Open the image
    img = np.array(Image.open(image_path))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Otsu's thresholding
    if otsu_threshold is None:
        threshold, mask = cv2.threshold(blurred, 0, 255, 
                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        threshold = otsu_threshold
        _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Apply morphological operations
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask, threshold

def filter_tissue_patches(mask: np.ndarray, patch_size: int, 
                         min_tissue_percentage: float = 0.5) -> np.ndarray:
    """
    Filter patches based on tissue content.
    
    Args:
        mask: Binary tissue mask
        patch_size: Size of patches to analyze
        min_tissue_percentage: Minimum required tissue percentage (0-1)
        
    Returns:
        Boolean array indicating valid patches
    """
    # Calculate the number of patches in each dimension
    n_rows = mask.shape[0] // patch_size
    n_cols = mask.shape[1] // patch_size
    
    # Initialize output array
    valid_patches = np.zeros((n_rows, n_cols), dtype=bool)
    
    # Analyze each patch
    for i in range(n_rows):
        for j in range(n_cols):
            # Extract patch
            patch = mask[i*patch_size:(i+1)*patch_size, 
                        j*patch_size:(j+1)*patch_size]
            
            # Calculate tissue percentage
            tissue_percentage = np.sum(patch > 0) / (patch_size * patch_size)
            
            # Mark patch as valid if it contains enough tissue
            valid_patches[i, j] = tissue_percentage >= min_tissue_percentage
            
    return valid_patches 