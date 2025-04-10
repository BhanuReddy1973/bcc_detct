import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import opening, closing, disk
from skimage.measure import label, regionprops
from pathlib import Path
import logging

class TissueSegmenter:
    def __init__(self):
        self.kernel = disk(5)  # For morphological operations
    
    def color_deconvolution(self, image):
        """Separate H&E components"""
        # Convert to optical density space
        od = -np.log((image.astype(float) + 1) / 256)
        
        # H&E color vectors
        he_vectors = np.array([
            [0.65, 0.70, 0.29],  # Hematoxylin
            [0.07, 0.99, 0.11]   # Eosin
        ])
        
        # Perform color deconvolution
        he_stains = np.dot(od, he_vectors.T)
        return he_stains[:, :, 0]  # Return hematoxylin channel
    
    def segment_tissue(self, image):
        """Segment tissue using Otsu's thresholding and morphological operations"""
        try:
            # Get hematoxylin channel
            he_channel = self.color_deconvolution(image)
            
            # Apply Otsu's thresholding
            threshold = threshold_otsu(he_channel)
            binary = he_channel > threshold
            
            # Apply morphological operations
            binary = opening(binary, self.kernel)
            binary = closing(binary, self.kernel)
            
            # Connected component analysis
            labeled = label(binary)
            regions = regionprops(labeled)
            
            # Filter small regions
            min_area = 5000  # Minimum area threshold
            mask = np.zeros_like(binary)
            for region in regions:
                if region.area >= min_area:
                    mask[labeled == region.label] = 1
            
            return mask
            
        except Exception as e:
            logging.error(f"Error in tissue segmentation: {str(e)}")
            raise

class TissuePacker:
    def __init__(self, patch_size=224, min_tissue_ratio=0.7):
        self.patch_size = patch_size
        self.min_tissue_ratio = min_tissue_ratio
    
    def extract_patches(self, image, mask):
        """Extract patches from tissue regions"""
        patches = []
        coordinates = []
        
        height, width = image.shape[:2]
        
        # Calculate step size for 50% overlap
        step = self.patch_size // 2
        
        for y in range(0, height - self.patch_size + 1, step):
            for x in range(0, width - self.patch_size + 1, step):
                # Extract patch
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                
                # Check if patch contains enough tissue
                patch_mask = mask[y:y+self.patch_size, x:x+self.patch_size]
                tissue_ratio = np.sum(patch_mask) / (self.patch_size * self.patch_size)
                
                if tissue_ratio >= self.min_tissue_ratio:
                    patches.append(patch)
                    coordinates.append((x, y))
        
        return patches, coordinates
    
    def process_slide(self, slide_path):
        """Process a whole slide image"""
        try:
            # Read slide (using OpenSlide or similar)
            # This is a placeholder - actual implementation depends on the slide format
            image = cv2.imread(str(slide_path))
            
            # Segment tissue
            segmenter = TissueSegmenter()
            mask = segmenter.segment_tissue(image)
            
            # Extract patches
            patches, coordinates = self.extract_patches(image, mask)
            
            return patches, coordinates
            
        except Exception as e:
            logging.error(f"Error processing slide {slide_path}: {str(e)}")
            raise 