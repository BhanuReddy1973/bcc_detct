import numpy as np
import cv2
from typing import Tuple, Optional

class StainNormalizer:
    """
    Implements Macenko method for stain normalization
    Reference: https://www.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
    """
    
    def __init__(self, target_stains: Optional[np.ndarray] = None,
                 target_concentrations: Optional[np.ndarray] = None):
        """
        Initialize stain normalizer with optional target stains and concentrations.
        
        Args:
            target_stains: Target stain matrix (2x3)
            target_concentrations: Target stain concentrations
        """
        self.target_stains = target_stains
        self.target_concentrations = target_concentrations
        
    def get_stain_matrix(self, img: np.ndarray, 
                        luminosity_threshold: float = 0.8,
                        angular_percentile: float = 99) -> np.ndarray:
        """
        Estimate stain matrix for given image.
        
        Args:
            img: RGB image
            luminosity_threshold: Threshold for filtering bright pixels
            angular_percentile: Percentile for angle vectors
            
        Returns:
            2x3 stain matrix
        """
        # Convert to optical density
        od = -np.log((img.astype(float) + 1) / 256)
        
        # Remove pixels with low optical density
        od_reshaped = od.reshape((-1, 3))
        od_thresh = od_reshaped[np.all(od_reshaped > luminosity_threshold, axis=1)]
        
        # Compute eigenvectors
        _, eigvecs = np.linalg.eigh(np.cov(od_thresh.T))
        
        # Project data onto eigenvectors
        proj = np.dot(od_thresh, eigvecs)
        
        # Find angle of each point
        angles = np.arctan2(proj[:, 1], proj[:, 0])
        
        # Find extremes (stain vectors)
        angle_percentile = np.percentile(angles, [100-angular_percentile, angular_percentile])
        stain_vectors = eigvecs[:, [1, 0]]
        
        if stain_vectors[0, 0] < 0:
            stain_vectors *= -1
            
        return stain_vectors
    
    def normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize staining of the input image.
        
        Args:
            img: RGB image to normalize
            
        Returns:
            Normalized RGB image
        """
        # Estimate stain matrix if target not provided
        if self.target_stains is None:
            self.target_stains = self.get_stain_matrix(img)
            
        # Get source stain matrix
        source_stains = self.get_stain_matrix(img)
        
        # Convert to optical density
        od = -np.log((img.astype(float) + 1) / 256)
        
        # Get source concentrations
        source_concentrations = np.linalg.lstsq(source_stains.T, od.reshape(-1, 3).T, 
                                              rcond=None)[0]
        
        # If target concentrations not provided, use source
        if self.target_concentrations is None:
            self.target_concentrations = source_concentrations
            
        # Reconstruct image
        od_normalized = np.dot(self.target_concentrations.T, self.target_stains)
        img_normalized = np.exp(-od_normalized) * 256 - 1
        
        # Clip to valid range and convert to uint8
        img_normalized = np.clip(img_normalized, 0, 255).astype(np.uint8)
        
        return img_normalized 