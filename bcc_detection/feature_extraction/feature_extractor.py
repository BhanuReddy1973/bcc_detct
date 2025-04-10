import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import cv2
from skfuzzy.cluster import cmeans

class FeatureExtractor:
    """Feature extraction module combining deep and color features"""
    
    def __init__(self, 
                 n_components: int = 256,
                 n_clusters: int = 3,
                 fuzziness: float = 2.0):
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.fuzziness = fuzziness
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
    
    def extract_color_features(self, image: np.ndarray) -> np.ndarray:
        """Extract color-based features from image"""
        features = []
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        
        # Extract features from each color space
        for color_space in [image, hsv, lab, ycrcb]:
            for channel in range(3):
                channel_data = color_space[:, :, channel]
                
                # First-order statistics
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.percentile(channel_data, 25),
                    np.percentile(channel_data, 75)
                ])
                
                # Histogram features
                hist = cv2.calcHist([channel_data], [0], None, [32], [0, 256])
                hist = hist.flatten() / hist.sum()
                features.extend(hist)
        
        return np.array(features)
    
    def apply_pca(self, features: np.ndarray) -> np.ndarray:
        """Apply PCA to reduce feature dimensionality"""
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Apply PCA
        reduced_features = self.pca.fit_transform(scaled_features)
        
        return reduced_features
    
    def fuzzy_clustering(self, features: np.ndarray) -> np.ndarray:
        """Apply Fuzzy C-Means clustering to color features"""
        # Transpose features for FCM (samples x features)
        features_t = features.T
        
        # Apply FCM
        cntr, u, _, _, _, _, _ = cmeans(
            features_t,
            self.n_clusters,
            self.fuzziness,
            error=0.005,
            maxiter=1000
        )
        
        # Return membership values
        return u.T
    
    def extract_features(self, 
                        deep_features: np.ndarray,
                        image: np.ndarray) -> np.ndarray:
        """Extract and combine all features"""
        # Extract color features
        color_features = self.extract_color_features(image)
        
        # Apply PCA to color features
        reduced_color_features = self.apply_pca(color_features.reshape(1, -1))
        
        # Apply FCM to color features
        fcm_features = self.fuzzy_clustering(reduced_color_features.T)
        
        # Concatenate features
        combined_features = np.concatenate([
            deep_features,
            fcm_features.flatten()
        ])
        
        return combined_features 