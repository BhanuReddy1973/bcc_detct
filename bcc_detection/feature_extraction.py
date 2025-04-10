import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
import cv2
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
from sklearn.decomposition import PCA
from skimage.color import rgb2hsv, rgb2lab, rgb2ycbcr
import logging

class FeatureExtractor(nn.Module):
    """Feature extractor using pre-trained ResNet50."""
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.stack([self.transform(img) for img in x])
        return self.features(x).squeeze()

def extract_resnet_features(patches):
    """Extract features using pre-trained ResNet50."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = FeatureExtractor().to(device)
    extractor.eval()
    
    with torch.no_grad():
        features = extractor(patches)
    return features.cpu().numpy()

def extract_traditional_features(patches):
    """Extract traditional computer vision features."""
    features = []
    for patch in patches:
        # Color features
        color_features = extract_color_features(patch)
        
        # Texture features
        texture_features = extract_texture_features(patch)
        
        # Shape features
        shape_features = extract_shape_features(patch)
        
        # Combine all features
        patch_features = np.concatenate([
            color_features,
            texture_features,
            shape_features
        ])
        features.append(patch_features)
    
    return np.array(features)

def extract_color_features(patch):
    """Extract color-based features."""
    # Mean and standard deviation of each channel
    mean = np.mean(patch, axis=(0, 1))
    std = np.std(patch, axis=(0, 1))
    
    # Color histogram features
    hist_r = np.histogram(patch[:,:,0], bins=8, range=(0, 256))[0]
    hist_g = np.histogram(patch[:,:,1], bins=8, range=(0, 256))[0]
    hist_b = np.histogram(patch[:,:,2], bins=8, range=(0, 256))[0]
    
    return np.concatenate([mean, std, hist_r, hist_g, hist_b])

def extract_texture_features(patch):
    """Extract texture-based features using GLCM."""
    gray = rgb2gray(patch)
    gray = (gray * 255).astype(np.uint8)
    
    # Calculate GLCM
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])
    
    # Extract texture properties
    contrast = graycoprops(glcm, 'contrast').ravel()
    dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
    homogeneity = graycoprops(glcm, 'homogeneity').ravel()
    energy = graycoprops(glcm, 'energy').ravel()
    correlation = graycoprops(glcm, 'correlation').ravel()
    
    return np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation])

def extract_shape_features(patch):
    """Extract shape-based features."""
    gray = rgb2gray(patch)
    gray = (gray * 255).astype(np.uint8)
    
    # Threshold and find contours
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return np.zeros(5)
    
    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Calculate shape features
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    _, (width, height), _ = cv2.minAreaRect(contour)
    aspect_ratio = float(width) / height if height > 0 else 0
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    return np.array([area, perimeter, width, height, aspect_ratio, circularity])

def extract_hybrid_features(patches):
    """Extract both deep learning and traditional features."""
    # Extract ResNet features
    resnet_features = extract_resnet_features(patches)
    
    # Extract traditional features
    traditional_features = extract_traditional_features(patches)
    
    # Combine features
    hybrid_features = np.concatenate([resnet_features, traditional_features], axis=1)
    
    return hybrid_features

class FeatureExtractor:
    def __init__(self, pca_components=256):
        self.pca_components = pca_components
        self.pca = None
        self._init_efficientnet()
    
    def _init_efficientnet(self):
        """Initialize EfficientNet-B7 with pretrained weights"""
        self.efficientnet = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
        # Remove classification head
        self.efficientnet.classifier = nn.Identity()
        self.efficientnet.eval()
    
    def extract_deep_features(self, patches):
        """Extract features using EfficientNet-B7"""
        try:
            # Convert patches to tensor
            patches_tensor = torch.stack([self._preprocess_patch(p) for p in patches])
            
            # Extract features
            with torch.no_grad():
                features = self.efficientnet(patches_tensor)
            
            return features.numpy()
            
        except Exception as e:
            logging.error(f"Error extracting deep features: {str(e)}")
            raise
    
    def _preprocess_patch(self, patch):
        """Preprocess patch for EfficientNet"""
        # Convert to tensor and normalize
        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
        patch_tensor = patch_tensor / 255.0
        patch_tensor = (patch_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
                      torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return patch_tensor
    
    def extract_color_features(self, patches):
        """Extract color-based features"""
        color_features = []
        
        for patch in patches:
            # Convert to different color spaces
            hsv = rgb2hsv(patch)
            lab = rgb2lab(patch)
            ycbcr = rgb2ycbcr(patch)
            
            # Extract statistical features
            stats = []
            for channel in [patch, hsv, lab, ycbcr]:
                for c in range(channel.shape[2]):
                    stats.extend([
                        np.mean(channel[:,:,c]),
                        np.std(channel[:,:,c]),
                        np.median(channel[:,:,c]),
                        np.percentile(channel[:,:,c], 25),
                        np.percentile(channel[:,:,c], 75)
                    ])
            
            # Extract texture features using GLCM
            gray = np.mean(patch, axis=2)
            glcm = graycomatrix(gray.astype(np.uint8), [1], [0], 256, symmetric=True, normed=True)
            texture_features = [
                graycoprops(glcm, 'contrast')[0, 0],
                graycoprops(glcm, 'dissimilarity')[0, 0],
                graycoprops(glcm, 'homogeneity')[0, 0],
                graycoprops(glcm, 'energy')[0, 0],
                graycoprops(glcm, 'correlation')[0, 0]
            ]
            
            # Combine all features
            features = np.concatenate([stats, texture_features])
            color_features.append(features)
        
        return np.array(color_features)
    
    def fit_pca(self, features):
        """Fit PCA on training features"""
        self.pca = PCA(n_components=self.pca_components)
        self.pca.fit(features)
    
    def transform_features(self, deep_features, color_features):
        """Transform features using PCA and concatenate"""
        try:
            # Reduce dimensionality of deep features
            if self.pca is None:
                raise ValueError("PCA not fitted. Call fit_pca first.")
            
            reduced_deep_features = self.pca.transform(deep_features)
            
            # Concatenate features
            combined_features = np.concatenate([reduced_deep_features, color_features], axis=1)
            
            return combined_features
            
        except Exception as e:
            logging.error(f"Error transforming features: {str(e)}")
            raise

class FuzzyCMeans:
    def __init__(self, n_clusters=3, m=2, max_iter=100):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.centers = None
    
    def fit(self, X):
        """Fit FCM model"""
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Initialize membership matrix
        U = np.random.rand(n_samples, self.n_clusters)
        U = U / np.sum(U, axis=1, keepdims=True)
        
        for _ in range(self.max_iter):
            # Update centers
            centers = np.zeros((self.n_clusters, n_features))
            for j in range(self.n_clusters):
                centers[j] = np.sum((U[:, j] ** self.m)[:, np.newaxis] * X, axis=0) / \
                           np.sum(U[:, j] ** self.m)
            
            # Update membership matrix
            distances = np.zeros((n_samples, self.n_clusters))
            for j in range(self.n_clusters):
                distances[:, j] = np.sum((X - centers[j]) ** 2, axis=1)
            
            U_new = np.zeros_like(U)
            for j in range(self.n_clusters):
                U_new[:, j] = 1.0 / np.sum((distances[:, j, np.newaxis] / distances) ** (2 / (self.m - 1)), axis=1)
            
            # Check convergence
            if np.max(np.abs(U_new - U)) < 1e-5:
                break
            
            U = U_new
        
        self.centers = centers
        return U
    
    def predict(self, X):
        """Predict membership values for new data"""
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        
        for j in range(self.n_clusters):
            distances[:, j] = np.sum((X - self.centers[j]) ** 2, axis=1)
        
        U = np.zeros_like(distances)
        for j in range(self.n_clusters):
            U[:, j] = 1.0 / np.sum((distances[:, j, np.newaxis] / distances) ** (2 / (self.m - 1)), axis=1)
        
        return U 