import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
import cv2

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