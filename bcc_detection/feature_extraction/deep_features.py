import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import List, Optional, Union, Tuple
import numpy as np

class DeepFeatureExtractor:
    """
    Feature extractor using EfficientNet-B7 backbone
    """
    
    def __init__(self, device: Optional[str] = None,
                 feature_layers: Optional[List[str]] = None):
        """
        Initialize the feature extractor.
        
        Args:
            device: Device to use ('cuda' or 'cpu')
            feature_layers: List of layer names to extract features from
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pretrained model
        self.model = models.efficientnet_b7(pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define feature layers if not provided
        self.feature_layers = feature_layers or ['features.6', 'features.7']
        
        # Initialize feature maps dictionary
        self.feature_maps = {}
        
        # Register hooks
        self._register_hooks()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def _register_hooks(self):
        """Register forward hooks for feature extraction"""
        def get_features(name):
            def hook(model, input, output):
                self.feature_maps[name] = output.detach()
            return hook
        
        for name in self.feature_layers:
            layer = dict([*self.model.named_modules()])[name]
            layer.register_forward_hook(get_features(name))
            
    def extract_features(self, image: Union[np.ndarray, torch.Tensor],
                        return_predictions: bool = False) -> Union[dict, Tuple[dict, torch.Tensor]]:
        """
        Extract features from the input image.
        
        Args:
            image: Input image (HxWxC numpy array or CxHxW tensor)
            return_predictions: Whether to return model predictions
            
        Returns:
            Dictionary of features or tuple of (features, predictions)
        """
        # Clear previous features
        self.feature_maps.clear()
        
        # Prepare input
        if isinstance(image, np.ndarray):
            image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            predictions = self.model(image)
            
        # Get features
        features = {name: self.feature_maps[name].cpu().numpy()
                   for name in self.feature_layers}
        
        if return_predictions:
            return features, predictions.cpu()
        return features
    
    def get_feature_dimensions(self) -> dict:
        """
        Get the dimensions of features from each layer.
        
        Returns:
            Dictionary of feature dimensions
        """
        return {name: dict([*self.model.named_modules()])[name].out_channels
                for name in self.feature_layers}
                
    def extract_batch_features(self, images: Union[np.ndarray, torch.Tensor],
                             batch_size: int = 32) -> dict:
        """
        Extract features from a batch of images.
        
        Args:
            images: Batch of images (BxHxWxC numpy array or BxCxHxW tensor)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary of features for each layer
        """
        # Initialize feature storage
        all_features = {name: [] for name in self.feature_layers}
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            # Extract features for batch
            batch_features = self.extract_features(batch)
            
            # Store features
            for name in self.feature_layers:
                all_features[name].append(batch_features[name])
        
        # Concatenate features
        for name in self.feature_layers:
            all_features[name] = np.concatenate(all_features[name], axis=0)
            
        return all_features 