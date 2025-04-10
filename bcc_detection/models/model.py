import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional
from ..feature_extraction.feature_extractor import FeatureExtractor
import numpy as np

class BCCDetectionModel(nn.Module):
    """
    BCC Detection Model using EfficientNet as backbone with feature extraction and classification.
    
    Args:
        backbone: Name of the EfficientNet backbone to use
        num_classes: Number of output classes
        dropout_rate: Dropout rate for the classifier
        feature_dim: Dimension of the feature vector
    """
    def __init__(
        self,
        backbone: str = "efficientnet_b7",
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        feature_dim: int = 256
    ):
        super().__init__()
        
        # Load pretrained EfficientNet
        self.backbone = getattr(models, backbone)(pretrained=True)
        
        # Remove classification head
        self.features = self.backbone.features
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + 3, 512),  # 256 deep features + 3 FCM features
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Extract deep features
        features = self.features(x)
        features = self.gap(features)
        features = features.view(features.size(0), -1)
        
        # Extract and combine features
        combined_features = []
        for i in range(x.size(0)):
            # Convert tensor to numpy for feature extraction
            img = x[i].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            
            # Extract and combine features
            combined_feature = self.feature_extractor.extract_features(
                features[i].detach().cpu().numpy(),
                img
            )
            combined_features.append(combined_feature)
        
        # Convert back to tensor
        combined_features = torch.tensor(
            np.array(combined_features),
            device=x.device,
            dtype=torch.float32
        )
        
        # Classification
        output = self.classifier(combined_features)
        
        return output
    
    def freeze_backbone(self) -> None:
        """Freeze the backbone layers."""
        for param in self.features.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self) -> None:
        """Unfreeze the backbone layers."""
        for param in self.features.parameters():
            param.requires_grad = True

def create_model(config):
    """Create and initialize the BCC detection model"""
    model = BCCDetectionModel(config["backbone"], config["num_classes"], config["dropout_rate"], config["feature_dim"])
    
    # Initialize weights for the classification head
    nn.init.normal_(model.classifier[-1].weight, 0, 0.01)
    nn.init.zeros_(model.classifier[-1].bias)
    
    return model

def create_model_with_backbone(config):
    """Create and initialize the BCC detection model with the backbone frozen"""
    model = BCCDetectionModel(config["backbone"], config["num_classes"], config["dropout_rate"], config["feature_dim"])
    
    # Freeze all layers except the classification head
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Initialize weights for the classification head
    nn.init.normal_(model.classifier[-1].weight, 0, 0.01)
    nn.init.zeros_(model.classifier[-1].bias)
    
    return model

def unfreeze_backbone(model):
    """Unfreeze all layers"""
    for param in model.features.parameters():
        param.requires_grad = True 