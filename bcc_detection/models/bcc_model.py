import torch
import torch.nn as nn
import torchvision.models as models
from ..configs.config import Config

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.attention(x)

class BCCModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained EfficientNet
        self.backbone = getattr(models, Config.MODEL_NAME)(pretrained=Config.PRETRAINED)
        
        # Get the number of features from the backbone
        if Config.MODEL_NAME == 'efficientnet_b4':
            num_features = 1792
        else:
            raise ValueError(f"Unsupported model: {Config.MODEL_NAME}")
        
        # Add attention mechanism
        self.attention = AttentionModule(num_features)
        
        # Modify the classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, Config.NUM_CLASSES)
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone.features(x)
        
        # Apply attention
        attended_features = self.attention(features)
        
        # Global average pooling
        pooled_features = self.backbone.avgpool(attended_features)
        pooled_features = torch.flatten(pooled_features, 1)
        
        # Classification
        output = self.backbone.classifier(pooled_features)
        
        return output
    
    def save(self, path):
        """Save model state"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': Config.__dict__
        }, path)
    
    @classmethod
    def load(cls, path):
        """Load model from saved state"""
        checkpoint = torch.load(path)
        model = cls()
        model.load_state_dict(checkpoint['model_state_dict'])
        return model 