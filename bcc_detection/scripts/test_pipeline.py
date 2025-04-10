import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from datetime import datetime
import cv2
from pathlib import Path
import shutil
import openslide
from glob import glob
from typing import List, Tuple

from ..models.model import BCCDetectionModel, create_model, create_model_with_backbone, unfreeze_backbone
from ..data.data_loader import create_data_loaders
from ..configs.config import get_config, update_data_paths
from ..preprocessing.tissue_segmentation import TissueSegmentation, get_tissue_mask, filter_tissue_patches
from ..preprocessing.tissue_packing import TissuePacking
from ..feature_extraction.feature_extractor import FeatureExtractor

def load_dataset_images(dataset_path: str, num_images: int = 100) -> Tuple[List[str], List[int]]:
    """Load images from the dataset directory"""
    # Get all TIF files from both BCC and non-malignant directories
    bcc_images = glob(os.path.join(dataset_path, "bcc", "*.tif"))
    non_malignant_images = glob(os.path.join(dataset_path, "non-malignant", "*.tif"))
    
    # Randomly select images
    np.random.shuffle(bcc_images)
    np.random.shuffle(non_malignant_images)
    
    # Take equal number of images from each class
    num_per_class = num_images // 2
    selected_bcc = bcc_images[:num_per_class]
    selected_non_malignant = non_malignant_images[:num_per_class]
    
    # Combine images and create labels
    images = selected_bcc + selected_non_malignant
    labels = [1] * len(selected_bcc) + [0] * len(selected_non_malignant)
    
    return images, labels

def create_test_data(dataset_path: str, num_images: int = 200):
    """Create test data from actual dataset images"""
    # Load images and labels
    images, labels = load_dataset_images(dataset_path, num_images)
    
    # Split into train/val/test (70/15/15)
    indices = np.random.permutation(len(images))
    train_idx = indices[:int(0.7 * len(images))]
    val_idx = indices[int(0.7 * len(images)):int(0.85 * len(images))]
    test_idx = indices[int(0.85 * len(images)):]
    
    # Split the data
    train_slides = [images[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    
    val_slides = [images[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    
    test_slides = [images[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    
    return (
        train_slides, train_labels,
        val_slides, val_labels,
        test_slides, test_labels
    )

def test_tissue_segmentation(slide_path: str):
    """Test tissue segmentation methods on actual WSI"""
    print("\nTesting tissue segmentation...")
    segmenter = TissueSegmentation()
    
    # Open the slide
    slide = openslide.OpenSlide(slide_path)
    
    # Read at level 0 (highest resolution)
    img = np.array(slide.read_region((0, 0), 0, slide.level_dimensions[0]))
    
    # Test segment_tissue
    mask, tissue_percentage = segmenter.segment_tissue(img)
    print(f"Tissue percentage: {tissue_percentage:.2f}")
    
    # Test get_tissue_mask
    mask, threshold = get_tissue_mask(slide_path)
    print(f"Otsu threshold: {threshold}")
    
    # Test filter_tissue_patches
    valid_patches = filter_tissue_patches(mask, patch_size=224, min_tissue_percentage=0.7)
    print(f"Valid patches: {np.sum(valid_patches)}")
    
    return mask

def test_feature_extraction(slide_path: str):
    """Test feature extraction methods on actual WSI"""
    print("\nTesting feature extraction...")
    extractor = FeatureExtractor()
    
    # Open the slide
    slide = openslide.OpenSlide(slide_path)
    
    # Read at level 0 (highest resolution)
    img = np.array(slide.read_region((0, 0), 0, slide.level_dimensions[0]))
    
    # Test color feature extraction
    color_features = extractor.extract_color_features(img)
    print(f"Color features shape: {color_features.shape}")
    
    # Test PCA
    reduced_features = extractor.apply_pca(color_features.reshape(1, -1))
    print(f"Reduced features shape: {reduced_features.shape}")
    
    # Test FCM
    fcm_features = extractor.fuzzy_clustering(reduced_features.T)
    print(f"FCM features shape: {fcm_features.shape}")
    
    return reduced_features, fcm_features

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': total_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
    
    return total_loss / len(val_loader), 100. * correct / total

def test_pipeline():
    """Test the complete pipeline with actual dataset"""
    print("Starting pipeline test...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create test data from actual dataset
    print("Loading dataset images...")
    dataset_path = os.path.join("..", "dataset", "package")
    train_slides, train_labels, val_slides, val_labels, test_slides, test_labels = create_test_data(dataset_path)
    
    # Test tissue segmentation on first training slide
    mask = test_tissue_segmentation(train_slides[0])
    
    # Test feature extraction on first training slide
    reduced_features, fcm_features = test_feature_extraction(train_slides[0])
    
    # Get configuration
    config = get_config()
    
    # Update configuration with test data paths
    config = update_data_paths(
        config,
        train_slides, train_labels,
        val_slides, val_labels,
        test_slides, test_labels
    )
    
    # Data augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_slides=config["train_slides"],
        train_labels=config["train_labels"],
        val_slides=config["val_slides"],
        val_labels=config["val_labels"],
        test_slides=config["test_slides"],
        test_labels=config["test_labels"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        patch_size=config["patch_size"],
        min_tissue_percentage=config["min_tissue_percentage"],
        overlap=config["overlap"],
        transform=transform
    )
    
    # Test different model creation methods
    print("\nTesting model creation methods...")
    model1 = create_model(config).to(device)
    model2 = create_model_with_backbone(config).to(device)
    
    # Create main model
    print("Initializing main model...")
    model = BCCDetectionModel(
        backbone=config["backbone"],
        num_classes=config["num_classes"],
        dropout_rate=config["dropout_rate"],
        feature_dim=config["feature_dim"]
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.1,
        patience=5,
        verbose=True
    )
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(config["checkpoint_dir"], timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
    
    # Test the model
    print("\nTesting model...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    print("Pipeline test completed!")

if __name__ == "__main__":
    test_pipeline() 