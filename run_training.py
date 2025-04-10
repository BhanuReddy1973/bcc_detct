import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, SubsetRandomSampler, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from pathlib import Path
import numpy as np
from PIL import Image
import random
from sklearn.model_selection import ParameterGrid, KFold
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import logging
from datetime import datetime
import torch.cuda.amp as amp
import gc
import argparse
import socket
import time
import psutil
import subprocess
from models.bcc_model import BCCModel
from configs.config import Config

class BCCDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, split='train', num_samples=None, used_samples=None):       
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        self.samples = self._load_samples(num_samples, used_samples)
        print(f"Loaded {len(self.samples)} samples for {split} split")

    def _load_samples(self, num_samples=None, used_samples=None):
        samples = []

        # Load BCC (positive) samples
        bcc_dir = self.data_dir / "package" / "bcc" / "data" / "images"
        if bcc_dir.exists():
            for img_path in bcc_dir.glob("*.tif"):
                samples.append({
                    'image_path': str(img_path),
                    'label': 1  # BCC is positive class
                })

        # Load non-malignant (negative) samples
        normal_dir = self.data_dir / "package" / "non-malignant" / "data" / "images"
        if normal_dir.exists():
            for img_path in normal_dir.glob("*.tif"):
                samples.append({
                    'image_path': str(img_path),
                    'label': 0  # Non-malignant is negative class
                })

        print(f"Found {len(samples)} total samples before filtering")

        # Exclude previously used samples
        if used_samples:
            samples = [s for s in samples if s['image_path'] not in used_samples]

        # Randomly sample if num_samples is specified
        if num_samples is not None:
            # Ensure equal number of samples from each class
            bcc_samples = [s for s in samples if s['label'] == 1]
            normal_samples = [s for s in samples if s['label'] == 0]

            print(f"Found {len(bcc_samples)} BCC samples and {len(normal_samples)} normal samples")

            # Take equal number of samples from each class
            num_samples_per_class = num_samples // 2
            bcc_samples = random.sample(bcc_samples, min(num_samples_per_class, len(bcc_samples)))
            normal_samples = random.sample(normal_samples, min(num_samples_per_class, len(normal_samples))) 

            samples = bcc_samples + normal_samples
            print(f"Selected {len(samples)} samples ({len(bcc_samples)} BCC, {len(normal_samples)} normal)")

        # Split into train/val/test
        random.shuffle(samples)
        n = len(samples)
        if self.split == 'train':
            return samples[:int(0.7*n)]
        elif self.split == 'val':
            return samples[int(0.7*n):int(0.85*n)]
        else:  # test
            return samples[int(0.85*n):]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            # Load and resize the image
            image = Image.open(sample['image_path'])

            # Get original dimensions
            width, height = image.size

            # Calculate new dimensions while maintaining aspect ratio
            target_size = 4096  # Maximum dimension
            if width > height:
                new_width = target_size
                new_height = int(height * (target_size / width))
            else:
                new_height = target_size
                new_width = int(width * (target_size / height))

            # Resize the image
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to RGB
            image = image.convert('RGB')

            # Extract a random 224x224 patch
            if new_width > 224 and new_height > 224:
                left = random.randint(0, new_width - 224)
                top = random.randint(0, new_height - 224)
                image = image.crop((left, top, left + 224, top + 224))
            else:
                # If image is smaller than 224x224, pad it
                image = image.resize((224, 224), Image.Resampling.LANCZOS)

            # Apply transforms if specified
            if self.transform:
                image = self.transform(image)

            return image, sample['label']

        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {str(e)}")
            # Return a black image and label 0 in case of error
            return torch.zeros((3, 224, 224)), 0

def create_data_loaders(data_dir, batch_size, num_samples=None, used_samples=None):
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = BCCDataset(data_dir, transform=transform, split='train', 
                             num_samples=num_samples, used_samples=used_samples)
    val_dataset = BCCDataset(data_dir, transform=transform, split='val', 
                           num_samples=num_samples, used_samples=used_samples)
    test_dataset = BCCDataset(data_dir, transform=transform, split='test', 
                            num_samples=num_samples, used_samples=used_samples)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader 

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(train_loader), 100 * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(val_loader), 100 * correct / total

def train_with_hyperparams(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=20, iteration=1):
    """Train model with given hyperparameters"""
    best_val_acc = 0.0
    best_model_path = None
    scaler = amp.GradScaler()

    for epoch in range(num_epochs):
        train_loss, train_acc, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc, val_preds, val_labels = validate(
            model, test_loader, criterion, device
        )

        logging.info(f'Epoch {epoch+1}/{num_epochs}:')
        logging.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logging.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = f'best_model_iteration_{iteration}.pth'
            torch.save(model.state_dict(), best_model_path)
            logging.info(f'New best model saved with validation accuracy: {val_acc:.2f}%')

    return best_model_path

def plot_confusion_matrix(y_true, y_pred, iteration=1):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot
    plots_dir = Path(__file__).parent / "visualizations"
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f'confusion_matrix_iteration_{iteration}.png')
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train BCC detection model')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to use')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data loaders
    data_dir = "dataset"  # Update with your dataset path
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir, args.batch_size, args.num_samples, num_workers=args.num_workers
    )

    # Initialize model
    model = BCCModel().to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')

if __name__ == '__main__':
    main() 