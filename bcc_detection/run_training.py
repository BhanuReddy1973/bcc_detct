import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import numpy as np
from PIL import Image
import random
from sklearn.model_selection import ParameterGrid
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import logging
from datetime import datetime
import torch.cuda.amp as amp

# Get the base directory
BASE_DIR = Path(__file__).parent.absolute()

# Disable PIL's decompression bomb check
Image.MAX_IMAGE_PIXELS = None

# Set up logging
def setup_logging():
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

# Visualization functions
def plot_metrics(train_losses, val_losses, train_accs, val_accs, iteration):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    vis_dir = BASE_DIR / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    plt.savefig(vis_dir / f'metrics_iteration_{iteration}.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, iteration):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    vis_dir = BASE_DIR / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    plt.savefig(vis_dir / f'confusion_matrix_iteration_{iteration}.png')
    plt.close()

class BCCDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, split='train', num_samples=None, used_samples=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        self.samples = self._load_samples(num_samples, used_samples)
    
    def _load_samples(self, num_samples=None, used_samples=None):
        samples = []
        
        # Load BCC (positive) samples
        bcc_dir = self.data_dir / "package" / "bcc" / "data" / "images"
        for img_path in bcc_dir.glob("*.tif"):
            samples.append({
                'image_path': str(img_path),
                'label': 1  # BCC is positive class
            })
        
        # Load non-malignant (negative) samples
        normal_dir = self.data_dir / "package" / "non-malignant" / "data" / "images"
        for img_path in normal_dir.glob("*.tif"):
            samples.append({
                'image_path': str(img_path),
                'label': 0  # Non-malignant is negative class
            })
        
        # Exclude previously used samples
        if used_samples:
            samples = [s for s in samples if s['image_path'] not in used_samples]
        
        # Randomly sample if num_samples is specified
        if num_samples is not None:
            # Ensure equal number of samples from each class
            bcc_samples = [s for s in samples if s['label'] == 1]
            normal_samples = [s for s in samples if s['label'] == 0]
            
            # Take equal number of samples from each class
            num_samples_per_class = num_samples // 2
            bcc_samples = random.sample(bcc_samples, min(num_samples_per_class, len(bcc_samples)))
            normal_samples = random.sample(normal_samples, min(num_samples_per_class, len(normal_samples)))
            
            samples = bcc_samples + normal_samples
        
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
        if self.transform:
            image = self.transform(image)
        
        return image, sample['label']

def create_data_loaders(data_dir, batch_size=32, num_samples=None, used_samples=None):
    # Define transforms
    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = BCCDataset(data_dir, transform, 'train', num_samples, used_samples)
    val_dataset = BCCDataset(data_dir, val_transform, 'val', num_samples, used_samples)
    test_dataset = BCCDataset(data_dir, val_transform, 'test', num_samples, used_samples)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# Modified BCCModel with GPU support
class BCCModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 28 * 28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train_epoch(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    return running_loss / len(train_loader), 100. * correct / total, all_preds, all_labels

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss / len(val_loader), 100. * correct / total, all_preds, all_labels

def train_with_hyperparams(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, iteration):
    best_val_acc = 0
    best_model = None
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Initialize gradient scaler for mixed precision training
    scaler = amp.GradScaler()
    
    for epoch in range(num_epochs):
        train_loss, train_acc, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        logging.info(f"\nEpoch {epoch+1}/{num_epochs}:")
        logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
            logging.info("New best model saved!")
            
            # Save confusion matrix for best model
            plot_confusion_matrix(val_labels, val_preds, f"{iteration}_epoch_{epoch+1}")
    
    # Plot metrics for this training run
    plot_metrics(train_losses, val_losses, train_accs, val_accs, iteration)
    
    return best_model, best_val_acc

def hyperparameter_tuning(train_loader, val_loader, device, iteration):
    param_grid = {
        'learning_rate': [0.001, 0.0005, 0.0001],
        'dropout_rate': [0.3, 0.5, 0.7],
        'batch_size': [16, 32, 64]
    }
    
    best_acc = 0
    best_params = None
    best_model = None
    
    for params in ParameterGrid(param_grid):
        logging.info(f"\nTrying parameters: {params}")
        
        model = BCCModel(dropout_rate=params['dropout_rate']).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        model, val_acc = train_with_hyperparams(
            model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, iteration=iteration
        )
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = params
            best_model = model
            logging.info(f"New best parameters found! Accuracy: {best_acc:.2f}%")
    
    return best_model, best_params, best_acc

def main():
    # Set up logging
    log_file = setup_logging()
    
    # Create necessary directories
    (BASE_DIR / "visualizations").mkdir(exist_ok=True)
    (BASE_DIR / "models").mkdir(exist_ok=True)
    (BASE_DIR / "results").mkdir(exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    data_dir = BASE_DIR / "dataset"  # Using relative path
    used_samples = set()
    num_iterations = 10
    total_samples = 100
    
    # Store results for all iterations
    all_results = []
    
    for iteration in range(num_iterations):
        logging.info(f"\nStarting iteration {iteration + 1}/{num_iterations}")
        
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir,
            batch_size=32,
            num_samples=total_samples,
            used_samples=used_samples
        )
        
        for loader in [train_loader, val_loader, test_loader]:
            for sample in loader.dataset.samples:
                used_samples.add(sample['image_path'])
        
        logging.info(f"Train samples: {len(train_loader.dataset)}")
        logging.info(f"Validation samples: {len(val_loader.dataset)}")
        logging.info(f"Test samples: {len(test_loader.dataset)}")
        
        best_model, best_params, best_val_acc = hyperparameter_tuning(
            train_loader, val_loader, device, iteration
        )
        
        logging.info(f"\nBest parameters found: {best_params}")
        logging.info(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, test_preds, test_labels = validate(best_model, test_loader, criterion, device)
        
        # Save test results
        results = {
            'iteration': iteration + 1,
            'best_params': best_params,
            'validation_accuracy': best_val_acc,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'classification_report': classification_report(test_labels, test_preds, output_dict=True)
        }
        all_results.append(results)
        
        # Save model
        model_path = BASE_DIR / "models" / f"best_model_iteration_{iteration + 1}.pth"
        torch.save(best_model.state_dict(), model_path)
        logging.info(f"Saved best model for iteration {iteration + 1}")
        
        # Plot confusion matrix for test set
        plot_confusion_matrix(test_labels, test_preds, f"{iteration + 1}_test")
    
    # Save all results
    results_path = BASE_DIR / "results" / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    logging.info(f"\nTraining completed. Log file: {log_file}")

if __name__ == "__main__":
    main() 