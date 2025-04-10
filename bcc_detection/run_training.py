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
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from bcc_detection.models import BCCModel
from bcc_detection.configs.config import Config
from bcc_detection.data_loading import DataLoader, PatchLoader
from bcc_detection.aggregation import SlidePredictor

# Get the base directory
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "dataset"

# Disable PIL's decompression bomb check
Image.MAX_IMAGE_PIXELS = None

class BCCDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, split='train', num_samples=None, used_samples=None, cache_dir=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Use the correct dataset path
        self.bcc_dir = self.data_dir / "package" / "bcc" / "data" / "images"
        self.non_malignant_dir = self.data_dir / "package" / "non-malignant" / "data" / "images"
        
        self.samples = self._load_samples(num_samples, used_samples)
        logging.info(f"Initialized {split} dataset with {len(self.samples)} samples")
    
    def _load_samples(self, num_samples=None, used_samples=None):
        samples = []
        cache_file = None
        
        if self.cache_dir:
            cache_file = self.cache_dir / f"{self.split}_samples.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        try:
            # Load BCC (positive) samples
            logging.info(f"Loading BCC samples from: {self.bcc_dir}")
            bcc_files = list(self.bcc_dir.glob("*.tif"))
            
            # Load non-malignant (negative) samples
            logging.info(f"Loading non-malignant samples from: {self.non_malignant_dir}")
            normal_files = list(self.non_malignant_dir.glob("*.tif"))
            
            # If no files found, create dummy data for testing
            if not bcc_files and not normal_files:
                logging.warning("No image files found. Creating dummy data for testing.")
                num_dummy_samples = num_samples or 10
                for i in range(num_dummy_samples):
                    # Create dummy BCC sample
                    samples.append({
                        'image_path': str(self.bcc_dir / f"dummy_bcc_{i}.tif"),
                        'label': 1,
                        'class': 'bcc',
                        'is_dummy': True
                    })
                    # Create dummy normal sample
                    samples.append({
                        'image_path': str(self.non_malignant_dir / f"dummy_normal_{i}.tif"),
                        'label': 0,
                        'class': 'normal',
                        'is_dummy': True
                    })
            else:
                # Balance classes
                min_samples = min(len(bcc_files), len(normal_files))
                bcc_files = random.sample(bcc_files, min_samples)
                normal_files = random.sample(normal_files, min_samples)
                
                # Create samples
                for img_path in bcc_files:
                    samples.append({
                        'image_path': str(img_path),
                        'label': 1,
                        'class': 'bcc',
                        'is_dummy': False
                    })
                
                for img_path in normal_files:
                    samples.append({
                        'image_path': str(img_path),
                        'label': 0,
                        'class': 'normal',
                        'is_dummy': False
                    })
            
            # Exclude previously used samples
            if used_samples:
                samples = [s for s in samples if s['image_path'] not in used_samples]
            
            # Randomly sample if num_samples is specified
            if num_samples is not None:
                samples = random.sample(samples, min(num_samples, len(samples)))
            
            # Split into train/val/test
            random.shuffle(samples)
            n = len(samples)
            if self.split == 'train':
                samples = samples[:int(0.7*n)]
            elif self.split == 'val':
                samples = samples[int(0.7*n):int(0.85*n)]
            else:  # test
                samples = samples[int(0.85*n):]
            
            # Cache samples if cache_dir is provided
            if cache_file:
                with open(cache_file, 'w') as f:
                    json.dump(samples, f)
            
            return samples
                
        except Exception as e:
            logging.error(f"Error loading samples: {str(e)}")
            raise

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Check cache first
        if self.cache_dir:
            cache_path = self.cache_dir / f"{Path(sample['image_path']).stem}.pt"
            if cache_path.exists():
                return torch.load(cache_path)
        
        # If it's a dummy sample, create a random image
        if sample.get('is_dummy', False):
            image = torch.rand(3, 224, 224)  # Create random RGB image
        else:
            # Load and preprocess image
            image = Image.open(sample['image_path'])
            width, height = image.size
            
            # Calculate new dimensions while maintaining aspect ratio
            target_size = 4096
            if width > height:
                new_width = target_size
                new_height = int(height * (target_size / width))
            else:
                new_height = target_size
                new_width = int(width * (target_size / height))
            
            # Resize and convert
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            image = image.convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
        
        # Cache processed image
        if self.cache_dir:
            torch.save((image, sample['label']), cache_path)
        
        return image, sample['label']

def setup_distributed():
    """Initialize distributed training"""
    if 'SLURM_PROCID' in os.environ:
        # SLURM environment
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        host = os.environ['SLURM_NODELIST']
        
        # Get IP address of the master node
        if rank == 0:
            host = subprocess.check_output(['scontrol', 'show', 'hostnames', host]).decode('utf-8').split('\n')[0]
            os.environ['MASTER_ADDR'] = host
        else:
            time.sleep(1)  # Wait for master to set MASTER_ADDR
            
        os.environ['MASTER_PORT'] = '29500'
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        
    else:
        # Local environment
        rank = 0
        local_rank = 0
        world_size = 1
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
    
    # Initialize process group
    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
    return rank, local_rank, world_size

def get_gpu_memory_info():
    """Get detailed GPU memory information"""
    if not torch.cuda.is_available():
        return None
    
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        gpu = torch.cuda.get_device_properties(i)
        total_memory = gpu.total_memory / 1024**3  # Convert to GB
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        cached = torch.cuda.memory_reserved(i) / 1024**3
        free = total_memory - allocated
        
        gpu_info.append({
            'id': i,
            'name': gpu.name,
            'total_memory_gb': total_memory,
            'allocated_gb': allocated,
            'cached_gb': cached,
            'free_gb': free
        })
    
    return gpu_info

def get_system_resources():
    """Get system resource information"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        'cpu_percent': cpu_percent,
        'memory_total_gb': memory.total / 1024**3,
        'memory_used_gb': memory.used / 1024**3,
        'memory_free_gb': memory.free / 1024**3,
        'disk_total_gb': disk.total / 1024**3,
        'disk_used_gb': disk.used / 1024**3,
        'disk_free_gb': disk.free / 1024**3
    }

def setup_logging(rank):
    """Set up logging with rank-specific file"""
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_rank{rank}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Log system information
    logging.info(f"Hostname: {socket.gethostname()}")
    logging.info(f"System Resources: {get_system_resources()}")
    if torch.cuda.is_available():
        logging.info(f"GPU Information: {get_gpu_memory_info()}")
    
    return log_file

def create_data_loaders(data_dir, batch_size=32, num_samples=100, used_samples=None, rank=0, world_size=1, cache_dir=None):
    try:
        # Define transforms with more augmentation
        transform = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
        
        # Create datasets with caching
        train_dataset = BCCDataset(data_dir, transform, 'train', num_samples, used_samples, cache_dir)
        val_dataset = BCCDataset(data_dir, val_transform, 'val', num_samples//5, used_samples, cache_dir)
        test_dataset = BCCDataset(data_dir, val_transform, 'test', num_samples//5, used_samples, cache_dir)
        
        # Create distributed samplers if using multiple GPUs
        if world_size > 1:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
            test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
        else:
            train_sampler = None
            val_sampler = None
            test_sampler = None
        
        # Create data loaders with optimized settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            timeout=60,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            timeout=60,
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            timeout=60,
            drop_last=True
        )
        
        return train_loader, val_loader, test_loader
    
    except Exception as e:
        logging.error(f"Error creating data loaders: {str(e)}")
        raise

def train_epoch(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    try:
        for inputs, labels in train_loader:
            try:
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
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.warning("GPU out of memory, skipping batch")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
                    
    except Exception as e:
        logging.error(f"Error in training epoch: {str(e)}")
        raise
    
    return running_loss / len(train_loader), 100. * correct / total, all_preds, all_labels

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    try:
        with torch.no_grad():
            for inputs, labels in val_loader:
                try:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logging.warning("GPU out of memory, skipping batch")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                        
    except Exception as e:
        logging.error(f"Error in validation: {str(e)}")
        raise
    
    return running_loss / len(val_loader), 100. * correct / total, all_preds, all_labels

def train_with_hyperparams(model, train_loader, test_loader, criterion, optimizer, device, num_epochs, iteration):
    """Train model with hyperparameters and return best model path"""
    best_acc = 0.0
    best_model_path = None
    patience = 5
    patience_counter = 0
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3, verbose=True
    )
    
    # Initialize gradient scaler for mixed precision
    scaler = amp.GradScaler()
    
    # Initialize metrics tracking
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        train_loss, train_acc, _, _ = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validation phase
        val_loss, val_acc, _, _ = validate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Track metrics
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        metrics['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            best_model_path = save_model(model, iteration, epoch, val_acc)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logging.info(f"Early stopping triggered at epoch {epoch}")
            break
        
        # Log progress
        logging.info(f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Save metrics
    save_metrics(metrics, iteration)
    
    return best_model_path

def plot_confusion_matrix(test_labels, test_preds, iteration):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save plot
    plots_dir = BASE_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f"confusion_matrix_iter{iteration}.png")
    plt.close()

def save_model(model, iteration, epoch, val_acc):
    """Save model checkpoint"""
    checkpoints_dir = BASE_DIR / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoints_dir / f"model_iter{iteration}_epoch{epoch}_acc{val_acc:.2f}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, checkpoint_path)
    
    return checkpoint_path

def save_metrics(metrics, iteration):
    """Save training metrics"""
    metrics_dir = BASE_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    metrics_path = metrics_dir / f"metrics_iter{iteration}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

def run_cross_validation(data_dir, num_folds=5, batch_size=32, num_samples=100, epochs=10, cache_dir=None):
    """Run k-fold cross validation"""
    all_metrics = []
    
    # Create k-fold splitter
    kfold = KFold(n_splits=num_folds, shuffle=True)
    
    # Load all samples
    dataset = BCCDataset(data_dir, split='train', num_samples=num_samples, cache_dir=cache_dir)
    samples = dataset.samples
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(samples)):
        logging.info(f"\nStarting fold {fold + 1}/{num_folds}")
        
        # Split samples
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        
        # Create datasets
        train_dataset = BCCDataset(data_dir, split='train', used_samples=val_samples, cache_dir=cache_dir)
        val_dataset = BCCDataset(data_dir, split='val', used_samples=train_samples, cache_dir=cache_dir)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize model
        model = BCCModel().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train model
        fold_metrics = train_with_hyperparams(
            model, train_loader, val_loader, criterion, optimizer,
            device, num_epochs=epochs, iteration=fold
        )
        
        all_metrics.append(fold_metrics)
    
    # Calculate average metrics
    avg_metrics = {
        'train_loss': np.mean([m['train_loss'][-1] for m in all_metrics]),
        'train_acc': np.mean([m['train_acc'][-1] for m in all_metrics]),
        'val_loss': np.mean([m['val_loss'][-1] for m in all_metrics]),
        'val_acc': np.mean([m['val_acc'][-1] for m in all_metrics])
    }
    
    logging.info("\nCross-validation Results:")
    logging.info(f"Average Train Loss: {avg_metrics['train_loss']:.4f}")
    logging.info(f"Average Train Accuracy: {avg_metrics['train_acc']:.2f}%")
    logging.info(f"Average Val Loss: {avg_metrics['val_loss']:.4f}")
    logging.info(f"Average Val Accuracy: {avg_metrics['val_acc']:.2f}%")
    
    return avg_metrics

def evaluate_model(model, test_loader, device):
    """Evaluate model performance with detailed metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'roc_auc': roc_auc_score(all_labels, [p[1] for p in all_probs]),
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, [p[1] for p in all_probs])
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {metrics["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    # Save ROC curve
    plots_dir = BASE_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / "roc_curve.png")
    plt.close()
    
    return metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Train BCC detection model')
    parser.add_argument('--num-samples', type=int, default=100,
                      help='Number of samples to use for training')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of epochs to train')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of workers for data loading')
    parser.add_argument('--num-folds', type=int, default=5,
                      help='Number of folds for cross-validation')
    return parser.parse_args()

def main():
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Setup distributed training
        rank, local_rank, world_size = setup_distributed()
        
        # Set up logging
        log_file = setup_logging(rank)
        logging.info(f"Starting training process on rank {rank} of {world_size}")
        
        # Set device
        device = torch.device("cpu")  # Force CPU for testing
        logging.info(f"Using device: {device}")
        
        # Run cross-validation
        avg_metrics = run_cross_validation(
            data_dir=DATA_DIR,
            num_folds=args.num_folds,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            epochs=args.epochs,
            cache_dir=None
        )
        
        # Create final data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=DATA_DIR,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            rank=rank,
            world_size=world_size,
            cache_dir=None
        )
        
        # Initialize final model
        model = BCCModel().to(device)
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank])
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train final model
        best_model_path = train_with_hyperparams(
            model, train_loader, val_loader, criterion, optimizer,
            device, num_epochs=args.epochs, iteration='final'
        )
        
        # Evaluate final model
        if rank == 0:
            metrics = evaluate_model(model, test_loader, device)
            logging.info("\nFinal Model Evaluation:")
            logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logging.info(f"Precision: {metrics['precision']:.4f}")
            logging.info(f"Recall: {metrics['recall']:.4f}")
            logging.info(f"F1 Score: {metrics['f1']:.4f}")
            logging.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
            
            # Plot confusion matrix
            test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
            plot_confusion_matrix(test_labels, test_preds, iteration='final')
        
        logging.info("Training and evaluation completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise
    finally:
        if world_size > 1:
            dist.destroy_process_group()

if __name__ == "__main__":
    main() 