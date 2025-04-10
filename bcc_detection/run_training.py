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

# Get the base directory
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR.parent / "dataset"

# Disable PIL's decompression bomb check
Image.MAX_IMAGE_PIXELS = None

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

def create_data_loaders(data_dir, batch_size=32, num_samples=100, used_samples=None, rank=0, world_size=1):
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
        
        # Create full datasets
        full_dataset = BCCDataset(data_dir, transform, 'train', num_samples, used_samples)
        test_dataset = BCCDataset(data_dir, val_transform, 'test', num_samples//5, used_samples)
        
        # Create distributed samplers if using multiple GPUs
        if world_size > 1:
            train_sampler = DistributedSampler(full_dataset, num_replicas=world_size, rank=rank)
            test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
        else:
            train_sampler = None
            test_sampler = None
        
        # Create data loaders with optimized settings
        train_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=2,  # Reduced number of workers
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,  # Add prefetch factor
            timeout=60,  # Add timeout
            drop_last=True  # Drop last incomplete batch
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            timeout=60,
            drop_last=True
        )
        
        return train_loader, test_loader
    
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

def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to use')
        parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
        parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
        parser.add_argument('--num-workers', type=int, default=2, help='Number of data loader workers')
        args = parser.parse_args()
        
        # Setup distributed training
        rank, local_rank, world_size = setup_distributed()
        
        # Set up logging
        log_file = setup_logging(rank)
        logging.info(f"Starting training process on rank {rank} of {world_size}")
        
        # Set device
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        
        # Create data loaders with optimized settings
        train_loader, test_loader = create_data_loaders(
            data_dir=DATA_DIR,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            rank=rank,
            world_size=world_size
        )
        
        # Initialize model
        model = BCCModel().to(device)
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank])
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train model
        best_model_path = train_with_hyperparams(
            model, train_loader, test_loader, criterion, optimizer,
            device, num_epochs=args.epochs, iteration=1
        )
        
        # Test the model
        if rank == 0:  # Only test on rank 0
            test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
            logging.info(f"\nFinal Test Results:")
            logging.info(f"Test Loss: {test_loss:.4f}")
            logging.info(f"Test Accuracy: {test_acc:.2f}%")
            
            # Plot confusion matrix
            plot_confusion_matrix(test_labels, test_preds, iteration=1)
            
            # Print and save classification report
            report = classification_report(test_labels, test_preds)
            logging.info("\nClassification Report:")
            logging.info(report)
            
            report_dir = BASE_DIR / "reports"
            report_dir.mkdir(exist_ok=True)
            with open(report_dir / "classification_report.txt", "w") as f:
                f.write(report)
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise
    finally:
        if world_size > 1:
            dist.destroy_process_group()

if __name__ == "__main__":
    main() 