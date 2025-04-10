import torch
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

class MixedPrecisionTrainer:
    """
    Trainer class with mixed precision support
    """
    
    def __init__(self, model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 device: str = 'cuda',
                 distributed: bool = False):
        """
        Initialize mixed precision trainer.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            device: Device to use
            distributed: Whether to use distributed training
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.distributed = distributed
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler()
        
        # Setup distributed training if needed
        if distributed:
            self.model = DistributedDataParallel(model)
            
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one training step with mixed precision.
        
        Args:
            batch: Dictionary containing input data and labels
            
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        inputs = batch['input'].to(self.device)
        labels = batch['label'].to(self.device)
        
        # Forward pass with mixed precision
        with autocast():
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Calculate metrics
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            accuracy = (preds == labels).float().mean()
            
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
        
        # Synchronize metrics in distributed mode
        if self.distributed:
            for k, v in metrics.items():
                dist.all_reduce(torch.tensor(v).to(self.device))
                metrics[k] = v / dist.get_world_size()
                
        return metrics
        
    @torch.no_grad()
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one validation step.
        
        Args:
            batch: Dictionary containing input data and labels
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        # Move data to device
        inputs = batch['input'].to(self.device)
        labels = batch['label'].to(self.device)
        
        # Forward pass
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        
        # Calculate metrics
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == labels).float().mean()
        
        metrics = {
            'val_loss': loss.item(),
            'val_accuracy': accuracy.item()
        }
        
        # Synchronize metrics in distributed mode
        if self.distributed:
            for k, v in metrics.items():
                dist.all_reduce(torch.tensor(v).to(self.device))
                metrics[k] = v / dist.get_world_size()
                
        return metrics
        
    def save_checkpoint(self, path: str, epoch: int,
                       metrics: Optional[Dict[str, float]] = None):
        """
        Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            metrics: Optional metrics to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict()
        }
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
            
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint
            
        Returns:
            Dictionary containing checkpoint data
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return checkpoint 