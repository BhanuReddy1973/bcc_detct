import os
import yaml
import torch
import logging
from pathlib import Path
import numpy as np
import random
from typing import Dict, Any, List, Tuple

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(log_dir: Path) -> Path:
    """Set up logging configuration."""
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

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_directories(base_dir: Path) -> None:
    """Create necessary directories."""
    directories = [
        "logs",
        "models",
        "results",
        "visualizations",
        "reports"
    ]
    for dir_name in directories:
        (base_dir / dir_name).mkdir(exist_ok=True)

def save_model(model: torch.nn.Module, path: Path, iteration: int) -> None:
    """Save model checkpoint."""
    torch.save(model.state_dict(), path / f"best_model_iteration_{iteration}.pth")

def load_model(model: torch.nn.Module, path: Path) -> torch.nn.Module:
    """Load model checkpoint."""
    model.load_state_dict(torch.load(path))
    return model

def get_device() -> torch.device:
    """Get the device to use for training."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_time(seconds: float) -> str:
    """Format time in seconds to a human-readable string."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s" 