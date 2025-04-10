import os
import sys
import torch
import yaml
from pathlib import Path
from typing import Dict, Any

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from preprocessing.tissue_segmentation import get_tissue_mask, filter_tissue_patches
from preprocessing.color_normalization import StainNormalizer
from feature_extraction.deep_features import DeepFeatureExtractor
from feature_extraction.fuzzy_clustering import FuzzyCMeans
from models.model import create_model, create_model_with_backbone, BCCDetectionModel
from optimization.mixed_precision_training import MixedPrecisionTrainer
from evaluation.metrics import calculate_metrics, calculate_uncertainty, calculate_patch_metrics, aggregate_slide_prediction
from utils.data_loader import create_data_loaders
from scripts.train import train_model

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_environment(config: Dict[str, Any]) -> None:
    """Setup environment and create necessary directories"""
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create required directories
    for dir_path in [
        config['data']['raw_dir'],
        config['data']['processed_dir'],
        config['data']['annotations_dir'],
        config['data']['splits_dir'],
        config['outputs']['model_dir'],
        config['outputs']['log_dir']
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
    return device

def preprocess_slide(slide_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess a whole slide image"""
    # Get tissue mask
    mask, threshold = get_tissue_mask(
        slide_path,
        level=config['preprocessing']['level'],
        otsu_threshold=config['preprocessing'].get('otsu_threshold')
    )
    
    # Filter tissue patches
    valid_patches = filter_tissue_patches(
        mask,
        patch_size=config['preprocessing']['patch_size'],
        min_tissue_percentage=config['preprocessing']['min_tissue_percentage']
    )
    
    # Initialize stain normalizer
    normalizer = StainNormalizer()
    
    return {
        'mask': mask,
        'threshold': threshold,
        'valid_patches': valid_patches,
        'normalizer': normalizer
    }

def extract_features(patches: torch.Tensor, config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract features from patches"""
    # Initialize feature extractor
    feature_extractor = DeepFeatureExtractor(
        device=config['device'],
        feature_layers=config['feature_extraction']['layers']
    )
    
    # Extract deep features
    deep_features = feature_extractor.extract_batch_features(
        patches,
        batch_size=config['training']['batch_size']
    )
    
    # Apply fuzzy clustering
    fuzzy_clusterer = FuzzyCMeans(
        n_clusters=config['feature_extraction']['n_clusters'],
        m=config['feature_extraction']['fuzziness']
    )
    
    # Get cluster memberships
    membership, centroids = fuzzy_clusterer.fit(deep_features['features.7'])
    
    return {
        'deep_features': deep_features,
        'membership': membership,
        'centroids': centroids
    }

def train_model(train_data: Dict[str, torch.Tensor], config: Dict[str, Any]) -> Dict[str, Any]:
    """Train the BCC detection model"""
    # Create model
    model = create_model_with_backbone(config['model'])
    model = model.to(config['device'])
    
    # Initialize optimizer and criterion
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    criterion = torch.nn.CrossEntropyLoss()
    
    # Initialize trainer
    trainer = MixedPrecisionTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=config['device'],
        distributed=config['training'].get('distributed', False)
    )
    
    # Training loop
    best_metrics = {'val_loss': float('inf')}
    for epoch in range(config['training']['epochs']):
        # Training step
        train_metrics = trainer.train_step(train_data)
        
        # Validation step
        val_metrics = trainer.validate_step(train_data)
        
        # Save best model
        if val_metrics['val_loss'] < best_metrics['val_loss']:
            best_metrics = val_metrics
            trainer.save_checkpoint(
                os.path.join(config['outputs']['model_dir'], 'best_model.pth'),
                epoch,
                metrics=val_metrics
            )
            
        # Early stopping
        if epoch - best_metrics.get('best_epoch', 0) > config['training']['early_stopping_patience']:
            print(f"Early stopping at epoch {epoch}")
            break
            
    return {
        'model': model,
        'metrics': best_metrics
    }

def evaluate_model(model: torch.nn.Module, test_data: Dict[str, torch.Tensor],
                  config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the trained model"""
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        outputs = model(test_data['input'].to(config['device']))
        predictions = torch.argmax(outputs, dim=1)
        probabilities = torch.softmax(outputs, dim=1)
        
    # Calculate metrics
    metrics = calculate_metrics(
        test_data['label'].cpu().numpy(),
        predictions.cpu().numpy(),
        probabilities[:, 1].cpu().numpy()
    )
    
    # Calculate uncertainty
    uncertainty = calculate_uncertainty(outputs)
    
    # Calculate patch-level metrics
    patch_metrics = calculate_patch_metrics(
        probabilities[:, 1].cpu().numpy(),
        test_data['label'].cpu().numpy()
    )
    
    return {
        'metrics': metrics,
        'uncertainty': uncertainty,
        'patch_metrics': patch_metrics
    }

def main():
    """Main function to run the training process."""
    try:
        # Set up environment
        device = setup_environment(load_config('configs/config.yaml'))
        
        # Load configuration
        config = load_config('configs/config.yaml')
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=config['data']['raw_dir'],
            batch_size=config['training']['batch_size']
        )
        
        # Train model
        model_results = train_model(train_loader, config)
        
        # Evaluate model
        eval_results = evaluate_model(model_results['model'], test_loader, config)
        
        print("\nTraining completed!")
        print(f"Best validation metrics: {model_results['metrics']}")
        print(f"Test metrics: {eval_results['metrics']}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 