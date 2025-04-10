import os
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import project modules
from preprocessing.tissue_segmentation import get_tissue_mask, filter_tissue_patches
from preprocessing.color_normalization import StainNormalizer
from feature_extraction.deep_features import DeepFeatureExtractor
from feature_extraction.fuzzy_clustering import FuzzyCMeans
from models.model import BCCDetectionModel
from optimization.mixed_precision_training import MixedPrecisionTrainer
from evaluation.metrics import calculate_metrics, calculate_uncertainty

class SamplePipeline:
    def __init__(self):
        # Setup paths
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.output_dir = self.base_dir / "outputs"
        self.docs_dir = self.base_dir / "docs"
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.docs_dir.mkdir(exist_ok=True)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Configuration
        self.config = {
            'preprocessing': {
                'level': 0,
                'patch_size': 224,
                'min_tissue_percentage': 70
            },
            'feature_extraction': {
                'layers': ['features.7'],
                'n_clusters': 5,
                'fuzziness': 2.0
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'epochs': 10,
                'early_stopping_patience': 3
            }
        }
    
    def stage1_preprocessing(self, slide_path):
        """Stage 1: Preprocessing and tissue segmentation"""
        print("\n=== Stage 1: Preprocessing ===")
        
        # Get tissue mask
        mask, threshold = get_tissue_mask(
            slide_path,
            level=self.config['preprocessing']['level']
        )
        
        # Filter tissue patches
        valid_patches = filter_tissue_patches(
            mask,
            patch_size=self.config['preprocessing']['patch_size'],
            min_tissue_percentage=self.config['preprocessing']['min_tissue_percentage']
        )
        
        # Save results
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(mask, cmap='gray')
        plt.title('Tissue Mask')
        plt.subplot(122)
        plt.imshow(valid_patches, cmap='gray')
        plt.title('Valid Patches')
        plt.savefig(self.output_dir / 'stage1_results.png')
        plt.close()
        
        return mask, valid_patches
    
    def stage2_feature_extraction(self, patches):
        """Stage 2: Feature extraction"""
        print("\n=== Stage 2: Feature Extraction ===")
        
        # Initialize feature extractor
        feature_extractor = DeepFeatureExtractor(
            device=self.device,
            feature_layers=self.config['feature_extraction']['layers']
        )
        
        # Extract deep features
        deep_features = feature_extractor.extract_batch_features(
            patches,
            batch_size=self.config['training']['batch_size']
        )
        
        # Apply fuzzy clustering
        fuzzy_clusterer = FuzzyCMeans(
            n_clusters=self.config['feature_extraction']['n_clusters'],
            m=self.config['feature_extraction']['fuzziness']
        )
        
        membership, centroids = fuzzy_clusterer.fit(deep_features['features.7'])
        
        # Save results
        np.save(self.output_dir / 'deep_features.npy', deep_features)
        np.save(self.output_dir / 'membership.npy', membership)
        
        return deep_features, membership
    
    def stage3_model_training(self, train_data):
        """Stage 3: Model training"""
        print("\n=== Stage 3: Model Training ===")
        
        # Create model
        model = BCCDetectionModel(
            backbone='efficientnet-b7',
            num_classes=2,
            dropout_rate=0.5
        ).to(self.device)
        
        # Initialize trainer
        trainer = MixedPrecisionTrainer(
            model=model,
            optimizer=torch.optim.Adam(
                model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            ),
            criterion=torch.nn.CrossEntropyLoss(),
            device=self.device
        )
        
        # Training loop
        best_metrics = {'val_loss': float('inf')}
        for epoch in range(self.config['training']['epochs']):
            train_metrics = trainer.train_step(train_data)
            val_metrics = trainer.validate_step(train_data)
            
            print(f"Epoch {epoch+1}/{self.config['training']['epochs']}")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            
            if val_metrics['val_loss'] < best_metrics['val_loss']:
                best_metrics = val_metrics
                trainer.save_checkpoint(
                    self.output_dir / 'best_model.pth',
                    epoch,
                    metrics=val_metrics
                )
        
        return model, best_metrics
    
    def stage4_evaluation(self, model, test_data):
        """Stage 4: Model evaluation"""
        print("\n=== Stage 4: Evaluation ===")
        
        model.eval()
        with torch.no_grad():
            outputs = model(test_data['input'].to(self.device))
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
        
        # Save results
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        sns.heatmap(confusion_matrix(
            test_data['label'].cpu().numpy(),
            predictions.cpu().numpy()
        ), annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.subplot(122)
        plt.plot(uncertainty)
        plt.title('Prediction Uncertainty')
        plt.savefig(self.output_dir / 'stage4_results.png')
        plt.close()
        
        return metrics, uncertainty
    
    def stage5_documentation(self, results):
        """Stage 5: Documentation and reporting"""
        print("\n=== Stage 5: Documentation ===")
        
        # Create documentation
        doc_path = self.docs_dir / 'pipeline_results.md'
        with open(doc_path, 'w') as f:
            f.write("# BCC Detection Pipeline Results\n\n")
            f.write("## Stage 1: Preprocessing\n")
            f.write("- Tissue mask generated\n")
            f.write("- Valid patches extracted\n")
            f.write(f"- Minimum tissue percentage: {self.config['preprocessing']['min_tissue_percentage']}%\n\n")
            
            f.write("## Stage 2: Feature Extraction\n")
            f.write("- Deep features extracted using EfficientNet-B7\n")
            f.write(f"- Number of clusters: {self.config['feature_extraction']['n_clusters']}\n\n")
            
            f.write("## Stage 3: Model Training\n")
            f.write(f"- Model: BCCDetectionModel with EfficientNet-B7 backbone\n")
            f.write(f"- Epochs: {self.config['training']['epochs']}\n")
            f.write(f"- Best validation loss: {results['best_metrics']['val_loss']:.4f}\n\n")
            
            f.write("## Stage 4: Evaluation\n")
            f.write(f"- Accuracy: {results['metrics']['accuracy']:.4f}\n")
            f.write(f"- AUC-ROC: {results['metrics']['auc']:.4f}\n")
            f.write(f"- Average uncertainty: {np.mean(results['uncertainty']):.4f}\n")
        
        print(f"Documentation saved to {doc_path}")

def main():
    # Initialize pipeline
    pipeline = SamplePipeline()
    
    # Load sample data (replace with actual data loading)
    sample_slide = pipeline.data_dir / "sample_slide.tif"
    
    # Run all stages
    mask, patches = pipeline.stage1_preprocessing(sample_slide)
    features, membership = pipeline.stage2_feature_extraction(patches)
    model, best_metrics = pipeline.stage3_model_training({
        'input': torch.randn(100, 3, 224, 224),  # Replace with actual data
        'label': torch.randint(0, 2, (100,))     # Replace with actual labels
    })
    metrics, uncertainty = pipeline.stage4_evaluation(model, {
        'input': torch.randn(20, 3, 224, 224),   # Replace with actual test data
        'label': torch.randint(0, 2, (20,))      # Replace with actual test labels
    })
    
    # Document results
    pipeline.stage5_documentation({
        'best_metrics': best_metrics,
        'metrics': metrics,
        'uncertainty': uncertainty
    })

if __name__ == "__main__":
    main() 