import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging

class ResultVisualizer:
    def __init__(self, output_dir='visualizations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_confusion_matrix(self, true_labels, pred_labels, title='Confusion Matrix'):
        """Plot confusion matrix"""
        try:
            cm = confusion_matrix(true_labels, pred_labels)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(title)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save plot
            plt.savefig(self.output_dir / 'confusion_matrix.png')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting confusion matrix: {str(e)}")
            raise
    
    def plot_roc_curve(self, fpr, tpr, auc, title='ROC Curve'):
        """Plot ROC curve"""
        try:
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend()
            
            # Save plot
            plt.savefig(self.output_dir / 'roc_curve.png')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting ROC curve: {str(e)}")
            raise
    
    def plot_heatmap(self, predictions, coordinates, slide_shape, title='Prediction Heatmap'):
        """Plot prediction heatmap"""
        try:
            # Create empty heatmap
            heatmap = np.zeros(slide_shape)
            
            # Fill heatmap with predictions
            for (x, y), pred in zip(coordinates, predictions):
                heatmap[y, x] = pred
            
            # Plot heatmap
            plt.figure(figsize=(12, 10))
            plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            plt.colorbar(label='Prediction Confidence')
            plt.title(title)
            
            # Save plot
            plt.savefig(self.output_dir / 'prediction_heatmap.png')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting heatmap: {str(e)}")
            raise
    
    def plot_training_metrics(self, metrics, title='Training Metrics'):
        """Plot training metrics over epochs"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot loss
            plt.subplot(2, 1, 1)
            plt.plot(metrics['train_loss'], label='Training Loss')
            plt.plot(metrics['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss over Epochs')
            plt.legend()
            
            # Plot accuracy
            plt.subplot(2, 1, 2)
            plt.plot(metrics['train_acc'], label='Training Accuracy')
            plt.plot(metrics['val_acc'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Accuracy over Epochs')
            plt.legend()
            
            plt.tight_layout()
            
            # Save plot
            plt.savefig(self.output_dir / 'training_metrics.png')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting training metrics: {str(e)}")
            raise
    
    def plot_feature_importance(self, features, importance_scores, title='Feature Importance'):
        """Plot feature importance scores"""
        try:
            # Sort features by importance
            sorted_idx = np.argsort(importance_scores)
            sorted_features = [features[i] for i in sorted_idx]
            sorted_scores = importance_scores[sorted_idx]
            
            # Plot
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(sorted_features)), sorted_scores)
            plt.yticks(range(len(sorted_features)), sorted_features)
            plt.xlabel('Importance Score')
            plt.title(title)
            
            # Save plot
            plt.savefig(self.output_dir / 'feature_importance.png')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting feature importance: {str(e)}")
            raise 