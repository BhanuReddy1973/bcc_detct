import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Union

def save_feature_visualization(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: Union[str, Path],
    title: str = 'Feature Visualization',
    feature_names: Optional[List[str]] = None
):
    """
    Save visualization of extracted features.
    
    Args:
        features (np.ndarray): Feature matrix
        labels (np.ndarray): Label array
        output_path (str or Path): Path to save the visualization
        title (str): Title for the plot
        feature_names (List[str], optional): Names of features for labeling
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot feature distributions
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(features.shape[1])]
    
    # Plot mean feature values for each class
    mean_features_bcc = features[labels == 1].mean(axis=0)
    mean_features_normal = features[labels == 0].mean(axis=0)
    
    x = np.arange(len(feature_names))
    width = 0.35
    
    ax1.bar(x - width/2, mean_features_bcc, width, label='BCC')
    ax1.bar(x + width/2, mean_features_normal, width, label='Normal')
    
    ax1.set_title('Mean Feature Values by Class')
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Mean Value')
    ax1.legend()
    
    # Plot feature correlation matrix
    correlation_matrix = np.corrcoef(features.T)
    sns.heatmap(correlation_matrix, ax=ax2, cmap='coolwarm', center=0)
    ax2.set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_tsne_visualization(
    tsne_features: np.ndarray,
    labels: np.ndarray,
    output_path: Union[str, Path],
    title: str = 't-SNE Visualization'
):
    """
    Save t-SNE visualization of features.
    
    Args:
        tsne_features (np.ndarray): t-SNE transformed features
        labels (np.ndarray): Label array
        output_path (str or Path): Path to save the visualization
        title (str): Title for the plot
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        tsne_features[:, 0],
        tsne_features[:, 1],
        c=labels,
        cmap='coolwarm',
        alpha=0.6
    )
    plt.colorbar(scatter)
    
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    plt.savefig(output_path)
    plt.close()

def save_feature_importance(
    importance_scores: np.ndarray,
    output_path: Union[str, Path],
    feature_names: Optional[List[str]] = None,
    title: str = 'Feature Importance'
):
    """
    Save visualization of feature importance scores.
    
    Args:
        importance_scores (np.ndarray): Array of feature importance scores
        output_path (str or Path): Path to save the visualization
        feature_names (List[str], optional): Names of features
        title (str): Title for the plot
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(len(importance_scores))]
    
    # Sort features by importance
    sorted_idx = np.argsort(importance_scores)
    pos = np.arange(len(sorted_idx))
    
    plt.figure(figsize=(12, 6))
    plt.barh(pos, importance_scores[sorted_idx])
    plt.yticks(pos, np.array(feature_names)[sorted_idx])
    plt.xlabel('Importance Score')
    plt.title(title)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close() 