import os
import numpy as np
import torch
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from bcc_detection.feature_extraction import (
    extract_resnet_features,
    extract_traditional_features,
    extract_hybrid_features
)
from bcc_detection.data.data_loader import load_patches
from bcc_detection.utils.visualization import save_feature_visualization

def test_feature_extraction():
    """Test different feature extraction methods and generate visualizations."""
    print("Starting feature extraction test...")
    
    # Load sample patches
    print("\nLoading sample patches...")
    bcc_patches = load_patches('bcc', num_samples=10)
    normal_patches = load_patches('non-malignant', num_samples=10)
    
    # Test ResNet features
    print("\nTesting ResNet feature extraction...")
    bcc_resnet_features = extract_resnet_features(bcc_patches)
    normal_resnet_features = extract_resnet_features(normal_patches)
    print(f"ResNet features shape: {bcc_resnet_features.shape}")
    
    # Test traditional features
    print("\nTesting traditional feature extraction...")
    bcc_traditional_features = extract_traditional_features(bcc_patches)
    normal_traditional_features = extract_traditional_features(normal_patches)
    print(f"Traditional features shape: {bcc_traditional_features.shape}")
    
    # Test hybrid features
    print("\nTesting hybrid feature extraction...")
    bcc_hybrid_features = extract_hybrid_features(bcc_patches)
    normal_hybrid_features = extract_hybrid_features(normal_patches)
    print(f"Hybrid features shape: {bcc_hybrid_features.shape}")
    
    # Normalize features
    print("\nNormalizing features...")
    scaler = StandardScaler()
    bcc_features_norm = scaler.fit_transform(bcc_hybrid_features)
    normal_features_norm = scaler.transform(normal_hybrid_features)
    
    # Feature selection
    print("\nPerforming feature selection...")
    all_features = np.vstack([bcc_features_norm, normal_features_norm])
    labels = np.array([1] * len(bcc_features_norm) + [0] * len(normal_features_norm))
    selector = SelectKBest(score_func=f_classif, k=100)
    selected_features = selector.fit_transform(all_features, labels)
    
    # t-SNE visualization
    print("\nGenerating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(selected_features)
    
    # Plot t-SNE results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=features_tsne[:, 0],
        y=features_tsne[:, 1],
        hue=labels,
        palette=['red', 'blue'],
        alpha=0.6
    )
    plt.title('t-SNE Visualization of Extracted Features')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    # Save visualization
    output_dir = Path('bcc_detection/reports/stage2_feature_extraction/visualizations/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'tsne_visualization.png')
    plt.close()
    
    # Save feature importance plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(selector.scores_)), selector.scores_)
    plt.title('Feature Importance Scores')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance Score')
    plt.savefig(output_dir / 'feature_importance.png')
    plt.close()
    
    print("\nFeature extraction test completed!")
    print(f"Visualizations saved to: {output_dir}")

if __name__ == '__main__':
    test_feature_extraction() 