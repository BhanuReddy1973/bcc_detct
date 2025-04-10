import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import matplotlib.pyplot as plt
import pytest
import yaml

from bcc_detection.preprocessing.tissue_segmentation import TissueSegmentation
from bcc_detection.preprocessing.tissue_packing import TissuePacking, PatchInfo
from bcc_detection.configs.config import (
    DATASET_ROOT, BCC_DIR, NON_MALIGNANT_DIR,
    IMAGES_DIR, TISSUE_MASKS_DIR, LABELS_DIR,
    DATA_CONFIG, MODEL_CONFIG
)
from ..utils.helpers import load_config, get_device
from ..run_training import BCCModel, BCCDataset, create_data_loaders
Image.MAX_IMAGE_PIXELS = None  # Disable PIL's DecompressionBomb check

def get_random_images(n_samples: int = 2) -> Tuple[List[str], List[int]]:
    """Get random samples from both BCC and non-malignant cases"""
    # Get BCC images
    bcc_images_dir = BCC_DIR / IMAGES_DIR
    bcc_images = [f for f in os.listdir(bcc_images_dir) if f.endswith('.tif')]
    
    # Get non-malignant images
    non_malignant_images_dir = NON_MALIGNANT_DIR / IMAGES_DIR
    non_malignant_images = [f for f in os.listdir(non_malignant_images_dir) if f.endswith('.tif')]
    
    # Select random samples
    bcc_samples = random.sample(bcc_images, min(n_samples, len(bcc_images)))
    non_malignant_samples = random.sample(non_malignant_images, min(n_samples, len(non_malignant_images)))
    
    # Create full paths and labels
    image_paths = []
    labels = []
    
    for img in bcc_samples:
        image_paths.append(str(bcc_images_dir / img))
        labels.append(1)  # BCC label
    
    for img in non_malignant_samples:
        image_paths.append(str(non_malignant_images_dir / img))
        labels.append(0)  # Non-malignant label
    
    return image_paths, labels

def load_image_safely(image_path: str, target_size: int = None) -> np.ndarray:
    """Load a large TIF image safely, with optional downsampling"""
    # Open the image
    with Image.open(image_path) as img:
        # Get original size
        width, height = img.size
        print(f"Original image size: {width}x{height}")
        
        if target_size and (width > target_size or height > target_size):
            # Calculate scaling factor
            scale = min(target_size / width, target_size / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            print(f"Resizing to: {new_width}x{new_height}")
            
            # Resize the image
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        return np.array(img)

def visualize_patches(image: np.ndarray, patches: List[PatchInfo], save_path: str = None):
    """Visualize the original image with extracted patches"""
    plt.figure(figsize=(15, 10))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot patches
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.title('Extracted Patches')
    plt.axis('off')
    
    # Draw rectangles around patches
    for patch in patches:
        x, y = patch.coordinates
        rect = plt.Rectangle((x, y), patch.image.shape[1], patch.image.shape[0],
                           linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        
        # Add tissue percentage as text
        plt.text(x, y, f'{patch.tissue_percentage:.1%}',
                color='r', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def test_pipeline_with_parameters():
    """Test the pipeline with different parameter combinations"""
    print("Starting pipeline test with parameter optimization...")
    
    # Get random samples
    n_samples = 2  # Number of samples from each class
    image_paths, labels = get_random_images(n_samples)
    
    print(f"\nSelected {len(image_paths)} images for testing:")
    for img_path, label in zip(image_paths, labels):
        print(f"- {Path(img_path).name} (Label: {'BCC' if label == 1 else 'Non-malignant'})")
    
    # Parameter combinations to try
    patch_sizes = [256, 512]  # Smaller patches might capture more tissue
    min_tissue_percentages = [0.5, 0.3]  # Lower thresholds to capture more patches
    overlaps = [0.5, 0.7]  # Higher overlap to ensure we don't miss tissue
    
    # Initialize components
    print("\nInitializing pipeline components...")
    tissue_segmenter = TissueSegmentation()
    
    # Create output directory for visualizations
    output_dir = Path("outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test each parameter combination
    for patch_size in patch_sizes:
        for min_tissue_percentage in min_tissue_percentages:
            for overlap in overlaps:
                print(f"\nTesting parameters:")
                print(f"- Patch size: {patch_size}x{patch_size}")
                print(f"- Min tissue percentage: {min_tissue_percentage:.0%}")
                print(f"- Overlap: {overlap:.0%}")
                
                tissue_packer = TissuePacking(
                    patch_size=patch_size,
                    min_tissue_percentage=min_tissue_percentage,
                    overlap=overlap
                )
                
                total_patches = 0
                for img_path in image_paths:
                    print(f"\nProcessing {Path(img_path).name}")
                    try:
                        # Load image with downsampling
                        image = load_image_safely(img_path, target_size=4096)
                        
                        # Segment tissue
                        mask, tissue_percentage = tissue_segmenter.segment_tissue(image)
                        print(f"Tissue percentage: {tissue_percentage:.2%}")
                        
                        # Extract patches
                        patches = tissue_packer.extract_patches(image, mask)
                        print(f"Extracted {len(patches)} valid patches")
                        total_patches += len(patches)
                        
                        if patches:
                            # Show information about first patch
                            first_patch = patches[0]
                            print(f"First patch shape: {first_patch.image.shape}")
                            print(f"First patch coordinates: {first_patch.coordinates}")
                            print(f"First patch tissue percentage: {first_patch.tissue_percentage:.2%}")
                            
                            # Visualize patches
                            img_name = Path(img_path).stem
                            save_path = output_dir / f"{img_name}_size{patch_size}_tissue{min_tissue_percentage:.2f}_overlap{overlap:.2f}.png"
                            visualize_patches(image, patches, save_path)
                            print(f"Visualization saved to: {save_path}")
                        
                    except Exception as e:
                        print(f"Error processing {Path(img_path).name}: {str(e)}")
                        continue
                
                print(f"\nTotal patches extracted with these parameters: {total_patches}")
                print("-" * 80)
    
    print("\nPipeline test completed!")

def test_config_loading():
    """Test if configuration file can be loaded."""
    config_path = Path(__file__).parent.parent / "configs" / "training_config.yaml"
    config = load_config(config_path)
    assert isinstance(config, dict)
    assert "training" in config
    assert "model" in config
    assert "data" in config

def test_model_creation():
    """Test if model can be created."""
    model = BCCModel()
    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, "features")
    assert hasattr(model, "classifier")

def test_dataset_creation():
    """Test if dataset can be created."""
    data_dir = Path(__file__).parent.parent / "dataset"
    dataset = BCCDataset(data_dir, split='train')
    assert len(dataset) > 0
    sample, label = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert isinstance(label, int)

def test_data_loader_creation():
    """Test if data loaders can be created."""
    data_dir = Path(__file__).parent.parent / "dataset"
    train_loader, val_loader, test_loader = create_data_loaders(data_dir, batch_size=4)
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0

def test_device_selection():
    """Test if device selection works."""
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ['cuda', 'cpu']

def test_model_forward_pass():
    """Test if model can perform forward pass."""
    model = BCCModel()
    device = get_device()
    model = model.to(device)
    x = torch.randn(1, 3, 224, 224).to(device)
    output = model(x)
    assert output.shape == (1, 2)

if __name__ == "__main__":
    pytest.main([__file__]) 