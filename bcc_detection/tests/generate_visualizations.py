import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBomb check
from pathlib import Path
import random
from bcc_detection.data.data_loader import TIFDataset
from bcc_detection.preprocessing.tissue_segmentation import TissueSegmentation
from bcc_detection.preprocessing.tissue_packing import TissuePacking

def create_visualizations():
    # Create output directories
    vis_dir = Path("bcc_detection/visualizations")
    patches_dir = vis_dir / "patches"
    training_dir = vis_dir / "training"
    results_dir = vis_dir / "results"
    
    for dir_path in [vis_dir, patches_dir, training_dir, results_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of image files
    dataset_path = Path("dataset")
    bcc_samples = list((dataset_path / "package" / "bcc" / "data" / "images").glob("*.tif"))
    normal_samples = list((dataset_path / "package" / "non-malignant" / "data" / "images").glob("*.tif"))
    
    if not bcc_samples or not normal_samples:
        print("No samples found in the dataset directory")
        print(f"BCC samples found: {len(bcc_samples)}")
        print(f"Normal samples found: {len(normal_samples)}")
        print(f"Looking in:")
        print(f"  BCC: {dataset_path / 'package' / 'bcc' / 'data' / 'images'}")
        print(f"  Normal: {dataset_path / 'package' / 'non-malignant' / 'data' / 'images'}")
        return
    
    # Select one sample from each class
    bcc_sample = str(random.choice(bcc_samples))
    normal_sample = str(random.choice(normal_samples))
    
    print(f"Selected samples:")
    print(f"BCC: {bcc_sample}")
    print(f"Normal: {normal_sample}")
    
    # Initialize components
    tissue_seg = TissueSegmentation()
    tissue_pack = TissuePacking()
    
    def load_and_resize_image(image_path, max_size=4096):
        """Load an image and resize if necessary to prevent memory issues"""
        img = Image.open(image_path)
        width, height = img.size
        
        # Calculate scaling factor if image is too large
        scale = min(max_size / width, max_size / height, 1.0)
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        return np.array(img)
    
    # Process BCC sample
    print(f"\nProcessing BCC sample: {bcc_sample}")
    bcc_image = load_and_resize_image(bcc_sample)
    print(f"BCC image shape: {bcc_image.shape}")
    bcc_mask, bcc_tissue_percentage = tissue_seg.segment_tissue(bcc_image)
    print(f"BCC tissue percentage: {bcc_tissue_percentage:.2%}")
    bcc_patches = tissue_pack.extract_patches(bcc_image, bcc_mask)
    print(f"Extracted {len(bcc_patches)} BCC patches")
    
    # Process normal sample
    print(f"\nProcessing normal sample: {normal_sample}")
    normal_image = load_and_resize_image(normal_sample)
    print(f"Normal image shape: {normal_image.shape}")
    normal_mask, normal_tissue_percentage = tissue_seg.segment_tissue(normal_image)
    print(f"Normal tissue percentage: {normal_tissue_percentage:.2%}")
    normal_patches = tissue_pack.extract_patches(normal_image, normal_mask)
    print(f"Extracted {len(normal_patches)} normal patches")
    
    # Save example images with masks
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(bcc_image)
    plt.title("BCC Sample")
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(bcc_mask, cmap='gray')
    plt.title("Tissue Mask")
    plt.axis('off')
    plt.savefig(patches_dir / "bcc_sample_with_mask.png")
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(normal_image)
    plt.title("Normal Sample")
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(normal_mask, cmap='gray')
    plt.title("Tissue Mask")
    plt.axis('off')
    plt.savefig(patches_dir / "normal_sample_with_mask.png")
    plt.close()
    
    # Save example patches
    if len(bcc_patches) > 0:
        plt.figure(figsize=(10, 10))
        for i in range(min(4, len(bcc_patches))):
            plt.subplot(2, 2, i+1)
            plt.imshow(bcc_patches[i].image)
            plt.title(f"BCC Patch {i+1}\nTissue: {bcc_patches[i].tissue_percentage:.1%}")
            plt.axis('off')
        plt.savefig(patches_dir / "bcc_patches.png")
        plt.close()
    
    if len(normal_patches) > 0:
        plt.figure(figsize=(10, 10))
        for i in range(min(4, len(normal_patches))):
            plt.subplot(2, 2, i+1)
            plt.imshow(normal_patches[i].image)
            plt.title(f"Normal Patch {i+1}\nTissue: {normal_patches[i].tissue_percentage:.1%}")
            plt.axis('off')
        plt.savefig(patches_dir / "normal_patches.png")
        plt.close()
    
    # Generate example training curves (dummy data for now)
    epochs = range(1, 101)
    train_loss = [1.0 * np.exp(-0.05 * x) for x in epochs]
    val_loss = [1.1 * np.exp(-0.04 * x) for x in epochs]
    train_acc = [0.5 + 0.5 * (1 - np.exp(-0.05 * x)) for x in epochs]
    val_acc = [0.5 + 0.4 * (1 - np.exp(-0.04 * x)) for x in epochs]
    
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(epochs, train_loss, label='Training')
    plt.plot(epochs, val_loss, label='Validation')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(epochs, train_acc, label='Training')
    plt.plot(epochs, val_acc, label='Validation')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(training_dir / "training_curves.png")
    plt.close()
    
    # Generate example confusion matrix (dummy data for now)
    confusion_matrix = np.array([[45, 5], [8, 42]])
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, cmap='Blues')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks([0, 1], ['Normal', 'BCC'])
    plt.yticks([0, 1], ['Normal', 'BCC'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(confusion_matrix[i, j]), ha='center', va='center')
    plt.savefig(results_dir / "confusion_matrix.png")
    plt.close()
    
    # Generate example ROC curve (dummy data for now)
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr)  # Example ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(results_dir / "roc_curve.png")
    plt.close()

if __name__ == "__main__":
    create_visualizations() 