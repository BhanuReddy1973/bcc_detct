# Practical Examples: Using the Patch Extraction System

## 1. Basic Usage Examples

### Example 1: Loading a Single Image
```python
from bcc_detection.data.data_loader import TIFDataset
from pathlib import Path

# Single image example
image_path = Path("path/to/your/image.tif")
label = 1  # 1 for BCC, 0 for non-malignant

# Create dataset
dataset = TIFDataset(
    image_paths=[str(image_path)],
    labels=[label],
    patch_size=256,
    min_tissue_percentage=0.3,
    overlap=0.7
)

# Access patches
patches = dataset.patches
print(f"Extracted {len(patches)} patches")
for i, patch in enumerate(patches[:5]):  # Show first 5 patches
    print(f"Patch {i}:")
    print(f"- Shape: {patch.image.shape}")
    print(f"- Coordinates: {patch.coordinates}")
    print(f"- Tissue percentage: {patch.tissue_percentage:.1%}")
```

### Example 2: Batch Processing Multiple Images
```python
from bcc_detection.data.data_loader import TIFDataset, create_data_loaders
import glob

# Get all images from a directory
bcc_images = glob.glob("path/to/bcc/*.tif")
non_malignant_images = glob.glob("path/to/non_malignant/*.tif")

# Create labels
bcc_labels = [1] * len(bcc_images)
non_malignant_labels = [0] * len(non_malignant_images)

# Combine data
all_images = bcc_images + non_malignant_images
all_labels = bcc_labels + non_malignant_labels

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(
    train_images=all_images[:int(0.7*len(all_images))],
    train_labels=all_labels[:int(0.7*len(all_labels))],
    val_images=all_images[int(0.7*len(all_images)):int(0.85*len(all_images))],
    val_labels=all_labels[int(0.7*len(all_labels)):int(0.85*len(all_labels))],
    test_images=all_images[int(0.85*len(all_images)):],
    test_labels=all_labels[int(0.85*len(all_labels)):],
    batch_size=32,
    num_workers=4,
    patch_size=256,
    min_tissue_percentage=0.3,
    overlap=0.7
)
```

## 2. Quality Verification Examples

### Example 1: Checking Patch Quality
```python
import matplotlib.pyplot as plt
import numpy as np
from bcc_detection.preprocessing.tissue_segmentation import TissueSegmentation
from bcc_detection.preprocessing.tissue_packing import TissuePacking

def verify_patch_quality(image_path, save_path=None):
    """Verify patch quality for a single image"""
    # Load image
    image = np.array(Image.open(image_path))
    
    # Initialize components
    segmenter = TissueSegmentation()
    packer = TissuePacking(
        patch_size=256,
        min_tissue_percentage=0.3,
        overlap=0.7
    )
    
    # Process image
    mask, tissue_percentage = segmenter.segment_tissue(image)
    patches = packer.extract_patches(image, mask)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Original image with tissue mask
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.imshow(mask, alpha=0.3, cmap='gray')
    plt.title(f'Tissue Percentage: {tissue_percentage:.1%}')
    
    # Patches
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    for patch in patches:
        x, y = patch.coordinates
        rect = plt.Rectangle((x, y), 256, 256,
                           linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(x, y, f'{patch.tissue_percentage:.1%}',
                color='r', fontsize=8)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return patches

# Usage
patches = verify_patch_quality(
    "path/to/your/image.tif",
    save_path="outputs/quality_check.png"
)
```

### Example 2: Batch Quality Analysis
```python
import pandas as pd
from pathlib import Path

def analyze_batch_quality(image_dir):
    """Analyze patch quality for a batch of images"""
    results = []
    
    for img_path in Path(image_dir).glob("*.tif"):
        patches = verify_patch_quality(str(img_path))
        
        # Collect statistics
        stats = {
            'image': img_path.name,
            'total_patches': len(patches),
            'avg_tissue_percentage': np.mean([p.tissue_percentage for p in patches]),
            'min_tissue_percentage': min([p.tissue_percentage for p in patches]),
            'max_tissue_percentage': max([p.tissue_percentage for p in patches])
        }
        results.append(stats)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    print("\nBatch Quality Analysis:")
    print(df.describe())
    
    return df

# Usage
quality_df = analyze_batch_quality("path/to/your/images")
```

## 3. Integration Examples

### Example 1: Training Integration
```python
import torch
from torch.utils.data import DataLoader
from bcc_detection.models.model import BCCDetectionModel

def train_with_patches(train_loader, val_loader):
    """Example training loop with patch-based data"""
    model = BCCDetectionModel(
        backbone="efficientnet_b7",
        num_classes=2,
        dropout_rate=0.3
    ).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(50):
        model.train()
        for batch in train_loader:
            patches, labels = batch
            patches = patches.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(patches)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Example 2: Inference Integration
```python
def predict_image(image_path, model):
    """Predict BCC for a single image using patches"""
    # Load and process image
    dataset = TIFDataset(
        image_paths=[image_path],
        labels=[0],  # Dummy label
        patch_size=256,
        min_tissue_percentage=0.3,
        overlap=0.7
    )
    
    # Create data loader
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Predict
    model.eval()
    predictions = []
    with torch.no_grad():
        for patches, _ in loader:
            patches = patches.to(device)
            outputs = model(patches)
            predictions.extend(outputs.softmax(dim=1)[:, 1].cpu().numpy())
    
    # Aggregate predictions
    final_prediction = np.mean(predictions)
    return final_prediction > 0.5, final_prediction

# Usage
is_bcc, confidence = predict_image("path/to/test_image.tif", model)
print(f"BCC detected: {is_bcc} (confidence: {confidence:.2%})")
``` 