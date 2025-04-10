# Tissue Segmentation

## Overview
This component identifies and segments tissue regions from the background in TIF images, enabling focused analysis on relevant areas.

## Key Features
- Adaptive thresholding
- Morphological operations
- Tissue percentage calculation
- Background removal

## Implementation Details

### Segmentation Process
```python
def segment_tissue(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Segment tissue regions from the background.
    
    Args:
        image: Input image array
    
    Returns:
        Tuple[np.ndarray, float]: Binary mask and tissue percentage
    """
    # Implementation details...
```

### Configuration Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_tissue_percentage` | 0.1 | Minimum tissue percentage threshold |
| `kernel_size` | 5 | Size of morphological operation kernel |
| `sigma` | 1.0 | Gaussian blur sigma value |

## Usage Example
```python
from bcc_detection.preprocessing.tissue_segmentation import segment_tissue

# Segment tissue from an image
mask, tissue_percentage = segment_tissue(image)
```

## Best Practices
1. Adjust `min_tissue_percentage` based on image characteristics
2. Fine-tune `kernel_size` for optimal morphological operations
3. Validate segmentation results visually

## Common Issues and Solutions
1. **Over-segmentation**
   - Increase `kernel_size`
   - Adjust threshold values
   - Apply additional morphological operations

2. **Under-segmentation**
   - Decrease `kernel_size`
   - Modify threshold values
   - Use adaptive thresholding

## Next Steps
After tissue segmentation, proceed to:
- [Patch Extraction](../patch_extraction/README.md)
- [Feature Extraction](../../stage2_feature_extraction/README.md) 