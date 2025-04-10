# Data Preparation

## Overview
This component handles the initial loading and processing of TIF images, ensuring they are properly formatted and sized for subsequent processing steps.

## Key Features
- Large TIF image handling
- Memory-efficient loading
- Image resizing and normalization
- Dataset organization

## Implementation Details

### Image Loading
```python
def load_image_safely(image_path: str, target_size: int = None) -> np.ndarray:
    """
    Safely load large TIF images with optional downsampling.
    
    Args:
        image_path: Path to the TIF image
        target_size: Maximum dimension for resizing (optional)
    
    Returns:
        numpy.ndarray: Loaded image array
    """
    # Implementation details...
```

### Configuration Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_size` | 4096 | Maximum dimension for resizing |
| `Image.MAX_IMAGE_PIXELS` | None | Disable PIL's DecompressionBomb check |

## Usage Example
```python
from bcc_detection.preprocessing.data_loading import load_image_safely

# Load and resize an image
image = load_image_safely("path/to/image.tif", target_size=4096)
```

## Best Practices
1. Always use `load_image_safely` for loading TIF images
2. Set appropriate `target_size` based on available memory
3. Monitor memory usage during batch processing

## Common Issues and Solutions
1. **Memory Errors**
   - Reduce `target_size`
   - Process images in smaller batches
   - Use memory-efficient data types

2. **Loading Failures**
   - Check file permissions
   - Verify TIF file integrity
   - Ensure sufficient disk space

## Next Steps
After data preparation, proceed to:
- [Tissue Segmentation](../tissue_segmentation/README.md)
- [Patch Extraction](../patch_extraction/README.md) 