# Patch Extraction

## Overview
This component extracts meaningful patches from segmented tissue regions, preparing them for feature extraction and model training.

## Key Features
- Sliding window extraction
- Overlap control
- Tissue content validation
- Patch size optimization

## Implementation Details

### Patch Extraction Process
```python
def extract_patches(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 512,
    overlap: float = 0.5,
    min_tissue_percentage: float = 0.5
) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Extract patches from tissue regions.
    
    Args:
        image: Input image array
        mask: Tissue segmentation mask
        patch_size: Size of extracted patches
        overlap: Overlap between adjacent patches
        min_tissue_percentage: Minimum tissue content threshold
    
    Returns:
        List[Tuple[np.ndarray, Tuple[int, int]]]: List of patches with coordinates
    """
    # Implementation details...
```

### Configuration Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch_size` | 512 | Size of extracted patches |
| `overlap` | 0.5 | Overlap between adjacent patches |
| `min_tissue_percentage` | 0.5 | Minimum tissue content threshold |

## Usage Example
```python
from bcc_detection.preprocessing.patch_extraction import extract_patches

# Extract patches from an image
patches = extract_patches(
    image=image,
    mask=mask,
    patch_size=512,
    overlap=0.5,
    min_tissue_percentage=0.5
)
```

## Best Practices
1. Choose appropriate `patch_size` based on image resolution
2. Balance `overlap` between coverage and redundancy
3. Set `min_tissue_percentage` based on tissue distribution
4. Validate patch quality and distribution

## Common Issues and Solutions
1. **Too Few Patches**
   - Decrease `min_tissue_percentage`
   - Reduce `patch_size`
   - Increase `overlap`

2. **Too Many Patches**
   - Increase `min_tissue_percentage`
   - Increase `patch_size`
   - Decrease `overlap`

3. **Poor Patch Quality**
   - Adjust tissue segmentation parameters
   - Modify patch extraction criteria
   - Implement additional quality checks

## Next Steps
After patch extraction, proceed to:
- [Feature Extraction](../../stage2_feature_extraction/README.md)
- [Model Training](../../stage3_model_training/README.md) 