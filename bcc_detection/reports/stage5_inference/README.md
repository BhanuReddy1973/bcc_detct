# Inference Pipeline

## Overview
This component provides tools for running inference on new images using trained BCC detection models.

## Key Features
- Batch processing of images
- Patch-based inference
- Confidence scoring
- Result visualization
- Export capabilities

## Implementation Details

### Inference Pipeline
```python
class BCCInferencePipeline:
    """
    Pipeline for BCC detection inference.
    
    Args:
        model_path: Path to trained model
        device: Inference device
        batch_size: Batch size for inference
        confidence_threshold: Minimum confidence for positive predictions
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 32,
        confidence_threshold: float = 0.5
    ):
        # Implementation details...
    
    def process_image(
        self,
        image_path: str,
        return_patches: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single image.
        
        Args:
            image_path: Path to input image
            return_patches: Whether to return patch-level results
        
        Returns:
            Dict containing prediction results
        """
        # Implementation details...
```

### Configuration Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Number of patches per batch |
| `confidence_threshold` | 0.5 | Minimum confidence for positive predictions |
| `patch_size` | 512 | Size of patches for inference |
| `overlap` | 0.1 | Overlap between patches |
| `min_tissue_percentage` | 0.5 | Minimum tissue percentage in patches |

## Usage Example
```python
from bcc_detection.inference import BCCInferencePipeline

# Initialize pipeline
pipeline = BCCInferencePipeline(
    model_path="path/to/model.pt",
    device="cuda",
    batch_size=32
)

# Process image
result = pipeline.process_image(
    image_path="path/to/image.tif",
    return_patches=True
)

# Access results
print(f"Overall prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
print(f"Number of patches: {len(result['patches'])}")
```

## Best Practices
1. Use appropriate batch size
2. Monitor memory usage
3. Validate input images
4. Handle edge cases
5. Log inference results

## Common Issues and Solutions
1. **Memory Issues**
   - Reduce batch size
   - Use smaller patch size
   - Enable gradient checkpointing
   - Use mixed precision

2. **Performance**
   - Enable CUDA optimization
   - Use data prefetching
   - Implement caching
   - Profile bottlenecks

3. **Quality**
   - Validate model outputs
   - Implement quality checks
   - Handle edge cases
   - Add confidence thresholds

## Next Steps
After setting up inference, proceed to:
- [Deployment Guide](../stage6_deployment/README.md)
- [API Documentation](../stage7_api/README.md)
- [Model Evaluation](../stage4_evaluation/README.md) 