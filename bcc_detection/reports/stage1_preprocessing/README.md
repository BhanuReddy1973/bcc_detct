# Stage 1: Preprocessing Pipeline

## Overview
This stage focuses on preparing the TIF images for model training through a series of preprocessing steps. The preprocessing pipeline consists of three main components:

1. [Data Preparation](data_preparation/README.md)
2. [Tissue Segmentation](tissue_segmentation/README.md)
3. [Patch Extraction](patch_extraction/README.md)

## Directory Structure
```
stage1_preprocessing/
├── data_preparation/          # Data loading and initial processing
├── tissue_segmentation/       # Tissue region identification
├── patch_extraction/          # Patch generation and filtering
├── visualizations/            # Visual documentation of the process
├── quality_control/          # Quality verification guides
└── examples/                 # Practical examples
```

## Pipeline Flow
1. **Data Preparation**
   - Load TIF images
   - Resize large images
   - Handle memory constraints

2. **Tissue Segmentation**
   - Color deconvolution
   - Otsu's thresholding
   - Morphological operations

3. **Patch Extraction**
   - Generate patches
   - Filter by tissue content
   - Quality control

## Visual Documentation
See [Visual Guide](visualizations/visual_guide.md) for detailed examples of each step in the preprocessing pipeline.

## Implementation Details
- All preprocessing code is located in the `bcc_detection/preprocessing` directory
- Configuration parameters are documented in each component's README
- Example usage and best practices are provided in the [examples](examples/practical_examples.md)

## Next Steps
After preprocessing, the data is ready for:
- [Feature Extraction](../stage2_feature_extraction/README.md)
- [Model Training](../stage3_model_training/README.md)
- [Model Evaluation](../stage4_evaluation/README.md) 