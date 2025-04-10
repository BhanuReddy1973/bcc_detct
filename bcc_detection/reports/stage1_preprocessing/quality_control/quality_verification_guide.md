# Patch Quality Verification Guide

## 1. Understanding the Visualizations

### 1.1 Visualization Components
Each visualization consists of two panels:
1. **Left Panel**: Original image with tissue mask overlay
   - Shows the entire image
   - Gray overlay indicates detected tissue regions
   - Tissue percentage displayed in title

2. **Right Panel**: Extracted patches
   - Shows the same image
   - Red rectangles indicate patch boundaries
   - Numbers show tissue percentage for each patch

### 1.2 Naming Convention
Visualization files follow this pattern:
```
{image_name}_size{patch_size}_tissue{tissue_percentage}_overlap{overlap}.png
```
Example: `image123_size256_tissue0.30_overlap0.70.png`

## 2. Quality Assessment Steps

### 2.1 Tissue Coverage Check
1. **Look at the left panel**:
   - Verify that the gray overlay (tissue mask) covers all relevant tissue areas
   - Check for any missed tissue regions
   - Note the overall tissue percentage

2. **Expected Results**:
   - Tissue mask should cover all biopsy tissue
   - Background should be mostly clear
   - Tissue percentage should be reasonable (typically 10-40%)

### 2.2 Patch Distribution Check
1. **Look at the right panel**:
   - Verify that patches cover all tissue regions
   - Check for gaps in coverage
   - Ensure patches are properly aligned

2. **Expected Results**:
   - Patches should cover all tissue areas
   - No large gaps between patches
   - Patches should align with tissue boundaries

### 2.3 Tissue Percentage Verification
1. **Check the numbers**:
   - Each patch shows its tissue percentage
   - Values should be above the minimum threshold (30%)
   - Look for consistent values across patches

2. **Expected Results**:
   - Most patches should have >30% tissue
   - Values should be consistent within similar regions
   - No patches with extremely low tissue content

## 3. Common Issues and Solutions

### 3.1 Insufficient Coverage
**Problem**: Gaps between patches or missed tissue regions
**Solution**:
- Increase overlap parameter (try 0.8)
- Decrease patch size (try 128×128)
- Adjust tissue segmentation parameters

### 3.2 Low Tissue Percentage
**Problem**: Many patches with low tissue content
**Solution**:
- Increase minimum tissue percentage threshold
- Improve tissue segmentation
- Adjust patch size

### 3.3 Too Many Patches
**Problem**: Excessive number of patches
**Solution**:
- Increase minimum tissue percentage
- Increase patch size
- Decrease overlap

## 4. Quality Metrics

### 4.1 Quantitative Metrics
1. **Patch Coverage**:
   - Percentage of tissue area covered by patches
   - Should be >90% for good coverage

2. **Tissue Content**:
   - Average tissue percentage across patches
   - Should be >40% for good quality

3. **Patch Distribution**:
   - Standard deviation of patch tissue percentages
   - Lower values indicate more consistent quality

### 4.2 Qualitative Metrics
1. **Visual Inspection**:
   - Check for missed tissue regions
   - Verify patch alignment
   - Assess overall coverage

2. **Edge Cases**:
   - Check patches at tissue boundaries
   - Verify handling of small tissue regions
   - Assess coverage of irregular shapes

## 5. Best Practices

### 5.1 Regular Verification
1. Check visualizations for every new batch of images
2. Monitor tissue percentages across patches
3. Verify coverage of different tissue types

### 5.2 Documentation
1. Keep records of quality issues
2. Document parameter adjustments
3. Track performance metrics over time

### 5.3 Continuous Improvement
1. Regularly review and update parameters
2. Test new parameter combinations
3. Gather feedback from pathologists

## 6. Troubleshooting Guide

### 6.1 Common Problems
1. **Missing Tissue**:
   - Check tissue segmentation parameters
   - Verify image preprocessing
   - Adjust contrast thresholds

2. **Patch Overlap Issues**:
   - Verify overlap calculation
   - Check patch size consistency
   - Ensure proper stride calculation

3. **Performance Problems**:
   - Optimize batch processing
   - Use appropriate hardware
   - Implement caching where possible

### 6.2 Parameter Tuning
1. **For Better Coverage**:
   - Increase overlap (0.7-0.8)
   - Decrease patch size
   - Lower tissue threshold

2. **For Higher Quality**:
   - Increase tissue threshold
   - Increase patch size
   - Decrease overlap

3. **For Better Performance**:
   - Increase batch size
   - Optimize preprocessing
   - Use parallel processing 