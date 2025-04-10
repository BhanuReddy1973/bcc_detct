# Model Evaluation

## Overview
This component handles the evaluation of trained BCC detection models using various metrics and visualization techniques.

## Key Features
- Performance metrics calculation
- Confusion matrix analysis
- ROC curve and AUC calculation
- Feature importance analysis
- Model comparison tools

## Implementation Details

### Evaluation Metrics
```python
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Evaluation device
    
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    # Implementation details...
```

### Visualization Tools
```python
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str] = ["Non-malignant", "BCC"],
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        save_path: Path to save plot
    """
    # Implementation details...
```

### Configuration Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `metrics` | ["accuracy", "precision", "recall", "f1", "auc"] | Evaluation metrics |
| `threshold` | 0.5 | Classification threshold |
| `n_bootstraps` | 1000 | Number of bootstrap samples |
| `confidence_level` | 0.95 | Confidence level for intervals |

## Usage Example
```python
from bcc_detection.evaluation import evaluate_model, plot_confusion_matrix

# Evaluate model
metrics = evaluate_model(
    model=model,
    test_loader=test_loader
)

# Plot confusion matrix
plot_confusion_matrix(
    y_true=true_labels,
    y_pred=predicted_labels,
    save_path="confusion_matrix.png"
)
```

## Best Practices
1. Use multiple evaluation metrics
2. Perform cross-validation
3. Calculate confidence intervals
4. Visualize results
5. Compare with baselines

## Common Issues and Solutions
1. **Class Imbalance**
   - Use balanced accuracy
   - Adjust decision threshold
   - Apply class weights
   - Use stratified sampling

2. **Metric Selection**
   - Consider clinical context
   - Use multiple metrics
   - Calculate confidence intervals
   - Compare with baselines

3. **Visualization**
   - Use appropriate scales
   - Include confidence intervals
   - Label axes clearly
   - Save high-resolution plots

## Next Steps
After model evaluation, proceed to:
- [Inference Pipeline](../stage5_inference/README.md)
- [Deployment Guide](../stage6_deployment/README.md)
- [Model Training](../stage3_model_training/README.md) 