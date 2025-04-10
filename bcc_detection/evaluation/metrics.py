import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import roc_curve, auc, confusion_matrix
import torch
import torch.nn.functional as F

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate various classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate basic metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score
    }
    
    # Calculate ROC and AUC if probabilities are provided
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        metrics.update({
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr
        })
        
    return metrics

def calculate_uncertainty(logits: torch.Tensor, method: str = 'entropy') -> torch.Tensor:
    """
    Calculate prediction uncertainty.
    
    Args:
        logits: Model logits
        method: Uncertainty method ('entropy' or 'margin')
        
    Returns:
        Uncertainty scores
    """
    probabilities = F.softmax(logits, dim=1)
    
    if method == 'entropy':
        # Entropy of the probability distribution
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
        return entropy
        
    elif method == 'margin':
        # Difference between top two probabilities
        sorted_probs, _ = torch.sort(probabilities, dim=1, descending=True)
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        return 1 - margin
        
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")

def calculate_patch_metrics(patch_predictions: np.ndarray,
                          slide_label: int,
                          threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate metrics for patch-level predictions.
    
    Args:
        patch_predictions: Predictions for each patch
        slide_label: Ground truth label for the slide
        threshold: Decision threshold
        
    Returns:
        Dictionary of patch-level metrics
    """
    # Convert predictions to binary
    binary_preds = (patch_predictions > threshold).astype(int)
    
    # Calculate percentage of positive patches
    positive_ratio = np.mean(binary_preds)
    
    # Calculate agreement score (percentage of patches agreeing with slide label)
    agreement = np.mean(binary_preds == slide_label)
    
    # Calculate confidence (mean prediction probability for correct class)
    confidence = np.mean(np.where(binary_preds == slide_label,
                                patch_predictions,
                                1 - patch_predictions))
    
    return {
        'positive_ratio': positive_ratio,
        'patch_agreement': agreement,
        'patch_confidence': confidence
    }

def aggregate_slide_prediction(patch_predictions: np.ndarray,
                             method: str = 'majority',
                             threshold: float = 0.5) -> Tuple[int, float]:
    """
    Aggregate patch-level predictions to slide-level.
    
    Args:
        patch_predictions: Predictions for each patch
        method: Aggregation method ('majority' or 'mean')
        threshold: Decision threshold
        
    Returns:
        Tuple of (slide prediction, confidence score)
    """
    if method == 'majority':
        # Count patches above threshold
        binary_preds = (patch_predictions > threshold).astype(int)
        positive_ratio = np.mean(binary_preds)
        
        # Make slide prediction
        slide_pred = int(positive_ratio > 0.5)
        confidence = abs(positive_ratio - 0.5) * 2  # Scale to [0,1]
        
    elif method == 'mean':
        # Average patch probabilities
        mean_prob = np.mean(patch_predictions)
        slide_pred = int(mean_prob > threshold)
        confidence = abs(mean_prob - threshold) / max(threshold, 1-threshold)
        
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
        
    return slide_pred, confidence 