import numpy as np
from sklearn.metrics import classification_report, precision_recall_curve, f1_score

# Function to find the best threshold
def find_best_threshold(y_true, y_prob, metric=f1_score):
    """
    Find the best threshold for the predicted probabilities to optimize the given metric.
    
    Args:
        y_true (array): True binary labels (0 and 1).
        y_prob (array): Predicted probabilities for the positive class.
        metric (function): A metric function to optimize. Default is F1 score.
        
    Returns:
        best_threshold (float): The threshold that optimizes the given metric.
        best_metric_value (float): The best value of the metric.
    """
    thresholds = np.arange(0.1, 1.0, 0.01)
    best_threshold = 0.5
    best_metric_value = -1
    
    for threshold in thresholds:
        y_pred_threshold = (y_prob >= threshold).astype(int)
        metric_value = metric(y_true, y_pred_threshold)
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_threshold = threshold
    
    return best_threshold, best_metric_value