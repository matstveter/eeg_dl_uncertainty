import os
from typing import List
import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from eegDlUncertainty.data.results.history import History


def write_metrics_to_file(metrics_majority_vote, metrics_final_classes, file_path):
    """Write the calculated metrics to a file."""
    with open(file_path, 'w') as f:
        f.write("Majority Vote Metrics:\n")
        for metric, value in metrics_majority_vote.items():
            f.write(f"{metric}: {value}\n")

        f.write("\nFinal Classes Metrics:\n")
        for metric, value in metrics_final_classes.items():
            f.write(f"{metric}: {value}\n")


def calculate_metrics(y_true, y_pred, average='macro'):
    """Calculate and return performance metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    return metrics

