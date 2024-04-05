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


def calculate_ensemble_performance(test_history_list: List[History], path: str):
    y_pred_prob_agg, y_pred_agg = [], []

    _, y_true = test_history_list[0].get_predictions()

    if test_history_list[0].num_classes > 1:
        y_true = np.argmax(np.array(y_true), axis=-1)

    for t in test_history_list:
        y_pred, _ = t.get_predictions()
        y_pred_prob_agg.append(y_pred)
        if t.num_classes == 1:
            y_pred_agg.append(y_pred.round())
        else:
            y_pred_agg.append(np.argmax(y_pred, axis=-1))

    # Convert lists to NumPy arrays
    y_pred_prob_agg_np = np.array(y_pred_prob_agg)
    y_pred_agg_np = np.array(y_pred_agg)
    y_true  # Assuming true labels are the same across all models

    combined_prob = np.mean(y_pred_prob_agg_np, axis=0)

    if test_history_list[0].num_classes == 1:
        final_classes = combined_prob.round()
    else:
        final_classes = np.argmax(combined_prob, axis=-1)

    majority_vote, _ = mode(y_pred_agg_np, axis=0, keepdims=False)

    # Now first round and argmax, then do majority voting
    metrics_majority_vote = calculate_metrics(y_true, majority_vote)
    metrics_final_classes = calculate_metrics(y_true, final_classes)

    # Specify your path location
    file_name = "ensemble_results.txt"
    file_path = os.path.join(path, file_name)

    # Write metrics to the file
    write_metrics_to_file(metrics_majority_vote, metrics_final_classes, file_path)
