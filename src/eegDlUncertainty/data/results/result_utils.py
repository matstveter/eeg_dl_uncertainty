import mlflow
import numpy as np
import torch

from eegDlUncertainty.data.results.dataset_shifts import activation_function
from eegDlUncertainty.data.results.uncertainty import calculate_performance_metrics, compute_classwise_uncertainty, \
    get_uncertainty_metrics
from eegDlUncertainty.data.utils import save_dict_to_pickle


def write_metrics_to_file(metrics_majority_vote, metrics_final_classes, file_path):
    """Write the calculated metrics to a file."""
    with open(file_path, 'w') as f:
        f.write("Majority Vote Metrics:\n")
        for metric, value in metrics_majority_vote.items():
            f.write(f"{metric}: {value}\n")

        f.write("\nFinal Classes Metrics:\n")
        for metric, value in metrics_final_classes.items():
            f.write(f"{metric}: {value}\n")


def ensemble_performance(model, test_loader, device, save_path):
    if not isinstance(model, list):
        logits, targets = model.get_ensemble_predictions(test_loader=test_loader, device=device, history=None,
                                                   num_forward=50)
    else:
        logits = []
        for m in model:
            ensemble_logits, targets = m.get_predictions(loader=test_loader, device=device)
            logits.append(ensemble_logits)
        logits = np.array(logits)  # shape: (num_models, num_samples, num_classes)

    mean_logits = torch.from_numpy(np.mean(logits, axis=0))
    all_predictions = activation_function(logits=torch.from_numpy(logits), ensemble=True)
    probs = activation_function(logits=mean_logits, ensemble=False)
    predictions = activation_function(logits=mean_logits, ensemble=False, ret_prob=False)
    target_classes = np.argmax(targets, axis=1)

    performance = calculate_performance_metrics(y_pred_prob=probs, y_pred_class=predictions,
                                                y_true_one_hot=targets, y_true_class=target_classes)

    uncertainty = get_uncertainty_metrics(probs=probs, targets=targets)

    class_uncertainty = compute_classwise_uncertainty(all_probs=all_predictions, mean_probs=probs,
                                                      one_hot_target=targets, targets=target_classes)

    results = {"performance": performance, "uncertainty": uncertainty, "class_uncertainty": class_uncertainty}

    save_dict_to_pickle(results, save_path, "ensemble_results")

    for k, v in results.items():
        if k in ("performance", "uncertainty"):
            for key, value in v.items():
                if key == "confusion_matrix":
                    continue
                mlflow.log_metric(f"ensemble_{key}", value)
