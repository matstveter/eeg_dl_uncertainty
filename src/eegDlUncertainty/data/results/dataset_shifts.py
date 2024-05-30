import numpy as np
import torch
from torch.utils.data import DataLoader

from eegDlUncertainty.data.data_generators.ShiftGenerators import EEGDatashift
from eegDlUncertainty.data.results.uncertainty import calculate_performance_metrics, get_uncertainty_metrics


def evaluate_shift(shift_type, model, test_subjects, dataset, use_age, device, batch_size, monte_carlo):
    shift_intensity = [0.0, 0.1, 0.2, 0.4, 0.8, 1.0]

    for s in shift_intensity:
        print(s)
        shift_dataset = EEGDatashift(subjects=test_subjects, dataset=dataset, use_age=use_age, device=device,
                                     shift_type=shift_type, shift_intensity=s)
        shift_loader = DataLoader(dataset=shift_dataset, batch_size=batch_size, shuffle=False)

        if monte_carlo:
            print("Testing with the combined EEG")
            logits, targets = model.get_mc_predictions(test_loader=shift_loader, device=device, history=None,
                                                       num_forward=50)
        else:
            raise NotImplementedError("Missing ensembles, that are not monte_carlo versions")

        mean_logits = torch.from_numpy(np.mean(logits, axis=0))
        probs = model.activation_function(logits=mean_logits).cpu().detach().numpy()
        predictions = model.activation_function(logits=mean_logits, ret_prob=False).cpu().detach().numpy()
        target_classes = np.argmax(targets, axis=1)

        print(calculate_performance_metrics(y_pred_prob=probs, y_pred_class=predictions,
                                            y_true_one_hot=targets, y_true_class=target_classes))
        print(get_uncertainty_metrics(probs=probs, targets=targets))


def evaluate_dataset_shifts(model, test_subjects, dataset, device, use_age, monte_carlo, batch_size):
    # evaluate_shift(shift_type="class_combination", model=model, test_subjects=test_subjects, dataset=dataset,
    #                device=device, use_age=use_age, monte_carlo=monte_carlo, batch_size=batch_size)
    evaluate_shift(shift_type="frequency_shift", model=model, test_subjects=test_subjects, dataset=dataset,
                   device=device, use_age=use_age, monte_carlo=monte_carlo, batch_size=batch_size)
