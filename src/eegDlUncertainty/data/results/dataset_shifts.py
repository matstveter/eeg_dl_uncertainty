import numpy as np
import torch
from torch.utils.data import DataLoader

from eegDlUncertainty.data.data_generators.ShiftGenerators import EEGDatashiftGenerator
from eegDlUncertainty.data.results.uncertainty import calculate_performance_metrics, compute_classwise_uncertainty, \
    get_uncertainty_metrics


def evaluate_shift(shift_type, model, test_subjects, dataset, use_age, device, batch_size, monte_carlo):
    shift_intensity = [0.0, 0.1, 0.2, 0.4, 0.8, 1.0]

    shift_results = {}
    shift_predictions = {}

    for s in shift_intensity:
        shift_dataset = EEGDatashiftGenerator(subjects=test_subjects, dataset=dataset, use_age=use_age, device=device,
                                              shift_type=shift_type, shift_intensity=s)
        shift_loader = DataLoader(dataset=shift_dataset, batch_size=batch_size, shuffle=False)

        if monte_carlo:
            print("Testing with the combined EEG")
            logits, targets = model.get_mc_predictions(test_loader=shift_loader, device=device, history=None,
                                                       num_forward=50)
        else:
            raise NotImplementedError("Missing ensembles, that are not monte_carlo versions")
            #for m in model:
            #    m.eval()
            #    m.predict_proba()

        mean_logits = torch.from_numpy(np.mean(logits, axis=0))
        all_predictions = torch.softmax(torch.from_numpy(logits), dim=2).numpy()
        probs = model.activation_function(logits=mean_logits).cpu().detach().numpy()
        predictions = model.activation_function(logits=mean_logits, ret_prob=False).cpu().detach().numpy()
        target_classes = np.argmax(targets, axis=1)

        shift_predictions[s] = {"mean_logits": mean_logits,
                                "all_probs": all_predictions,
                                "probs": probs,
                                "predictions": predictions,
                                "target_classes": target_classes}
        performance = calculate_performance_metrics(y_pred_prob=probs, y_pred_class=predictions,
                                                    y_true_one_hot=targets, y_true_class=target_classes)

        uncertainty = get_uncertainty_metrics(probs=probs, targets=targets)

        class_uncertainty = compute_classwise_uncertainty(all_probs=all_predictions, mean_probs=probs,
                                                          one_hot_target=targets, targets=target_classes)

        shift_results[s] = {"performance": performance,
                            "uncertainty": uncertainty,
                            "class_uncertainty": class_uncertainty}

    return shift_results, shift_predictions


def evaluate_dataset_shifts(model, test_subjects, dataset, device, use_age, monte_carlo, batch_size):

    # res, pred = evaluate_shift(shift_type="class_combination_spatial", model=model, test_subjects=test_subjects,
    #                            dataset=dataset, device=device, use_age=use_age, monte_carlo=monte_carlo,
    #                            batch_size=batch_size)
    res, pred = evaluate_shift(shift_type="alpha_bandpass", model=model, test_subjects=test_subjects,
                               dataset=dataset, device=device, use_age=use_age, monte_carlo=monte_carlo,
                               batch_size=batch_size)
    # print(res)
    # res, pred = evaluate_shift(shift_type="class_combination_temporal", model=model, test_subjects=test_subjects,
    #                            dataset=dataset, device=device, use_age=use_age, monte_carlo=monte_carlo,
    #                            batch_size=batch_size)
    # print(res)
    # res, pred = evaluate_shift(shift_type="timereverse", model=model, test_subjects=test_subjects, dataset=dataset,
    #                            device=device, use_age=use_age, monte_carlo=monte_carlo, batch_size=batch_size)
    # print(res)
    # res, pred = evaluate_shift(shift_type="signflip", model=model, test_subjects=test_subjects, dataset=dataset,
    #                            device=device, use_age=use_age, monte_carlo=monte_carlo, batch_size=batch_size)
    # print(res)
    # res, pred = evaluate_shift(shift_type="gaussian", model=model, test_subjects=test_subjects, dataset=dataset,
    #                            device=device, use_age=use_age, monte_carlo=monte_carlo, batch_size=batch_size)
    # print(res)
    # res, pred = evaluate_shift(shift_type="gaussian", model=model, test_subjects=test_subjects, dataset=dataset,
    #                            device=device, use_age=use_age, monte_carlo=monte_carlo, batch_size=batch_size)
    # print(res)
    # res, pred = evaluate_shift(shift_type="interpolate", model=model, test_subjects=test_subjects, dataset=dataset,
    #                            device=device, use_age=use_age, monte_carlo=monte_carlo, batch_size=batch_size)
    # print(res)
    # todo Mix with EEG from different class

    # todo How should this be saved and plotted?
