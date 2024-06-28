import numpy as np
import torch
from torch.utils.data import DataLoader

from eegDlUncertainty.data.data_generators.ShiftGenerators import EEGDatashiftGenerator
from eegDlUncertainty.data.results.plotter import single_datashift_plotter
from eegDlUncertainty.data.results.uncertainty import calculate_performance_metrics, compute_classwise_uncertainty, \
    get_uncertainty_metrics
from eegDlUncertainty.data.utils import save_dict_to_pickle


def activation_function(logits, ensemble, ret_prob=True):
    """
    Apply softmax activation function to the logits and optionally return the class with the highest probability.

    This function applies the softmax activation function to the logits to convert them into probabilities.
    If `ensemble` is True, it assumes that the logits are from an ensemble of models and applies softmax along the second dimension.
    If `ensemble` is False, it assumes that the logits are from a single model and applies softmax along the first dimension.

    If `ret_prob` is False, the function also returns the class with the highest probability.

    Parameters
    ----------
    logits : torch.Tensor
        The logits to which the softmax activation function will be applied.
    ensemble : bool
        Whether the logits are from an ensemble of models.
    ret_prob : bool, optional
        Whether to return the probabilities or the class with the highest probability. Default is True.

    Returns
    -------
    numpy.ndarray
        The probabilities after applying the softmax activation function, or the class with the highest probability.
    """
    if ensemble:
        outp = torch.softmax(logits, dim=2)
        if not ret_prob:
            _, outp = torch.max(outp, dim=2)
    else:
        outp = torch.softmax(logits, dim=1)
        if not ret_prob:
            _, outp = torch.max(outp, dim=1)
    return outp.numpy()


def evaluate_shift(shift_type, model, test_subjects, dataset, use_age, device, batch_size, baseline, save_path, 
                   **kwargs):
    if baseline:
        shift_intensity = [0.0]
    else:
        shift_intensity = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

    shift_results = {}
    shift_predictions = {}

    for s in shift_intensity:
        shift_dataset = EEGDatashiftGenerator(subjects=test_subjects, dataset=dataset, use_age=use_age, device=device,
                                              shift_type=shift_type, shift_intensity=s, **kwargs)
        shift_loader = DataLoader(dataset=shift_dataset, batch_size=batch_size, shuffle=False)

        if not isinstance(model, list):
            print("Testing with the combined EEG")
            logits, targets = model.get_mc_predictions(test_loader=shift_loader, device=device, history=None,
                                                       num_forward=50)
        else:
            print("Testing with the ensemble")
            logits = []
            for m in model:
                ensemble_logits, targets = m.get_predictions(loader=shift_loader, device=device)
                logits.append(ensemble_logits)
            logits = np.array(logits)  # shape: (num_models, num_samples, num_classes)

        mean_logits = torch.from_numpy(np.mean(logits, axis=0))
        all_predictions = activation_function(logits=torch.from_numpy(logits), ensemble=True)
        probs = activation_function(logits=mean_logits, ensemble=False)
        predictions = activation_function(logits=mean_logits, ensemble=False, ret_prob=False)
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
                            f"uncertainty": uncertainty,
                            "class_uncertainty": class_uncertainty}

    single_datashift_plotter(shift_result=shift_results, shift_type=shift_type, save_path=save_path)
    return shift_results, shift_predictions


def evaluate_dataset_shifts(model, test_subjects, dataset, device, use_age, batch_size, save_path):

    final_results = {}

    # res_baseline, pred_baseline = evaluate_shift(shift_type=None, model=model, test_subjects=test_subjects,
    #                                              dataset=dataset, device=device, use_age=use_age, batch_size=batch_size,
    #                                              baseline=True, save_path=save_path)
    # final_results["baseline"] = {"results": res_baseline, "predictions": pred_baseline}
    #
    res_class_combi, pred_class_combi = evaluate_shift(shift_type="class_combination_temporal", model=model,
                                                       test_subjects=test_subjects, dataset=dataset, device=device,
                                                       use_age=use_age, batch_size=batch_size, baseline=False,
                                                       save_path=save_path)
    final_results["class_combination"] = {"results": res_class_combi, "predictions": pred_class_combi}
    #
    # bandpass = ['delta', 'theta', 'alpha', 'beta', 'hbeta', 'lbeta', 'gamma']
    # # Testing the effect of different bandpass filters
    # for band in bandpass:
    #     res_band, pred_band = evaluate_shift(shift_type=f"{band}_bandpass", model=model, test_subjects=test_subjects,
    #                                          dataset=dataset, device=device, use_age=use_age, batch_size=batch_size,
    #                                          baseline=False, save_path=save_path)
    #     final_results[band] = {"results": res_band, "predictions": pred_band}
    #
    # gaussian_std = [0.1, 0.2, 0.4, 0.8, 1.0]
    # # Gaussian noise
    # for g in gaussian_std:
    #     res_gaussian, pred_gaussian = evaluate_shift(shift_type="gaussian_channel", model=model,
    #                                                  test_subjects=test_subjects, dataset=dataset,
    #                                                  device=device, use_age=use_age, batch_size=batch_size,
    #                                                  baseline=False, save_path=save_path gaussian_std=g)
    #     final_results[f"gaussian_channel_{g}"] = {"results": res_gaussian, "predictions": pred_gaussian}
    #
    # res_gaussian, pred_gaussian = evaluate_shift(shift_type="gaussian", model=model, test_subjects=test_subjects,
    #                                              dataset=dataset, device=device, use_age=use_age, batch_size=batch_size,
    #                                              baseline=False, save_path=save_path)
    # final_results["gaussian_all"] = {"results": res_gaussian, "predictions": pred_gaussian}

    # phase_shift = [np.pi/8, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
    # # Phase shift
    # for p in phase_shift:
    #     res_phase_shift, pred_phase_shift = evaluate_shift(shift_type="phase_shift_channel", model=model,
    #                                                        test_subjects=test_subjects, dataset=dataset,
    #                                                        device=device, use_age=use_age, batch_size=batch_size,
    #                                                        baseline=False, save_path=save_path, phase_shift=p)
    #     final_results[f"phase_shift_channel_{p}"] = {"results": res_phase_shift, "predictions": pred_phase_shift}
    #
    # scalar_modulation = [0.1, 0.5, 0.75, 1.5, 2.0]
    # # Scalar modulation
    # for s in scalar_modulation:
    #     res_scalar, pred_scalar = evaluate_shift(shift_type="scalar_modulation", model=model, test_subjects=test_subjects,
    #                                              dataset=dataset, device=device, use_age=use_age, batch_size=batch_size,
    #                                              baseline=False, save_path=save_path,
    #                                              scalar_multi=s)
    #     final_results[f"scalar_modulation_{s}"] = {"results": res_scalar, "predictions": pred_scalar}
    #
    # for s in scalar_modulation:
    #     res_scalar_c, pred_scalar_c = evaluate_shift(shift_type="scalar_modulation_channel", model=model,
    #                                                  test_subjects=test_subjects, dataset=dataset, device=device,
    #                                                  use_age=use_age, batch_size=batch_size, baseline=False,
    #                                                  save_path=save_path,
    #                                                  scalar_multi=s)
    #     final_results[f"scalar_modulation_channel_{s}"] = {"results": res_scalar_c, "predictions": pred_scalar_c}
    #
    # # interpolate
    res_interpolate, pred_interpolate = evaluate_shift(shift_type="interpolate", model=model,
                                                       test_subjects=test_subjects,
                                                       dataset=dataset, device=device, use_age=use_age,
                                                       batch_size=batch_size, baseline=False, save_path=save_path)
    final_results["interpolate"] = {"results": res_interpolate, "predictions": pred_interpolate}

    save_dict_to_pickle(final_results, save_path, "datashift_results")
    return final_results
