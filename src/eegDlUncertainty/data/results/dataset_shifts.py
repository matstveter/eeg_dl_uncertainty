import numpy as np
import torch
from torch.utils.data import DataLoader

from eegDlUncertainty.data.data_generators.ShiftGenerators import EEGDatashiftGenerator
from eegDlUncertainty.data.results.uncertainty import calculate_performance_metrics, compute_classwise_uncertainty, \
    get_uncertainty_metrics
from eegDlUncertainty.data.utils import save_dict_to_pickle


def evaluate_shift(shift_type, model, test_subjects, dataset, use_age, device, batch_size, baseline):
    if baseline:
        shift_intensity = [0.0]
    else:
        shift_intensity = [0.1, 0.2, 0.4, 0.8, 1.0]

    shift_results = {}
    shift_predictions = {}

    for s in shift_intensity:
        shift_dataset = EEGDatashiftGenerator(subjects=test_subjects, dataset=dataset, use_age=use_age, device=device,
                                              shift_type=shift_type, shift_intensity=s)
        shift_loader = DataLoader(dataset=shift_dataset, batch_size=batch_size, shuffle=False)

        if not isinstance(model, list):
            print("Testing with the combined EEG")
            logits, targets = model.get_mc_predictions(test_loader=shift_loader, device=device, history=None,
                                                       num_forward=50)
        else:
            print("Testing with the ensemble")
            logits = []
            for m in model:
                ensemble_logits, targets = m.get_predictions(test_loader=shift_loader, device=device)
                logits.append(ensemble_logits)
            logits = np.array(logits)  # shape: (num_models, num_samples, num_classes)

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
                            f"uncertainty": uncertainty,
                            "class_uncertainty": class_uncertainty}

    return shift_results, shift_predictions


def evaluate_dataset_shifts(model, test_subjects, dataset, device, use_age, batch_size, save_path):
    res_baseline, pred_baseline = evaluate_shift(shift_type=None, model=model, test_subjects=test_subjects,
                                                 dataset=dataset, device=device, use_age=use_age, batch_size=batch_size,
                                                 baseline=True)

    res_class_combi, pred_class_combi = evaluate_shift(shift_type="class_combination_temporal", model=model,
                                                       test_subjects=test_subjects, dataset=dataset, device=device,
                                                       use_age=use_age, batch_size=batch_size, baseline=False)

    # Testing the effect of different bandpass filters
    res_alpha, pred_alpha = evaluate_shift(shift_type="alpha_bandpass", model=model, test_subjects=test_subjects,
                                           dataset=dataset, device=device, use_age=use_age, batch_size=batch_size,
                                           baseline=False)
    res_beta, pred_beta = evaluate_shift(shift_type="beta_bandpass", model=model, test_subjects=test_subjects,
                                         dataset=dataset, device=device, use_age=use_age, batch_size=batch_size,
                                         baseline=False)
    res_hbeta, pred_hbeta = evaluate_shift(shift_type="hbeta_bandpass", model=model, test_subjects=test_subjects,
                                           dataset=dataset, device=device, use_age=use_age, batch_size=batch_size,
                                           baseline=False)
    res_lbeta, pred_lbeta = evaluate_shift(shift_type="lbeta_bandpass", model=model, test_subjects=test_subjects,
                                           dataset=dataset, device=device, use_age=use_age, batch_size=batch_size,
                                           baseline=False)
    res_theta, pred_theta = evaluate_shift(shift_type="theta_bandpass", model=model, test_subjects=test_subjects,
                                           dataset=dataset, device=device, use_age=use_age, batch_size=batch_size,
                                           baseline=False)
    res_delta, pred_delta = evaluate_shift(shift_type="delta_bandpass", model=model, test_subjects=test_subjects,
                                           dataset=dataset, device=device, use_age=use_age, batch_size=batch_size,
                                           baseline=False)
    res_gamma, pred_gamma = evaluate_shift(shift_type="gamma_bandpass", model=model, test_subjects=test_subjects,
                                           dataset=dataset, device=device, use_age=use_age, batch_size=batch_size,
                                           baseline=False)

    # Gaussian noise
    res_gaussian, pred_gaussian = evaluate_shift(shift_type="gaussian", model=model, test_subjects=test_subjects,
                                                 dataset=dataset, device=device, use_age=use_age, batch_size=batch_size,
                                                 baseline=False)
    res_gaussian_c, pred_gaussian_c = evaluate_shift(shift_type="gaussian_channel", model=model,
                                                     test_subjects=test_subjects,
                                                     dataset=dataset, device=device, use_age=use_age,
                                                     batch_size=batch_size, baseline=False)

    # Phase shift
    res_phase_shift_c, pred_phase_shift_c = evaluate_shift(shift_type="phase_shift_channel", model=model,
                                                           test_subjects=test_subjects,
                                                           dataset=dataset, device=device, use_age=use_age,
                                                           batch_size=batch_size, baseline=False)

    # Scalar modulation
    # todo Remember to change the scalar modulation value in the datashift generator
    res_scalar, pred_scalar = evaluate_shift(shift_type="scalar_modulation", model=model, test_subjects=test_subjects,
                                             dataset=dataset, device=device, use_age=use_age, batch_size=batch_size,
                                             baseline=False)
    res_scalar_c, pred_scalar_c = evaluate_shift(shift_type="scalar_modulation_channel", model=model,
                                                 test_subjects=test_subjects,
                                                 dataset=dataset, device=device, use_age=use_age, batch_size=batch_size,
                                                 baseline=False)

    # interpolate
    res_interpolate, pred_interpolate = evaluate_shift(shift_type="interpolate", model=model,
                                                       test_subjects=test_subjects,
                                                       dataset=dataset, device=device, use_age=use_age,
                                                       batch_size=batch_size, baseline=False)

    # todo Consider adding muscle artifact, eye blink, and eye movement as noise?

    final_results = {"baseline": {"results": res_baseline, "predictions": pred_baseline},
                     "class_combination": {"results": res_class_combi, "predictions": pred_class_combi},
                     "alpha": {"results": res_alpha, "predictions": pred_alpha},
                     "beta": {"results": res_beta, "predictions": pred_beta},
                     "hbeta": {"results": res_hbeta, "predictions": pred_hbeta},
                     "lbeta": {"results": res_lbeta, "predictions": pred_lbeta},
                     "theta": {"results": res_theta, "predictions": pred_theta},
                     "delta": {"results": res_delta, "predictions": pred_delta},
                     "gamma": {"results": res_gamma, "predictions": pred_gamma},
                     "gaussian": {"results": res_gaussian, "predictions": pred_gaussian},
                     "gaussian_channel": {"results": res_gaussian_c, "predictions": pred_gaussian_c},
                     "phase_shift_channel": {"results": res_phase_shift_c, "predictions": pred_phase_shift_c},
                     "scalar_modulation": {"results": res_scalar, "predictions": pred_scalar},
                     "scalar_modulation_channel": {"results": res_scalar_c, "predictions": pred_scalar_c},
                     "interpolate": {"results": res_interpolate, "predictions": pred_interpolate}}

    save_dict_to_pickle(final_results, save_path, "datashift_results")

    return final_results
