from torch.utils.data import DataLoader

from eegDlUncertainty.data.data_generators.ShiftGenerators import EEGDatashiftGenerator
from eegDlUncertainty.data.results.plotter import single_datashift_plotter
from eegDlUncertainty.data.utils import save_dict_to_pickle


def evaluate_shift(shift_type, ensemble_class, test_subjects, dataset, use_age, device, batch_size, baseline, save_path,
                   **kwargs):
    print(f"\n----------- Evaluating shift type: {shift_type} -----------\n")
    if baseline:
        shift_intensity = [0.0]
    else:
        shift_intensity = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

    shift_results = {}
    for s in shift_intensity:
        shift_dataset = EEGDatashiftGenerator(subjects=test_subjects, dataset=dataset, use_age=use_age, device=device,
                                              shift_type=shift_type, shift_intensity=s, **kwargs)
        shift_loader = DataLoader(dataset=shift_dataset, batch_size=batch_size, shuffle=False)
        shift_results[s] = ensemble_class.ensemble_performance_and_uncertainty(data_loader=shift_loader, device=device,
                                                                               save_path=save_path,
                                                                               save_name=f"Shift_{shift_type}_{s}",
                                                                               save_to_mlflow=False)
    single_datashift_plotter(shift_result=shift_results, shift_type=shift_type, save_path=save_path)
    return shift_results


def eval_dataset_shifts(ensemble_class, test_subjects, dataset, device, use_age, batch_size, save_path):
    final_results = {}

    # res_baseline, pred_baseline = evaluate_shift(shift_type=None, ensemble_class=ensemble_class,
    #                                              test_subjects=test_subjects,
    #                                              dataset=dataset, device=device, use_age=use_age, batch_size=batch_size,
    #                                              baseline=True, save_path=save_path)
    # final_results["baseline"] = {"results": res_baseline, "predictions": pred_baseline}
    class_combi = evaluate_shift(shift_type="class_combination_temporal",
                                 ensemble_class=ensemble_class,
                                 test_subjects=test_subjects, dataset=dataset, device=device,
                                 use_age=use_age, batch_size=batch_size, baseline=False,
                                 save_path=save_path)
    final_results["class_combination"] = class_combi
    # bandpass = ['delta', 'theta', 'alpha', 'beta', 'hbeta', 'lbeta', 'gamma']
    # # Testing the effect of different bandpass filters
    # for band in bandpass:
    #     band = evaluate_shift(shift_type=f"{band}_bandpass", ensemble_class=ensemble_class,
    #                           test_subjects=test_subjects, dataset=dataset, device=device,
    #                           use_age=use_age, batch_size=batch_size,
    #                           baseline=False, save_path=save_path)
    #     final_results[band] = band
    # gaussian_std = [0.1, 0.2, 0.4, 0.8, 1.0]
    # # Gaussian noise
    # for g in gaussian_std:
    #     gaussian_ch = evaluate_shift(shift_type="gaussian_channel", ensemble_class=ensemble_class,
    #                                  test_subjects=test_subjects, dataset=dataset,
    #                                  device=device, use_age=use_age, batch_size=batch_size,
    #                                  baseline=False, save_path=save_path, gaussian_std=g)
    #     final_results[f"gaussian_channel_{g}"] = gaussian_ch
    #
    # gaussian_all = evaluate_shift(shift_type="gaussian", ensemble_class=ensemble_class, test_subjects=test_subjects,
    #                               dataset=dataset, device=device, use_age=use_age, batch_size=batch_size,
    #                               baseline=False, save_path=save_path)
    # final_results["gaussian_all"] = gaussian_all

    # phase_shift = [np.pi / 8, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]
    # # Phase shift
    # for p in phase_shift:
    #     phase = evaluate_shift(shift_type="phase_shift_channel",
    #                            ensemble_class=ensemble_class,
    #                            test_subjects=test_subjects, dataset=dataset,
    #                            device=device, use_age=use_age, batch_size=batch_size,
    #                            baseline=False, save_path=save_path, phase_shift=p)
    #     final_results[f"phase_shift_channel_{p}"] = phase
    #
    # scalar_modulation = [0.1, 0.5, 0.75, 1.5, 2.0]
    # Scalar modulation
    # for s in scalar_modulation:
    #     scalar = evaluate_shift(shift_type="scalar_modulation", ensemble_class=ensemble_class,
    #                             test_subjects=test_subjects,
    #                             dataset=dataset, device=device, use_age=use_age, batch_size=batch_size,
    #                             baseline=False, save_path=save_path,
    #                             scalar_multi=s)
    #     final_results[f"scalar_modulation_{s}"] = scalar
    #
    # for s in scalar_modulation:
    #     scalar_c = evaluate_shift(shift_type="scalar_modulation_channel",
    #                               ensemble_class=ensemble_class,
    #                               test_subjects=test_subjects, dataset=dataset, device=device,
    #                               use_age=use_age, batch_size=batch_size, baseline=False,
    #                               save_path=save_path,
    #                               scalar_multi=s)
    #     final_results[f"scalar_modulation_channel_{s}"] = scalar_c
    # interpolate
    interpo = evaluate_shift(shift_type="interpolate", ensemble_class=ensemble_class,
                             test_subjects=test_subjects,
                             dataset=dataset, device=device, use_age=use_age,
                             batch_size=batch_size, baseline=False, save_path=save_path)
    final_results["interpolate"] = interpo

    save_dict_to_pickle(final_results, save_path, "datashift_results")
    return final_results
