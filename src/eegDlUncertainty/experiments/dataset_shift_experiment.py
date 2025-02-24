from torch.utils.data import DataLoader
import numpy as np

from eegDlUncertainty.data.data_generators.ShiftGenerator import EEGDatashiftGenerator
from eegDlUncertainty.data.results.plotter import single_datashift_plotter
from eegDlUncertainty.data.utils import save_dict_to_pickle


def evaluate_shift(shift_type, ensemble_class, test_subjects, dataset, use_age, device, batch_size, save_path,
                   use_same_shift, **kwargs):
    print(f"\n----------- Evaluating shift type: {shift_type} -----------\n")

    shift_intensity = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

    shift_results = {}
    for s in shift_intensity:
        shift_dataset = EEGDatashiftGenerator(subjects=test_subjects,
                                              dataset=dataset,
                                              use_age=use_age,
                                              device=device,
                                              shift_type=shift_type,
                                              shift_intensity=s,
                                              random_seed=42,
                                              use_same_shift=use_same_shift,
                                              plot_difference=True,
                                              **kwargs)
        shift_loader = DataLoader(dataset=shift_dataset, batch_size=batch_size, shuffle=False)
        shift_results[s] = ensemble_class.ensemble_performance_and_uncertainty(data_loader=shift_loader, device=device,
                                                                               save_path=save_path,
                                                                               save_name=f"Shift_{shift_type}_{s}",
                                                                               save_to_mlflow=True)
    single_datashift_plotter(shift_result=shift_results, shift_type=shift_type, save_path=save_path)
    return shift_results


def eval_dataset_shifts(ensemble_class, test_subjects, dataset, device, use_age, batch_size, save_path):
    final_results = {}

    ############################################################################################################
    # Interpolation
    ############################################################################################################

    # interpo = evaluate_shift(shift_type="interpolate", ensemble_class=ensemble_class,
    #                          test_subjects=test_subjects,
    #                          dataset=dataset, device=device, use_age=use_age,
    #                          batch_size=batch_size, use_same_shift=False, save_path=save_path)
    # final_results["interpolate"] = interpo

    ############################################################################################################
    # Bandpass shift
    ############################################################################################################

    # bandpass = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    # bandpass = ['beta']
    # # Testing the effect of different bandpass filters
    # for band in bandpass:
    #     band_res = evaluate_shift(shift_type=f"{band}_bandpass", ensemble_class=ensemble_class,
    #                               test_subjects=test_subjects, dataset=dataset, device=device,
    #                               use_age=use_age, batch_size=batch_size, use_same_shift=False,
    #                               save_path=save_path)
    #     final_results[f"{band}_band"] = band_res

    ############################################################################################################
    # Circular shift
    ############################################################################################################

    # # Timepoints is 6000, so we crate shifts that are somewhat reasonable (1%, 5%, 10%, 25%, 50%)
    # num_time_points = 6000
    # phase_shift = [0.01 * num_time_points, 0.05 * num_time_points,
    #                0.1 * num_time_points, 0.25 * num_time_points, 0.5 * num_time_points]
    # # Phase shift
    # for p in phase_shift:
    #     phase = evaluate_shift(shift_type="circular_shift",
    #                            ensemble_class=ensemble_class,
    #                            test_subjects=test_subjects,
    #                            dataset=dataset,
    #                            device=device,
    #                            use_age=use_age,
    #                            batch_size=batch_size,
    #                            save_path=save_path,
    #                            use_same_shift=False,
    #                            circular_shift=p)
    #     final_results[f"circular_shift_sampled_{p}"] = phase
    #
    # # Here we test the effect of the same shift on all channels
    # for p in phase_shift:
    #     phase = evaluate_shift(shift_type="circular_shift",
    #                            ensemble_class=ensemble_class,
    #                            test_subjects=test_subjects,
    #                            dataset=dataset,
    #                            device=device,
    #                            use_age=use_age,
    #                            batch_size=batch_size,
    #                            save_path=save_path,
    #                            use_same_shift=True,
    #                            circular_shift=p)
    #     final_results[f"circular_shift_static_{p}"] = phase

    ############################################################################################################
    # Phase shift
    ############################################################################################################

    # phase_shift = [np.pi / 16, np.pi / 8, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]
    # # Phase shift
    # for p in phase_shift:
    #     phase = evaluate_shift(shift_type="phase_shift",
    #                            ensemble_class=ensemble_class,
    #                            test_subjects=test_subjects,
    #                            dataset=dataset,
    #                            device=device,
    #                            use_age=use_age,
    #                            batch_size=batch_size,
    #                            save_path=save_path,
    #                            use_same_shift=False,
    #                            phase_shift=p)
    #     final_results[f"phase_shift_sampled_{p}"] = phase
    #
    # for p in phase_shift:
    #     phase = evaluate_shift(shift_type="phase_shift",
    #                            ensemble_class=ensemble_class,
    #                            test_subjects=test_subjects,
    #                            dataset=dataset,
    #                            device=device,
    #                            use_age=use_age,
    #                            batch_size=batch_size,
    #                            save_path=save_path,
    #                            use_same_shift=True,
    #                            phase_shift=p)
    #     final_results[f"phase_shift_static_{p}"] = phase

    ############################################################################################################
    # Amplitude shift
    ############################################################################################################
    # amplitude_modulation = [0.01, 0.05, 0.1, 0.5, 0.75, 1.5, 2.0, 4.0, 8.0]
    #
    # for s in amplitude_modulation:
    #     scalar_c = evaluate_shift(shift_type="amplitude_change",
    #                               ensemble_class=ensemble_class,
    #                               test_subjects=test_subjects,
    #                               dataset=dataset,
    #                               device=device,
    #                               use_age=use_age,
    #                               batch_size=batch_size,
    #                               save_path=save_path,
    #                               use_same_shift=False,
    #                               scalar_multi=s)
    #     final_results[f"amplitude_mod_sampled_{s}"] = scalar_c
    #
    # for s in amplitude_modulation:
    #     scalar_c = evaluate_shift(shift_type="amplitude_change",
    #                               ensemble_class=ensemble_class,
    #                               test_subjects=test_subjects,
    #                               dataset=dataset,
    #                               device=device,
    #                               use_age=use_age,
    #                               batch_size=batch_size,
    #                               save_path=save_path,
    #                               use_same_shift=True,
    #                               scalar_multi=s)
    #     final_results[f"amplitude_mod_static_{s}"] = scalar_c

    ############################################################################################################
    # Gaussian noise
    ############################################################################################################

    # gaussian_std = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0]
    # # Gaussian noise
    # for g in gaussian_std:
    #     gaussian_ch = evaluate_shift(shift_type="gaussian",
    #                                  ensemble_class=ensemble_class,
    #                                  test_subjects=test_subjects,
    #                                  dataset=dataset,
    #                                  device=device,
    #                                  use_age=use_age,
    #                                  batch_size=batch_size,
    #                                  save_path=save_path,
    #                                  use_same_shift=False,
    #                                  gaussian_std=g)
    #     final_results[f"gaussian_sampled_{g}"] = gaussian_ch
    
    # for g in gaussian_std:
    #     gaussian_ch = evaluate_shift(shift_type="gaussian",
    #                                  ensemble_class=ensemble_class,
    #                                  test_subjects=test_subjects,
    #                                  dataset=dataset,
    #                                  device=device,
    #                                  use_age=use_age,
    #                                  batch_size=batch_size,
    #                                  save_path=save_path,
    #                                  use_same_shift=True,
    #                                  gaussian_std=g)
    #     final_results[f"gaussian_static_{g}"] = gaussian_ch

    ############################################################################################################
    # Baseline drift
    ############################################################################################################

    # # max_drift_values = [0.01, 0.1, 0.25, 0.5]
    # # num_sinus_waves = [1, 2, 3]
    #
    # for num_sinus_wave in num_sinus_waves:
    #     for max_drift in max_drift_values:
    #
    #         drift_results = evaluate_shift(
    #             shift_type="baseline_drift",
    #             ensemble_class=ensemble_class,
    #             test_subjects=test_subjects,
    #             dataset=dataset,
    #             device=device,
    #             use_age=use_age,
    #             batch_size=batch_size,
    #             save_path=save_path,
    #             max_drift_amplitude=max_drift,
    #             num_sinusoids=num_sinus_wave,
    #             use_same_shift=False)
    #
    #         final_results[f"baseline_drift_max_drift_{max_drift}_num_sinus_{num_sinus_wave}_sampled"] = drift_results
    #
    # for num_sinus_wave in num_sinus_waves:
    #     for max_drift in max_drift_values:
    #
    #         drift_results = evaluate_shift(
    #             shift_type="baseline_drift",
    #             ensemble_class=ensemble_class,
    #             test_subjects=test_subjects,
    #             dataset=dataset,
    #             device=device,
    #             use_age=use_age,
    #             batch_size=batch_size,
    #             save_path=save_path,
    #             max_drift_amplitude=max_drift,
    #             num_sinusoids=num_sinus_wave,
    #             use_same_shift=True)
    #
    #         final_results[f"baseline_drift_max_drift_{max_drift}_num_sinus_{num_sinus_wave}_static"] = drift_results

    ############################################################################################################
    # Timewarp
    ############################################################################################################
    # max_warp_values = [0.01]
    # for max_warp in max_warp_values:
    #
    #     drift_results = evaluate_shift(
    #         shift_type="timewarp",
    #         ensemble_class=ensemble_class,
    #         test_subjects=test_subjects,
    #         dataset=dataset,
    #         device=device,
    #         use_age=use_age,
    #         batch_size=batch_size,
    #         save_path=save_path,
    #         max_warp_ratio=max_warp,
    #         use_same_shift=False)
    #
    #     final_results[f"timewarp_max_warp_{max_warp}_sampled"] = drift_results

    ############################################################################################################
    # Peak shift frequency
    ############################################################################################################
    peak_shift = [0.5, 1.0, 2.0, 4.0]

    for p in peak_shift:
        band_res = evaluate_shift(shift_type=f"alpha_peak_shift",
                                  ensemble_class=ensemble_class,
                                  test_subjects=test_subjects,
                                  dataset=dataset,
                                  device=device,
                                  use_age=use_age,
                                  batch_size=batch_size,
                                  use_same_shift=False,
                                  save_path=save_path,
                                  peak_shift=p)
        final_results[f"alpha_peak_shift_{p}_sampled"] = band_res

    for p in peak_shift:
        band_res = evaluate_shift(shift_type=f"alpha_peak_shift",
                                  ensemble_class=ensemble_class,
                                  test_subjects=test_subjects,
                                  dataset=dataset,
                                  device=device,
                                  use_age=use_age,
                                  batch_size=batch_size,
                                  use_same_shift=True,
                                  save_path=save_path,
                                  peak_shift=p)
        final_results[f"alpha_peak_shift_{p}_static"] = band_res

    ############################################################################################################
    # Channel rotation
    ############################################################################################################
    # band_res = evaluate_shift(shift_type=f"rotate_channels",
    #                           ensemble_class=ensemble_class,
    #                           test_subjects=test_subjects,
    #                           dataset=dataset,
    #                           device=device,
    #                           use_age=use_age,
    #                           batch_size=batch_size,
    #                           use_same_shift=False,
    #                           save_path=save_path)
    # final_results[f"rotate_channels"] = band_res
    # raise ValueError("Not implemented")
    

    save_dict_to_pickle(final_results, save_path, "datashift_results")
    return final_results
