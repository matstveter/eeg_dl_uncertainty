import os

from torch.utils.data import DataLoader
import numpy as np

from eegDlUncertainty.data.data_generators.ShiftGenerator import EEGDatashiftGenerator
from eegDlUncertainty.data.file_utils import save_dict_to_pickle


def evaluate_shift(shift_type, ensemble_class, test_subjects, dataset, use_age, device, batch_size, save_path,
                   **kwargs):
    print(f"\n----------- Evaluating shift type: {shift_type} -----------\n")

    if shift_type == "baseline":
        shift_intensity = [0.0]
    else:
        shift_intensity = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

    shift_results = {}
    for s in shift_intensity:
        print(f"\n----------- Shift intensity: {s} -----------\n")

        shift_dataset = EEGDatashiftGenerator(subjects=test_subjects,
                                              dataset=dataset,
                                              use_age=use_age,
                                              device=device,
                                              shift_type=shift_type,
                                              shift_intensity=s,
                                              random_seed=42,
                                              use_same_shift=False,
                                              plot_difference=True,
                                              **kwargs)
        shift_loader = DataLoader(dataset=shift_dataset, batch_size=batch_size, shuffle=False)
        shift_results[s] = ensemble_class.ensemble_performance_and_uncertainty(data_loader=shift_loader, device=device,
                                                                               save_path=save_path,
                                                                               save_name=f"Shift_{shift_type}_{s}",
                                                                               save_to_mlflow=True)
    return shift_results


def eval_dataset_shifts(ensemble_class, test_subjects, dataset, device, use_age, batch_size, save_path):

    # Create a new folder, called dataset_shifts, to save the results
    if use_age:
        save_path = os.path.join(save_path, "dataset_shifts")
    else:
        save_path = os.path.join(save_path, "dataset_shifts_without_age")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    baseline = evaluate_shift(shift_type="baseline", ensemble_class=ensemble_class,
                              test_subjects=test_subjects, dataset=dataset, device=device,
                              use_age=use_age, batch_size=batch_size, save_path=save_path)
    save_dict_to_pickle(baseline, save_path, "baseline")
    ############################################################################################################
    # Interpolation
    ############################################################################################################
    interpo = evaluate_shift(shift_type="interpolate", ensemble_class=ensemble_class,
                             test_subjects=test_subjects,
                             dataset=dataset, device=device, use_age=use_age,
                             batch_size=batch_size, save_path=save_path)
    save_dict_to_pickle(interpo, save_path, "interpolate")

    ############################################################################################################
    # Bandpass shift
    ############################################################################################################
    bandpass_results = {}
    bandpass = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    # Testing the effect of different bandpass filters
    for band in bandpass:
        print(f"\n----------- Evaluating bandpass shift: {band} -----------\n")
        band_res = evaluate_shift(shift_type=f"{band}_bandpass",
                                  ensemble_class=ensemble_class,
                                  test_subjects=test_subjects,
                                  dataset=dataset,
                                  device=device,
                                  use_age=use_age,
                                  batch_size=batch_size,
                                  save_path=save_path)
        bandpass_results[f"{band}_band"] = band_res

    save_dict_to_pickle(bandpass_results, save_path, "bandpass_results")
    ############################################################################################################
    # Circular shift
    ############################################################################################################
    circular_shifts_results = {}
    # Timepoints is 6000
    time_points_per_second = 200
    phase_shift = [1 * time_points_per_second, 2 * time_points_per_second, 4 * time_points_per_second,
                   8 * time_points_per_second, 10 * time_points_per_second, 15 * time_points_per_second,
                   20 * time_points_per_second, 30 * time_points_per_second]

    # Phase shift
    for p in phase_shift:
        print(f"\n----------- Evaluating circular shift: {p} -----------\n")
        phase = evaluate_shift(shift_type="circular_shift",
                               ensemble_class=ensemble_class,
                               test_subjects=test_subjects,
                               dataset=dataset,
                               device=device,
                               use_age=use_age,
                               batch_size=batch_size,
                               save_path=save_path,
                               circular_shift=p)
        circular_shifts_results[f"circular_shift_{p}"] = phase

    save_dict_to_pickle(circular_shifts_results, save_path, "circular_shift_results")
    ############################################################################################################
    # Phase shift
    ############################################################################################################
    phase_shift_results = {}
    phase_shift = [np.pi / 16, np.pi / 8, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2]

    # Phase shift
    for p in phase_shift:
        print(f"\n----------- Evaluating phase shift: {p} SAMPLED -----------\n")
        phase = evaluate_shift(shift_type="phase_shift",
                               ensemble_class=ensemble_class,
                               test_subjects=test_subjects,
                               dataset=dataset,
                               device=device,
                               use_age=use_age,
                               batch_size=batch_size,
                               save_path=save_path,
                               phase_shift=p)
        phase_shift_results[f"phase_shift_{p}"] = phase

    save_dict_to_pickle(phase_shift_results, save_path, "phase_shift_results")
    ############################################################################################################
    # Amplitude shift
    ############################################################################################################
    amplitude_results = {}
    amplitude_modulation = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.5, 2.0, 2.5, 4.0, 6.0, 8.0]

    for s in amplitude_modulation:
        print(f"\n----------- Evaluating amplitude: {s} SAMPLED -----------\n")
        scalar_c = evaluate_shift(shift_type="amplitude_change",
                                  ensemble_class=ensemble_class,
                                  test_subjects=test_subjects,
                                  dataset=dataset,
                                  device=device,
                                  use_age=use_age,
                                  batch_size=batch_size,
                                  save_path=save_path,
                                  scalar_multi=s)
        amplitude_results[f"amplitude_mod_{s}"] = scalar_c

    save_dict_to_pickle(amplitude_results, save_path, "amplitude_results")
    ############################################################################################################
    # Gaussian noise
    ############################################################################################################
    gaussian_results = {}
    gaussian_std = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 1.0, 1.5, 2.0, 2.5, 4.0, 8.0]

    # Gaussian noise
    for g in gaussian_std:
        print(f"\n----------- Evaluating gaussian noise: {g} SAMPLED -----------\n")
        gaussian_ch = evaluate_shift(shift_type="gaussian",
                                     ensemble_class=ensemble_class,
                                     test_subjects=test_subjects,
                                     dataset=dataset,
                                     device=device,
                                     use_age=use_age,
                                     batch_size=batch_size,
                                     save_path=save_path,
                                     gaussian_std=g)
        gaussian_results[f"gaussian_sampled_{g}"] = gaussian_ch

    save_dict_to_pickle(gaussian_results, save_path, "gaussian_results")
    ############################################################################################################
    # Baseline drift
    ############################################################################################################
    baseline_drift_results = {}
    max_drift_values = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
    num_sinus_waves = [1, 2, 3, 5, 7, 10, 15]

    for num_sinus_wave in num_sinus_waves:
        for max_drift in max_drift_values:
            print(f"\n----------- Evaluating baseline drift: {max_drift} / {num_sinus_wave} SAMPLED -----------\n")
            drift_results = evaluate_shift(
                shift_type="baseline_drift",
                ensemble_class=ensemble_class,
                test_subjects=test_subjects,
                dataset=dataset,
                device=device,
                use_age=use_age,
                batch_size=batch_size,
                save_path=save_path,
                max_drift_amplitude=max_drift,
                num_sinusoids=num_sinus_wave)

            baseline_drift_results[f"baseline_drift_max_drift_{max_drift}_num_sinus_{num_sinus_wave}"] = drift_results
    save_dict_to_pickle(baseline_drift_results, save_path, "baseline_drift_results")
    ############################################################################################################
    # Timewarp
    ############################################################################################################
    timewarp_results = {}
    max_warp_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75]

    for max_warp in max_warp_values:
        print(f"\n----------- Evaluating timewarp: {max_warp} SAMPLED -----------\n")
        drift_results = evaluate_shift(
            shift_type="timewarp",
            ensemble_class=ensemble_class,
            test_subjects=test_subjects,
            dataset=dataset,
            device=device,
            use_age=use_age,
            batch_size=batch_size,
            save_path=save_path,
            max_warp_ratio=max_warp)

        timewarp_results[f"timewarp_max_warp_{max_warp}"] = drift_results

    save_dict_to_pickle(timewarp_results, save_path, "timewarp_results")
    ############################################################################################################
    # Peak shift frequency
    ############################################################################################################
    peak_shift_results = {}
    peak_shift = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]

    for p in peak_shift:
        print(f"\n----------- Evaluating alpha peak shift {p} -----------\n")
        band_res = evaluate_shift(shift_type=f"alpha_peak_shift",
                                  ensemble_class=ensemble_class,
                                  test_subjects=test_subjects,
                                  dataset=dataset,
                                  device=device,
                                  use_age=use_age,
                                  batch_size=batch_size,
                                  save_path=save_path,
                                  peak_shift=p)
        peak_shift_results[f"alpha_peak_shift_{p}"] = band_res
    save_dict_to_pickle(peak_shift_results, save_path, "peak_shift_results")
    ############################################################################################################
    # Channel rotation
    ############################################################################################################
    channel_rotation_results = {}
    band_res = evaluate_shift(shift_type=f"rotate_channels",
                              ensemble_class=ensemble_class,
                              test_subjects=test_subjects,
                              dataset=dataset,
                              device=device,
                              use_age=use_age,
                              batch_size=batch_size,
                              save_path=save_path)
    channel_rotation_results[f"rotate_channels"] = band_res
    save_dict_to_pickle(channel_rotation_results, save_path, "channel_rotation_results")
