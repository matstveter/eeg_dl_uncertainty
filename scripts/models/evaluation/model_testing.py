import argparse
import os.path
from torch.utils.data import DataLoader
import torch

from eegDlUncertainty.data.data_generators.CauDataGenerator import CauDataGenerator
from eegDlUncertainty.data.data_generators.ExplainabilityFrequencyGenerator import ExplainabilityFrequencyGenerator
from eegDlUncertainty.data.dataset.CauEEGDataset import CauEEGDataset
from eegDlUncertainty.data.results.history import History
from eegDlUncertainty.experiments.utils_exp import get_parameters_from_config
from eegDlUncertainty.models.classifiers.main_classifier import MainClassifier


def main():
    arg_parser = argparse.ArgumentParser(description="Run script for testing a model")
    arg_parser.add_argument("--result_folder", type=str, help="Path to result folder",
                            required=False)
    args = arg_parser.parse_args()
    args.result_folder = "/home/tvetern/PhD/dl_uncertainty/results/Keep/InceptionNetwork_2024-03-25 18_52_37/"

    path_objects = os.listdir(args.result_folder)
    for p in path_objects:
        if p.endswith(".json"):
            config_path = os.path.join(args.result_folder, p)

    parameters = get_parameters_from_config(config_path=config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(args.result_folder, "model", f"{parameters['classifier_name']}_model")

    dataset = CauEEGDataset(dataset_version=parameters['dataset_version'],
                            targets=parameters['prediction'],
                            eeg_len_seconds=parameters['num_seconds'],
                            epochs=parameters['eeg_epochs'])
    if dataset.num_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    _, _, test_subjects = dataset.get_splits()

    baseline = CauDataGenerator(subjects=test_subjects, dataset=dataset, device=device)
    baseline_loader = DataLoader(baseline, batch_size=parameters['batch_size'], shuffle=True)
    test_hist = History(num_classes=dataset.num_classes, set_name="test", loader_lenght=len(baseline_loader),
                        save_path=None)
    model = MainClassifier(model_name=parameters['classifier_name'], pretrained=model_path)
    model.test_model(test_loader=baseline_loader, device=device, loss_fn=criterion, test_hist=test_hist)

    for freq in ("delta", "theta", "alpha", "low_beta", "high_beta", "gamma"):
        print(f"\n--------------- Testing frequency band: {freq} ---------------")
        
        test_gen = ExplainabilityFrequencyGenerator(subjects=test_subjects, dataset=dataset, frequency_band=freq,
                                                    device=device, keep_band=False)
        test_loader = DataLoader(test_gen, batch_size=parameters['batch_size'], shuffle=True)
        test_hist = History(num_classes=dataset.num_classes, set_name="test", loader_lenght=len(test_loader),
                            save_path=None)
        model.test_model(test_loader=test_loader, device=device, loss_fn=criterion, test_hist=test_hist)


if __name__ == "__main__":
    main()
