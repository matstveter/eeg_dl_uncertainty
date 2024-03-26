import json


def get_baseparameters_from_config(config_path):
    with open(config_path) as json_file:
        config = json.load(json_file)

    kwargs = {
        'prediction': config['data']['prediction'],
        'dataset_version': config['data']['version'],
        'eeg_epochs': config['data']['eeg_epochs'],
        'num_seconds': config['data']['num_seconds'],
        'prediction_type': config['data']['prediction_type'],
        'which_one_vs_all': config['data']['which_one_vs_all_class'],
        'pairwise_class': config['data']['pairwise'],
        'classifier_name': config['model']['name'],
        'learning_rate': config['hyperparameters']['learning_rate'],
        'batch_size': config['hyperparameters']['batch_size'],
        'training_epochs': config['hyperparameters']['epochs'],
        'augmentations': config['hyperparameters']['augmentations'],
    }

    return kwargs
