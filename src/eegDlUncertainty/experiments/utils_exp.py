import json


def get_baseparameters_from_config(config_path):
    with open(config_path) as json_file:
        config = json.load(json_file)

    kwargs = {
        'prediction': config['data']['prediction'],
        'dataset_version': config['data']['version'],
        'eeg_epochs': config['data']['eeg_epochs'],
        'epoch_overlap': config['data']['overlapping_epochs'],
        'num_seconds': config['data']['num_seconds'],
        'prediction_type': config['data']['prediction_type'],
        'which_one_vs_all': config['data']['which_one_vs_all_class'],
        'pairwise_class': config['data']['pairwise'],
        'classifier_name': config['model']['name'],
        'learning_rate': config['hyperparameters']['learning_rate'],
        'batch_size': config['hyperparameters']['batch_size'],
        'augmentations': config['hyperparameters']['augmentations'],
        'training_epochs': config['model']['epochs'],
        'earlystopping': config['model']['earlystopping']
    }
    possible_prediction_types = ("normal", 'pairwise', 'one_vs_all')
    possible_predictions = ('dementia', 'abnormal')
    if kwargs['prediction_type'] not in possible_prediction_types:
        raise KeyError(f"Prediction type should be one of {possible_prediction_types}, "
                       f"got '{kwargs['prediction_type']}'")

    if kwargs['prediction'] not in possible_predictions:
        raise KeyError(f"Prediction not in in {possible_predictions}, got '{kwargs['prediction']}'")

    return kwargs
