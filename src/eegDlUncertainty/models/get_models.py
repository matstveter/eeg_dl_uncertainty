from eegDlUncertainty.models.classifiers.inceptionTime import InceptionNetwork


def get_models(model_name, **kwargs):

    available_models = (InceptionNetwork,)

    for a_m in available_models:
        if model_name == a_m.__name__:
            return a_m(**kwargs)
    else:
        raise KeyError(f"Model name {model_name} not recognized, available models: {available_models}")
