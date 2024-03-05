from eegDlUncertainty.models.classifiers.brain_decode_classifiers import EEGNetv4MTSC


def get_models(model_name, **kwargs):

    available_models = (EEGNetv4MTSC,)

    for a_m in available_models:
        if model_name == a_m.__name__:
            return a_m(name=a_m, **kwargs)
    else:
        raise KeyError("")
