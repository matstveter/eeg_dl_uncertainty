from eegDlUncertainty.models.classifiers.brain_decode_classifiers import (Deep4NetMTSC, EEGITNetMTSC, EEGNetv4MTSC,
                                                                          EEGNetv1MTSC)
from eegDlUncertainty.models.classifiers.inceptionTime import InceptionNetwork


def get_models(model_name, **kwargs):

    available_models = (EEGNetv4MTSC, EEGNetv1MTSC, Deep4NetMTSC, EEGITNetMTSC, InceptionNetwork)

    for a_m in available_models:
        if model_name == a_m.__name__:
            return a_m(**kwargs)
    else:
        raise KeyError(f"Model name {model_name} not recognized, available models: {available_models}")
