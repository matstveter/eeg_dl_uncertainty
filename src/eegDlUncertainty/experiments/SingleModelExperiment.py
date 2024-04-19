from eegDlUncertainty.experiments.mainExperiment import BaseExperiment
from eegDlUncertainty.models.classifiers.age_included_classifier import AgeClassifier
from eegDlUncertainty.models.classifiers.main_classifier import MainClassifier


class SingleModelExperiment(BaseExperiment):
    def create_model(self, **kwargs):
        if self.dataset is not None:
            hyperparameters = {"in_channels": self.dataset.num_channels,
                               "num_classes": self.dataset.num_classes,
                               "time_steps": self.dataset.eeg_len,
                               "save_path": self.paths,
                               "lr": self.learning_rate}
            kwargs.update(hyperparameters)
            self.model = self.get_model(model_name=self.model_name, **kwargs)

        else:
            raise ValueError("Dataset is not provided!")

    @staticmethod
    def add_hyperparameters(change_in_hyperparameters, **kwargs):
        for k, v in change_in_hyperparameters.items():
            kwargs[k] = v
        return kwargs
