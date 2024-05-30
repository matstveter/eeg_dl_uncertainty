import torch.cuda
import mlflow

from eegDlUncertainty.data.results.dataset_shifts import evaluate_dataset_shifts
from eegDlUncertainty.data.results.history import MCHistory
from eegDlUncertainty.data.results.uncertainty import get_uncertainty_metrics
from eegDlUncertainty.experiments.mainExperiment import BaseExperiment
from eegDlUncertainty.models.classifiers.swag_classifier import SWAGClassifier


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

    def run(self):
        # Start mflow, get data
        train_loader, val_loader, test_loader = self.prepare_run()
        self.create_model(**self.kwargs)

        try:
            self.train(train_loader=train_loader, val_loader=val_loader)
        except torch.cuda.OutOfMemoryError as e:
            mlflow.set_tag("Exception", "CUDA Out of Memory Error")
            mlflow.log_param("Exception Message", str(e))
            self.cleanup_function()
            print(f"Cuda Out Of Memory -> Cleanup -> Error message: {e}")
        else:
            self.test_models(test_loader=test_loader, val_loader=val_loader, use_temp_scaling=False)
            self.finish_run()
        finally:
            mlflow.end_run()

    @staticmethod
    def add_hyperparameters(change_in_hyperparameters, **kwargs):
        for k, v in change_in_hyperparameters.items():
            kwargs[k] = v
        return kwargs


class MCDExperiment(BaseExperiment):

    def create_model(self, **kwargs):
        if not kwargs['mc_dropout_enabled']:
            kwargs['mc_dropout_enabled'] = True
        self.model = self.get_model(model_name=self.model_name, **kwargs)

    def mc_dropout(self, test_loader, history):
        self.model.get_mc_predictions(test_loader=test_loader, device=self.device, history=history)

    def run(self):
        # Start mflow, get data
        train_loader, val_loader, test_loader = self.prepare_run()
        self.create_model(**self.kwargs)

        try:
            self.train(train_loader=train_loader, val_loader=val_loader)
        except torch.cuda.OutOfMemoryError as e:
            mlflow.set_tag("Exception", "CUDA Out of Memory Error")
            mlflow.log_param("Exception Message", str(e))
            self.cleanup_function()
            print(f"Cuda Out Of Memory -> Cleanup -> Error message: {e}")
        else:
            self.test_models(test_loader=test_loader, val_loader=val_loader, use_temp_scaling=False)

            mc_history = MCHistory(save_path=self.paths, num_classes=self.dataset.num_classes)

            # todo We can perhaps also use SWA as an option for all classifiers? At least test and save the performance

            self.mc_dropout(test_loader=test_loader, history=mc_history)
            mlflow.log_metric("MC Dropout Performance", mc_history.ensemble_accuracy)

            probs, targets = mc_history.get_prediction_set

            print(get_uncertainty_metrics(probs=probs, targets=targets))

            # evaluate_dataset_shifts(model=self.model,
            #                         test_subjects=self.test_subjects,
            #                         dataset=self.dataset,
            #                         device=self.device,
            #                         use_age=self.use_age,
            #                         monte_carlo=True,
            #                         batch_size=self.batch_size)
            # mc_history_calibrated = MCHistory(save_path=self.paths, num_classes=self.dataset.num_classes)
            # self.temperature_scaling(val_loader=val_loader)
            # self.mc_dropout(test_loader=test_loader, history=mc_history_calibrated)
            # probs_cal, targets_2 = mc_history_calibrated.get_prediction_set
            # print(brier_score(probs=probs_cal, targets=targets_2))
            self.finish_run()
        finally:
            mlflow.end_run()


class SWAGExperiment(BaseExperiment):

    def create_model(self, **kwargs):
        assert self.swag_enabled, "SWAG is not enabled!"
        self.model = self.get_model(model_name=self.model_name, **kwargs)

    def swag(self, train_loader, val_loader, test_loader, history):
        swag_classifier = SWAGClassifier(pretrained_model=self.model, learning_rate=self.learning_rate,
                                         save_path=self.paths,
                                         model_hyperparameters=self.model.hyperparameters,
                                         name=self.model_name)

        swag_classifier.fit(train_loader=train_loader, val_loader=val_loader, swa_epochs=2,
                            device=self.device, loss_fn=self.criterion, swa_lr=self.swag_lr)
        # todo Can probably use the forward method using the self.swa_g_classifier for testing the performance
        # todo Log performance using history objects...
        # todo can do the actual testing here as well!

        return swag_classifier

    def run(self):
        # Start mflow, get data
        train_loader, val_loader, test_loader = self.prepare_run()
        self.create_model(**self.kwargs)

        try:
            self.train(train_loader=train_loader, val_loader=val_loader)
        except torch.cuda.OutOfMemoryError as e:
            mlflow.set_tag("Exception", "CUDA Out of Memory Error")
            mlflow.log_param("Exception Message", str(e))
            self.cleanup_function()
            print(f"Cuda Out Of Memory -> Cleanup -> Error message: {e}")
        else:
            # self.test_models(test_loader=test_loader, val_loader=val_loader, use_temp_scaling=False)

            swag_history = MCHistory(save_path=self.paths, num_classes=self.dataset.num_classes)

            # todo We can perhaps also use SWA as an option for all classifiers? At least test and save the performance

            self.swag(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, history=swag_history)

            # todo save data from MCD experiment, all predictions from the models
            # todo Brier score, ECE, NLL
            # todo Test on new samples that have been augmented


            self.finish_run()


        finally:
            mlflow.end_run()
