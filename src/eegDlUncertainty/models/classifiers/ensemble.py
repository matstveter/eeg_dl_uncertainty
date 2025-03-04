from collections import defaultdict
from pickletools import optimize
from typing import List

import mlflow
import numpy as np
import torch

import torch.nn as nn

from eegDlUncertainty.data.results.uncertainty import calculate_performance_metrics, compute_classwise_uncertainty, \
    get_uncertainty_metrics
from eegDlUncertainty.data.utils import save_dict_to_pickle


class Ensemble(torch.nn.Module):
    def __init__(self, classifiers, device):
        super(Ensemble, self).__init__()

        self.classifiers = classifiers
        self.device = device
        if isinstance(classifiers, list):
            self.method = "ensemble"
        else:
            self.method = "single"

        self.temperature = torch.nn.Parameter(torch.ones(1))

    def forward(self, x, apply_mean=True):
        """ Forward pass of the ensemble model. The output is the average of the logits of the classifiers.
            If the model is either swag or MCD, it uses the forward_ensemble method of the classifier.
            It divides the output by the temperature parameter.

        Parameters
        ----------
        apply_mean
        x : torch.Tensor
            The input tensor containing data for prediction. The shape and data type
            of `x` should be compatible with the model's expected input.

        Returns
        -------
        torch.Tensor: The output tensor containing the predicted logits. The shape and data type
            of the output tensor should be compatible with the model's expected output.

        """

        self.to(device=self.device)
        self.temperature = self.temperature.to(device=self.device)
        if self.method == "ensemble":
            logits = []
            for m in self.classifiers:
                logits.append(m(x))
            logits = torch.stack(logits)
        else:
            logits = self.classifiers.forward_ensemble(x)

        if apply_mean:
            logits = torch.mean(logits, dim=0)
            # temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
            return logits / self.temperature
        else:
            return logits / self.temperature

    def predict(self, x: torch.Tensor):
        return self.activation_function(logits=self.forward(x), ret_prob=False)

    def predict_prob(self, x: torch.Tensor):
        return self.activation_function(logits=self.forward(x), ret_prob=True)

    def set_temperature_scale_ensemble(self, data_loader, device, criterion, patience=250):
        self.eval()
        self.to(device)
        # Setting this to 1.5 based on the original paper
        self.temperature.data.fill_(1.0)
        self.temperature = self.temperature.to(device)
        self.temperature.requires_grad = True  # Ensure it's a learnable parameter

        nll_criterion = nn.CrossEntropyLoss().cuda()

        logits_list = []
        labels_list = []

        with torch.no_grad():
            for data, labels in data_loader:
                data, labels = data.to(device), labels.to(device)
                logits = self(data)

                logits_list.append(logits)
                labels_list.append(labels)

        # Concatenate along the correct dimension
        logits = torch.cat(logits_list, dim=0)  # Shape: (N, num_classes)
        labels = torch.cat(labels_list, dim=0)  # Shape: (N,)

        # 🔹 Print shapes to debug
        print(f"Final logits shape: {logits.shape}")  # Expected (N, 3)
        print(f"Final labels shape: {labels.shape}")  # Expected (N,)

        # 🔹 Ensure labels are integer class indices (not one-hot)
        if labels.ndim > 1:
            labels = labels.argmax(dim=1)  # Convert (N, 3) → (N,)

        # 🔹 Ensure correct dtype for torchmetrics
        labels = labels.to(torch.int64)

        nll_criterion_before = nll_criterion(logits, labels).item()

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=250)

        def evaluation():
            optimizer.zero_grad()
            # temp = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
            # loss = criterion(logits / temp, labels)
            loss = criterion(logits / self.temperature, labels)
            loss.backward()
            print(f"Loss: {loss.item()}")
            return loss

        optimizer.step(evaluation)

        nll_criterion_after = nll_criterion(logits / self.temperature, labels).item()

        print(f"Temperature: {self.temperature.item()}, "
              f"NLL before: {nll_criterion_before}, "
              f"NLL after: {nll_criterion_after}")

        print("Temperature: ", self.temperature.item())
        mlflow.log_metric("optimal_temp_ensemble", self.temperature.item())

        from sklearn.calibration import calibration_curve
        import matplotlib.pyplot as plt
        from torchmetrics.classification import MulticlassCalibrationError

        ece_metric = MulticlassCalibrationError(num_classes=3, n_bins=10, norm='l1')

        probs_before = torch.nn.functional.softmax(logits, dim=1)
        probs_after = torch.nn.functional.softmax(logits / self.temperature, dim=1)

        ece_before = ece_metric(probs_before, labels).item()
        ece_after = ece_metric(probs_after, labels).item()
        print(f"ECE Before Scaling: {ece_before:.3f}, ECE After Scaling: {ece_after:.3f}")

        num_classes = 3  # Adjust for your dataset

        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")  # Diagonal reference line

        for class_idx in range(num_classes):
            class_labels = (labels == class_idx).int()  # Convert to binary: 1 if label matches class, else 0

            fraction_of_positives_before, mean_predicted_value_before = calibration_curve(
                class_labels.cpu().numpy(), probs_before[:, class_idx].detach().cpu().numpy(), n_bins=10
            )
            fraction_of_positives_after, mean_predicted_value_after = calibration_curve(
                class_labels.cpu().numpy(), probs_after[:, class_idx].detach().cpu().numpy(), n_bins=10
            )

            plt.plot(mean_predicted_value_before, fraction_of_positives_before, "s-",
                     label=f"Before Scaling (Class {class_idx})")
            plt.plot(mean_predicted_value_after, fraction_of_positives_after, "s-",
                     label=f"After Scaling (Class {class_idx})")

        plt.legend()
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curve (Per Class)")
        plt.show()

    def test_ensemble(self, data_loader, device, loss_fn, test_history):
        self.to(device)
        with torch.no_grad():
            self.eval()
            for inputs, targets in data_loader:
                inp, tar = inputs.to(device), targets.to(device)

                outp = self(inp)
                loss = loss_fn(outp, tar)

                y_pred = self.activation_function(outp)
                test_history.batch_stats(y_pred=y_pred, y_true=tar, loss=loss)
            test_history.on_epoch_end()

    def get_subject_predictions(self, data_loader, device):

        subject_keys = []
        logits = []
        targets = []

        with torch.no_grad():
            self.eval()
            for inp, tar, subject_indices in data_loader:
                inp, tar = inp.to(device), tar.to(device)

                subject_keys.extend(data_loader.dataset.get_subject_keys_from_indices(subject_indices))

                logit = self(inp, apply_mean=False)
                logits.append(logit)
                targets.extend(tar)

        logits = torch.cat(logits, dim=1).cpu().numpy()

        # Reshape so that it is epochs, ensembles, classes
        reshaped_logits = np.transpose(logits, (1, 0, 2))
        targets = torch.stack(targets).cpu().numpy()

        subject_pred_dict = defaultdict(list)
        subject_labels = {}

        # Iterate over the reshaped logits and targets to get the predictions for each subject
        for sub_key, pred, label in zip(subject_keys, reshaped_logits, targets):
            subject_pred_dict[sub_key].append(pred)
            if sub_key not in subject_labels:
                subject_labels[sub_key] = label

        return subject_pred_dict, subject_labels

    @staticmethod
    def get_subject_prediction(value, merge_method, merge_logits):
        # Generate docstring
        """
        Get the prediction for a single subject. This method is used to get the prediction for a single subject
        from the ensemble model. It can either merge the logits or the probabilities of the ensemble members.

        Parameters
        ----------
        value: np.array
            The logits or probabilities of the ensemble members.
            The shape of the array should be (epochs, ensembles, classes).
        merge_method: str
            The method to merge the logits or probabilities. Avg or first
        merge_logits: bool
            Whether to merge the logits or the probabilities.

        Returns
        -------

        """
        if merge_method == "avg":
            if merge_logits:
                outp = np.mean(value, axis=0)
                ensemble_predictions = torch.softmax(torch.tensor(outp), dim=0).numpy()
                outp = np.mean(outp, axis=0)  # then average across ensembles
                outp_softmax = torch.softmax(torch.tensor(outp), dim=0).numpy()
            else:
                # **softmax before averaging**
                outp_softmax = torch.softmax(torch.tensor(value), dim=2).numpy()
                outp_softmax = np.mean(outp_softmax, axis=0)
                ensemble_predictions = outp_softmax
                outp_softmax = np.mean(outp_softmax, axis=0)  # then average across ensembles
        else:
            # Simply select the first epoch
            if merge_logits:
                outp = value[0]
                ensemble_predictions = torch.softmax(torch.tensor(outp), dim=0).numpy()
                outp = np.mean(outp, axis=0)  # then average across ensembles
                outp_softmax = torch.softmax(torch.tensor(outp), dim=0).numpy()
            else:
                # **softmax before averaging**
                outp_softmax = torch.softmax(torch.tensor(value[0]), dim=1).numpy()
                ensemble_predictions = outp_softmax
                outp_softmax = np.mean(outp_softmax, axis=0)  # then average across ensembles

        ensemble_classes = np.argmax(ensemble_predictions, axis=1)

        return outp_softmax, np.argmax(outp_softmax), ensemble_predictions, ensemble_classes

    def merge_predictions_and_calculate_metrics(self, subject_pred_dict, subject_labels,
                                                merge_logits, merge_method="avg"):
        """
        Merges ensemble model predictions per subject and computes metrics.

        Args:
            subject_pred_dict (dict): Dictionary {subject_id: list of logits}
                                      Shape = (num_epochs, num_ensembles, num_classes).
            subject_labels (dict): Dictionary {subject_id: one-hot encoded labels}.
            merge_logits (bool): If True, average logits before softmax; else, apply softmax first.
            merge_method (str): Method to merge predictions across ensemble models ("avg" or others).

        Returns:
            dict: Aggregated predictions, class labels, and raw data for analysis.
        """

        # Storage for different types of predictions
        raw_logits_per_subject = []
        all_epochwise_probabilities = []

        all_ensemble_probabilities = []
        all_ensemble_class_predictions = []

        subject_one_hot_labels = []
        subject_class_labels = []

        final_subject_probabilities = []
        final_subject_class_predictions = []

        for subject_id, logits_per_epoch in subject_pred_dict.items():
            # Store subject labels
            subject_one_hot_labels.append(subject_labels[subject_id])
            subject_class_labels.append(np.argmax(subject_labels[subject_id]))

            # Convert logits to NumPy
            logits_per_epoch = np.array(logits_per_epoch)

            # Compute subject-level predictions
            final_prob, final_class, ensemble_probabilities, ensemble_class_predictions = (
                self.get_subject_prediction(value=logits_per_epoch,
                                            merge_logits=merge_logits,
                                            merge_method=merge_method))

            #######################################################################################################
            # Save Subject-Level Predictions (Averaged Ensemble)
            #######################################################################################################
            final_subject_probabilities.append(final_prob)
            final_subject_class_predictions.append(final_class)
            #######################################################################################################

            #######################################################################################################
            # Save All Ensemble Model Predictions (Before Merging)
            #######################################################################################################
            all_ensemble_probabilities.append(ensemble_probabilities)
            all_ensemble_class_predictions.append(ensemble_class_predictions)
            #######################################################################################################

            #######################################################################################################
            # Save Raw Data (for Debugging/Analysis)
            #######################################################################################################
            raw_logits_per_subject.append(logits_per_epoch)
            all_epochwise_probabilities.append(torch.softmax(torch.tensor(logits_per_epoch), dim=2).numpy())

        # Convert lists to NumPy arrays
        raw_logits_per_subject = np.array(raw_logits_per_subject)
        all_epochwise_probabilities = np.array(all_epochwise_probabilities)
        subject_one_hot_labels = np.array(subject_one_hot_labels)
        subject_class_labels = np.array(subject_class_labels)
        final_subject_probabilities = np.array(final_subject_probabilities)
        final_subject_class_predictions = np.array(final_subject_class_predictions)
        all_ensemble_probabilities = np.array(all_ensemble_probabilities)
        all_ensemble_class_predictions = np.array(all_ensemble_class_predictions)

        # Create structured dictionary for outputs
        prediction_dict = {
            "raw_logits_per_subject": raw_logits_per_subject,
            "all_epochwise_probabilities": all_epochwise_probabilities,
            "subject_one_hot_labels": subject_one_hot_labels,
            "subject_class_labels": subject_class_labels,
            "final_subject_probabilities": final_subject_probabilities,
            "final_subject_class_predictions": final_subject_class_predictions,
            "all_ensemble_probabilities": all_ensemble_probabilities,
            "all_ensemble_class_predictions": all_ensemble_class_predictions,
            "subject_ids": list(subject_labels.keys())  # Maintain mapping of subject IDs
        }

        performance = calculate_performance_metrics(y_pred_prob=final_subject_probabilities,
                                                    y_pred_class=final_subject_class_predictions,
                                                    y_true_one_hot=subject_one_hot_labels,
                                                    y_true_class=subject_class_labels)

        uncertainty = get_uncertainty_metrics(probs=final_subject_probabilities,
                                              targets=subject_one_hot_labels)

        class_wise_uncertainty = compute_classwise_uncertainty(mean_probs=final_subject_probabilities,
                                                               one_hot_target=subject_one_hot_labels,
                                                               targets=subject_class_labels)

        # Save results to dictionary
        result_dict = {"performance": performance,
                       "uncertainty": uncertainty,
                       "class_uncertainty": class_wise_uncertainty,
                       "predictions": prediction_dict}

        return result_dict

    def ensemble_performance_and_uncertainty(self, data_loader, device, save_path, save_name, save_to_pickle=False,
                                                 save_to_mlflow=False):
        self.to(device)

        # todo: Ensemble test_ensemble method does not work with the new data loader... fix this

        final_prediction_dict = {}

        subject_pred_dict, subject_labels = self.get_subject_predictions(data_loader, device)
        average_logits = self.merge_predictions_and_calculate_metrics(subject_pred_dict=subject_pred_dict,
                                                                      subject_labels=subject_labels,
                                                                      merge_logits=True)
        average_softmax = self.merge_predictions_and_calculate_metrics(subject_pred_dict=subject_pred_dict,
                                                                       subject_labels=subject_labels,
                                                                       merge_logits=False)
        first_logits = self.merge_predictions_and_calculate_metrics(subject_pred_dict=subject_pred_dict,
                                                                    subject_labels=subject_labels,
                                                                    merge_logits=True, merge_method="first")
        first_softmax = self.merge_predictions_and_calculate_metrics(subject_pred_dict=subject_pred_dict,
                                                                     subject_labels=subject_labels,
                                                                     merge_logits=False, merge_method="first")
        final_prediction_dict["average_epochs_merge_logits"] = average_logits
        final_prediction_dict["average_epochs_merge_softmax"] = average_softmax
        final_prediction_dict["first_epoch_merge_logits"] = first_logits
        final_prediction_dict["first_epoch_merge_softmax"] = first_softmax

        if save_to_pickle:
            save_dict_to_pickle(final_prediction_dict, save_path, save_name)

        if save_to_mlflow:
            res_keys = ['average_epochs_merge_logits',
                        'average_epochs_merge_softmax',
                        'first_epoch_merge_logits',
                        'first_epoch_merge_softmax']

            for r in res_keys:
                res_dict = final_prediction_dict[r]
                for k, v in res_dict.items():
                    if k in ("performance", "uncertainty"):
                        for key, value in v.items():
                            if key == "confusion_matrix":
                                continue
                            mlflow.log_metric(f"{save_name}_{r}_{key}", value)

        return final_prediction_dict

    @staticmethod
    def activation_function(logits, ret_prob=True):
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        outp = torch.softmax(logits, dim=1)
        if not ret_prob:
            _, outp = torch.max(outp, dim=1)
        return outp
