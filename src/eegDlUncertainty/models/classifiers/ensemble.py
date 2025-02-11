from typing import List

import mlflow
import torch

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
        self.temperature.to(self.device)
        if self.method == "ensemble":
            logits = []
            for m in self.classifiers:
                logits.append(m(x))
            logits = torch.stack(logits)
        else:
            logits = self.classifiers.forward_ensemble(x)

        if apply_mean:
            logits = torch.mean(logits, dim=0)
            temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
            return logits / temperature
        else:
            return logits

    def predict(self, x: torch.Tensor):
        return self.activation_function(logits=self.forward(x), ret_prob=False)

    def predict_prob(self, x: torch.Tensor):
        return self.activation_function(logits=self.forward(x), ret_prob=True)

    def set_temperature_scale_ensemble(self, data_loader, device, criterion, patience=250):
        self.eval()
        self.to(device)
        # Setting this to 1.5 based on the original paper
        self.temperature.data.fill_(1.5)
        self.temperature.to(device)

        logits_list = []
        labels_list = []

        with torch.no_grad():
            for data, labels in data_loader:
                data, labels = data.to(device), labels.to(device)
                logits = self(data)
                logits_list.append(logits)
                labels_list.append(labels)
            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.001, max_iter=500)

        def evaluation():
            optimizer.zero_grad()
            temp = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
            loss = criterion(logits / temp, labels)
            loss.backward()
            return loss

        optimizer.step(lambda: evaluation())

        print("Temperature: ", self.temperature.item())
        mlflow.log_metric("optimal_temp_ensemble", self.temperature.item())

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

    def ensemble_performance_and_uncertainty(self, data_loader, device, save_path, save_name, save_to_pickle=False,
                                             save_to_mlflow=False):
        self.to(device)

        logits = []
        probs = []
        predictions = []
        all_predictions = []
        target_one_hot: List[torch.Tensor] = []
        target_class: List[torch.Tensor] = []
        with torch.no_grad():
            self.eval()
            for inputs, tar in data_loader:
                inp, tar = inputs.to(device), tar.to(device)

                outp = self(inp)
                logits.extend(outp)
                probs.extend(self.activation_function(outp, ret_prob=True))
                predictions.extend(self.activation_function(outp, ret_prob=False))
                target_one_hot.extend(tar)
                target_class.extend(torch.argmax(tar, dim=1))

                # Here we want to save the predictions of each classifier in the ensemble for the uncertainty analysis
                # We need to save the probabilities of each class for each sample without temperature scaling
                all_ensemble_pred = self.activation_function(self(inp, apply_mean=False), ret_prob=True)
                all_predictions.append(all_ensemble_pred)

        # Convert all to numpy arrays
        logits = torch.stack(logits).cpu().numpy()
        probs = torch.stack(probs).cpu().numpy()
        predictions = torch.stack(predictions).cpu().numpy()
        target_one_hot = torch.stack(target_one_hot).cpu().numpy()
        target_class = torch.stack(target_class).cpu().numpy()
        all_predictions = torch.cat(all_predictions, dim=1).cpu().numpy()

        prediction_dict = {"logits": logits, "probs": probs, "predictions": predictions,
                           "target_one_hot": target_one_hot, "target_classes": target_class,
                           "all_predictions": all_predictions}

        performance = calculate_performance_metrics(y_pred_prob=probs, y_pred_class=predictions,
                                                    y_true_one_hot=target_one_hot, y_true_class=target_class)
        uncertainty = get_uncertainty_metrics(probs=probs, targets=target_one_hot)

        class_uncertainty = compute_classwise_uncertainty(all_probs=all_predictions, mean_probs=probs,
                                                          one_hot_target=target_one_hot, targets=target_class)
        results = {"performance": performance, "uncertainty": uncertainty, "class_uncertainty": class_uncertainty,
                   "predictions": prediction_dict}

        if save_to_pickle:
            save_dict_to_pickle(results, save_path, save_name)

        if save_to_mlflow:
            for k, v in results.items():
                if k in ("performance", "uncertainty"):
                    for key, value in v.items():
                        if key == "confusion_matrix":
                            continue
                        mlflow.log_metric(f"{save_name}_{key}", value)

        return results

    @staticmethod
    def activation_function(logits, ret_prob=True):
        outp = torch.softmax(logits, dim=1)
        if not ret_prob:
            _, outp = torch.max(outp, dim=1)
        return outp
