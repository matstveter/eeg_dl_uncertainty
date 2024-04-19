import os

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from eegDlUncertainty.data.results.history import History, MCHistory
from eegDlUncertainty.models.classifiers.main_classifier import MainClassifier
import mlflow
from torchcontrib.optim import SWA


class AgeClassifier(MainClassifier):
    
    def forward_age(self, x: torch.Tensor, age, **kwargs):
        """ Forward function calling the self.classifiers forward function, in addition in divides the produced logits
        by the self.temperature value. If this value is not optimized it will only be 1.


        Returns
        -------
        temperature scaled predictions if self.temperature is trained

        """
        logits = self.classifier(input_tensor=x, age=age, **kwargs)
        return logits / self.temperature
    
    def fit_model(self, *, train_loader: DataLoader, val_loader: DataLoader, training_epochs: int,
                  device: torch.device, loss_fn: _Loss, train_hist: History, val_history: History,
                  earlystopping_patience: int):
        
        best_loss = 1_000_000
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self._learning_rate)
        self.to(device)

        # For earlystopping
        epochs_with_no_improvement = 0
        early_stop = False
        for epoch in range(training_epochs):
            if early_stop:
                break

            print(f"\n-------------------------  EPOCH {epoch + 1} / {training_epochs}  -------------------------")
            self.train()

            for data, ages, targets in train_loader:
                inputs, age, targets = data.to(device), ages.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = self.forward_age(inputs, age)
                loss = loss_fn(outputs, targets)
                # Store values in the history object
                y_pred = self.activation_function(logits=outputs)
                train_hist.batch_stats(y_pred=y_pred, y_true=targets, loss=loss)

                loss.backward()
                optimizer.step()

            train_hist.on_epoch_end()

            self.eval()
            with torch.no_grad():
                for inputs, ages, targets in val_loader:
                    inputs, age, targets = inputs.to(device), ages.to(device), targets.to(device)
                    outputs = self.forward_age(inputs, age)

                    val_loss = loss_fn(outputs, targets)

                    # Activation function and store values
                    y_pred = self.activation_function(logits=outputs)
                    val_history.batch_stats(y_pred=y_pred, y_true=targets, loss=val_loss)
                val_history.on_epoch_end()

            if val_history.get_last_loss() < best_loss:
                best_loss = val_history.get_last_loss()
                path = os.path.join(self._model_path, f"{self._name}_model")
                self.classifier.save(path=path)
                epochs_with_no_improvement = 0
            else:
                epochs_with_no_improvement += 1

            if 0 < earlystopping_patience <= epochs_with_no_improvement:
                early_stop = True
                mlflow.log_metric('earlystopping_inferred', (epoch + 1))
                print(f"\nStopping early at epoch {epoch + 1}!")

        path = os.path.join(self._model_path, f"{self._name}_last_model")
        self.classifier.save(path=path)
        self.save_hyperparameters()
    
    def test_model(self, *, test_loader: DataLoader, device: torch.device, test_hist: History, loss_fn: _Loss):
        self.to(device)
        with torch.no_grad():
            self.eval()
            for inputs, ages, targets in test_loader:
                inputs, age, target = inputs.to(device), ages.to(device), targets.to(device)

                outputs = self.forward_age(inputs, age)
                loss = loss_fn(outputs, target)

                y_pred = self.activation_function(outputs)
                test_hist.batch_stats(y_pred=y_pred, y_true=target, loss=loss)
            test_hist.on_epoch_end()

    def get_mc_predictions(self, *, test_loader: DataLoader, device: torch.device, history: MCHistory, num_forward=50):
        """
        Generate Monte Carlo predictions from the model by enabling dropout during test time.
        This is often used to obtain the predictive uncertainty estimates from models like Bayesian Neural Networks.

        Parameters
        ----------
        history: MCHistory
        test_loader : DataLoader
            The DataLoader provides batches of test data.
        device : torch.device
            The device (e.g., 'cuda' or 'cpu') the model and data should be moved to for computation.
        num_forward : int, optional
            The number of forward passes to perform with dropout enabled. Defaults to 50.

        Notes
        -----
        This function modifies the model in-place by setting it to evaluation mode and manually turning on
        the training mode for any dropout layers found in the model. This enables stochastic behaviors, such
        as dropout during the forward passes, which is crucial for generating multiple predictive outcomes.

        The function collects all predictions and targets from the test loader, performs the specified number
        of forward passes, and calculates metrics based on these predictions. Each forward pass can potentially
        lead to different predictions due to the randomness introduced by the dropout layers.

        After predictions, the model is used to compute metrics which are then printed. This function does not
        return any values directly, but rather outputs through side effects (printing).

        Examples
        --------
        >>> model = MyModel()
        >>> t_loader = DataLoader(my_dataset, batch_size=10)
        >>> devi = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> model.get_mc_predictions(test_loader=t_loader, device=devi, num_forward=100)
        """
        self.to(device)
        with torch.no_grad():
            self.eval()
            # Turn on the training mode for modules that is of instance nn.Dropout
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.train()

            for _ in range(num_forward):
                predictions = []
                targets = []
                for inputs, ages, targets_batch in test_loader:
                    inputs, age, target_batch = inputs.to(device), ages.to(device), targets_batch.to(device)

                    outputs = self.forward_age(inputs, age)
                    y_pred = self.activation_function(outputs)

                    predictions.extend(y_pred.cpu().numpy())
                    targets.extend(target_batch.cpu().numpy())

                history.on_pass_end(predictions=predictions, labels=targets)

        history.calculate_metrics()

    def fit_swa(self, *, train_loader: DataLoader, val_loader: DataLoader, training_epochs: int,
                  device: torch.device, loss_fn: _Loss, train_hist: History, val_history: History,
                swa_start, swa_freq, swa_lr):
        pass