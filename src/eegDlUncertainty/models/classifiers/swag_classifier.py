import copy
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

from eegDlUncertainty.models.model_utils import mapping_avg_state_dict


class SWAClassifier(nn.Module):
    def __init__(self, pretrained_model, learning_rate, save_path, model_hyperparameters, name):
        super().__init__()

        self.model = copy.deepcopy(pretrained_model)
        self._learning_rate = learning_rate

        self._model_path = os.path.join(save_path, "model")
        self._model_hyperparameters = model_hyperparameters
        self._name = name
    
    def save(self, path: str) -> None:
        """
        Method for saving
        Args:
            path: Path to save object to

        Returns: Nothing

        """
        # Get state (everything needed to load the model)
        state = {"state_dict": self.state_dict(), "classifier_name": self._name,
                 "hyperparameters": self._model_hyperparameters}
        # Save
        torch.save(state, f"{path}")
    
    def forward(self, x: torch.Tensor, **kwargs):
        return self.model(x, **kwargs)

    def fit(self, *, train_loader: DataLoader, val_loader: DataLoader, swa_epochs: int,
            device: torch.device, loss_fn: _Loss, swa_lr):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._learning_rate)

        # Create an averaged model for SWA
        swa_model = AveragedModel(self.model)
        self.to(device)

        # Define the learning rate scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=100)

        # Set SWA start epoch and create SWA learning rate scheduler
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)

        for epoch in range(swa_epochs):
            print(
                f"\n-------------------------  SWA EPOCH {epoch + 1} / {swa_epochs}  -------------------------")

            self.train()
            for data, targets in train_loader:
                inputs, targets = data.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
            scheduler.step()

            swa_model.update_parameters(self.model)
            swa_scheduler.step()

            # Evaluate on validation set
            self.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = swa_model(inputs)

        # Update batch normalization statistics for the SWA model
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        self.model.load_state_dict(mapping_avg_state_dict(averaged_model_state_dict=swa_model.state_dict()))
        path = os.path.join(self._model_path, f"swa_model")
        self.save(path=path)


class SWAGClassifier(nn.Module):
    pass
