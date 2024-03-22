import abc
from typing import Any, Dict

import torch.nn as nn
import torch


class BaseClassifier(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self._hyperparameters = kwargs
        self._name = kwargs.get("name")

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        return self._hyperparameters

    # ----------------
    # Save and load
    # ----------------
    def save(self, path: str) -> None:
        """
        Method for saving
        Args:
            path: Path to save object to

        Returns: Nothing

        """
        # Get state (everything needed to load the model)
        state = {"state_dict": self.state_dict(), "classifier_name": self._name,
                 "hyperparameters": self.hyperparameters}

        # Save
        torch.save(state, f"{path}")

