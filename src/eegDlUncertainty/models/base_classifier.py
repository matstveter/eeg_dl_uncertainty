import abc
from typing import Any, Dict

import torch.nn as nn
import torch


class BaseClassifier(abc.ABC, nn.Module):

    def __init__(self, name, **kwargs):
        super().__init__()

        self._hyperparameters = kwargs
        self._name = name

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

    @classmethod
    def from_disk(cls, path: str):
        # Get state
        state = torch.load(path)

        # Initialise model
        model = cls(classifier_name=state["classifier_name"], **state["hyperparameters"])

        # Load parameters
        model.load_state_dict(state_dict=state["state_dict"], strict=True)

        return model
