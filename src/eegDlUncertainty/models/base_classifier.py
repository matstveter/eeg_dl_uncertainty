from typing import Any, Dict
import torch.nn as nn
import torch


class BaseClassifier(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self._hyperparameters = kwargs
        self._name = kwargs.get("classifier_name")

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
    def load(cls, path: str):
        """
        Method for loading
        Args:
            path: Path to load the object from

        Returns: Loaded model
        """
        # Load the state
        state = torch.load(path)

        # Create an instance of the class with the saved hyperparameters
        model = cls(**state["hyperparameters"])

        # Load the saved state dictionary into the model
        model.load_state_dict(state["state_dict"])

        # Set the classifier name
        model._name = state["classifier_name"]

        return model
