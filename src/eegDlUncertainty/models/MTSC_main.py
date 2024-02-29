import torch
import torch.nn as nn

from eegDlUncertainty.models.classifiers.brain_decode_classifiers import EEGNetv4MTSC


class MTSClassifier(nn.Module):

    def __init__(self, classifier_name: str, *args, **kwargs):
        super().__init__()
        self._classifier = self._get_classifier(classifier_name, **kwargs)
        self._classifier_name = classifier_name

    @staticmethod
    def _get_classifier(classifier_name: str, **kwargs):
        if classifier_name in ("EEGNetv4", "EEGNetv4MTSC"):
            return EEGNetv4MTSC(**kwargs)

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
    def from_disk(cls, path: str) -> 'MTSClassifier':
        # Get state
        state = torch.load(path)

        # Initialise model
        model = cls(classifier_name=state["classifier_name"], **state["hyperparameters"])

        # Load parameters
        model.load_state_dict(state_dict=state["state_dict"], strict=True)

        return model

    # ---------------------
    # Properties
    # ---------------------
    # @property
    # def name(self) -> str:
    #     return self._name
    #
    # @property
    # def hyperparameters(self) -> Dict[str, Any]:
    #     return self._classifier.hyperparameters
