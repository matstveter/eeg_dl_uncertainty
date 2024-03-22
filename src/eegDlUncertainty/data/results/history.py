import os.path
import pickle
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, f1_score, cohen_kappa_score, precision_score, recall_score,
                             matthews_corrcoef, confusion_matrix)


class History:

    def __init__(self, num_classes: int, set_name: str, loader_lenght, save_path, verbose=True):
        self._loss: List[float] = []
        self._accuracy: List[float] = []
        self._precision: List[float] = []
        self._recall: List[float] = []
        self._f1: List[float] = []
        self._auc: List[float] = []
        self._kappa: List[float] = []
        self._mcc: List[float] = []
        self._conf_mat: List[float] = []
        self._num_classes = num_classes

        self.epoch_y_true: List[torch.Tensor] = []
        self.epoch_y_pred: List[torch.Tensor] = []
        self.epoch_loss: int = 0
        self.verbose: bool = verbose
        self._set_name: str = set_name
        self._loader_lenght: int = loader_lenght
        self._save_path = save_path

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def get_last_loss(self) -> float:
        return self._loss[-1]

    def get_last_auc(self) -> float:
        return self._auc[-1]

    def get_last_acc(self) -> float:
        return self._accuracy[-1]

    def _update_metrics(self) -> None:
        self._loss.append(self.epoch_loss / self._loader_lenght)
        y_pred = torch.tensor(self.epoch_y_pred)
        y_true = torch.tensor(self.epoch_y_true)

        if self._num_classes == 1:
            y_pred_proba = y_pred.detach().cpu().numpy()
            y_pred = torch.round(y_pred)
            y_pred = y_pred.detach().cpu().numpy()
            y_true = y_true.detach().cpu().numpy()
            self._precision.append(precision_score(y_true=y_true, y_pred=y_pred, zero_division=0))
            self._recall.append(recall_score(y_true=y_true, y_pred=y_pred))
            self._f1.append(f1_score(y_true=y_true, y_pred=y_pred))
            self._auc.append(roc_auc_score(y_true=y_true, y_score=y_pred_proba))
        else:
            y_pred_proba = y_pred.detach().cpu().numpy()
            y_true_one_hot = y_true.detach().cpu().numpy()
            _, y_pred = torch.max(y_pred, dim=1)
            _, y_true = torch.max(y_true, dim=1)
            y_pred = y_pred.detach().cpu().numpy()
            y_true = y_true.detach().cpu().numpy()
            self._precision.append(precision_score(y_true=y_true, y_pred=y_pred, average="weighted",
                                                   zero_division=0))
            self._recall.append(recall_score(y_true=y_true, y_pred=y_pred, average="weighted"))
            self._f1.append(f1_score(y_true=y_true, y_pred=y_pred, average="weighted"))
            self._auc.append(
                roc_auc_score(y_true=y_true_one_hot, y_score=y_pred_proba, multi_class="ovr", average="weighted"))

        self._kappa.append(cohen_kappa_score(y1=y_true, y2=y_pred))
        self._mcc.append(matthews_corrcoef(y_true=y_true, y_pred=y_pred))

        if self._set_name.lower() == "val":
            self._conf_mat.append(confusion_matrix(y_pred=y_pred, y_true=y_true))

        self._accuracy.append(self._calculate_accuracy(y_pred=y_pred, y_true=y_true))

    def print_metrics(self) -> None:
        if self._set_name == "test":
            print("\n\n")
        print(f"{self._set_name.upper()}: Loss: {self._loss[-1]:.4f}"
              f"  Accuracy: {self._accuracy[-1]:.2f}"
              f"  AUC: {self._auc[-1]:.2f}"
              f"  Precision: {self._precision[-1]:.2f}"
              f"  Recall: {self._recall[-1]:.2f}"
              f"  F1: {self._f1[-1]:.2f}"
              f"  MCC: {self._mcc[-1]:.2f}  ", end="")

    def on_epoch_end(self) -> None:
        self._update_metrics()
        if self.verbose:
            self.print_metrics()

        if self._set_name != "test":
            self.epoch_y_true = []
            self.epoch_y_pred = []
            self.epoch_loss = 0

    def batch_stats(self, y_pred, y_true, loss) -> None:
        # Extend epoch_y_pred and epoch_y_true based on label representation
        if self._num_classes == 1:  # Binary classification
            self.epoch_y_pred.extend(y_pred)
            self.epoch_y_true.extend(y_true)
        else:  # Multiclass classification
            self.epoch_y_pred.extend(y_pred.tolist())  # Convert predicted probabilities to list
            self.epoch_y_true.extend(y_true.tolist())  # Convert one-hot encoded

        self.epoch_loss += loss.item()

    @staticmethod
    def _calculate_accuracy(y_pred, y_true) -> float:
        correct_percentage = ((y_pred == y_true).sum() / len(y_true)) * 100
        return correct_percentage

    def get_as_dict(self) -> Dict[str, Any]:
        return {
            "loss": self._loss,
            "accuracy": self._accuracy,
            "precision": self._precision,
            "recall": self._recall,
            "f1": self._f1,
            "auc": self._auc,
            "kappa": self._kappa,
            "mcc": self._mcc,
            "conf_mat": self._conf_mat,
            "num_classes": self._num_classes,
            "set_name": self._set_name,
            "loader_length": self._loader_lenght
        }

    def save_to_pickle(self):
        data_path = os.path.join(self._save_path, "data")

        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)

        full_file_path = os.path.join(data_path, f"{self._set_name.upper()}_data_dict.pkl")
        data_dict = self.get_as_dict()

        with open(full_file_path, 'wb') as file:
            pickle.dump(data_dict, file)

    def get_predictions(self):
        if self._num_classes == 1:
            y_pred_numpy = np.array([tensor.cpu().numpy() for tensor in self.epoch_y_pred])
            y_true_numpy = np.array([tensor.cpu().numpy() for tensor in self.epoch_y_true])
        else:
            y_pred_numpy = np.array(self.epoch_y_pred)
            y_true_numpy = np.array(self.epoch_y_true)

        return y_pred_numpy, y_true_numpy
