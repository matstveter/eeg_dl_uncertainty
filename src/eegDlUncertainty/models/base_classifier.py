import abc
from typing import Any, Dict

import torch.nn as nn


class BaseClassifier(abc.ABC, nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self._hyperparameters = kwargs

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        return self._hyperparameters
