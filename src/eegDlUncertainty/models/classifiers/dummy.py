import torch
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from eegDlUncertainty.data.results.history import History


class DummyModel:
    def __init__(self, train_loader):
        self._majority_class = self._get_majority_class(train_loader)

    @staticmethod
    def activation_function(logits, ret_prob=True):
        """
        Applies an activation function to the logits based on their shape.

        This method applies a softmax activation function if the last dimension
        of `logits` is 3, indicating a multi-class classification problem. Otherwise,
        it applies a sigmoid activation function, assuming a binary classification
        problem. It can return either probabilities or class labels based on the
        `ret_prob` flag.

        Parameters
        ----------
        logits : torch.Tensor
            The input tensor containing logits from a model's output. The shape of
            `logits` determines which activation function is applied.
        ret_prob : bool, optional
            A flag determining the type of output. If `True` (default), the method
            returns probabilities. If `False`, it returns class labels. For softmax,
            class labels are the indices of the max probability. For sigmoid, labels
            are obtained by rounding the probabilities.

        Returns
        -------
        torch.Tensor
            The output tensor after applying the activation function. If `ret_prob`
            is `True`, it contains probabilities. If `False`, it contains class labels
            as integers for multi-class or binaries for binary classification problems.

        Notes
        -----
        - For multi-class classification (softmax), the output tensor has the same shape
          as the input if `ret_prob` is `True`. If `ret_prob` is `False`, the output tensor
          shape will have one less dimension, representing the class label with the highest
          probability for each input.
        - For binary classification (sigmoid), the output tensor always has the same shape
          as the input, with each element representing the probability or binary class label.

        """
        if logits.shape[1] == 3:
            outp = torch.softmax(logits, dim=1)
            if not ret_prob:
                _, outp = torch.max(outp, dim=1)
        else:
            outp = torch.sigmoid(logits)
            if not ret_prob:
                outp = torch.round(outp)
        return outp

    def _get_majority_class(self, train_loader):
        class_counts = {}
        for _, tar in train_loader:
            for label in tar:

                label = torch.argmax(label).item()

                if label in class_counts:
                    class_counts[label] += 1
                else:
                    class_counts[label] = 1

        return max(class_counts, key=class_counts.get)

    def test_model(self, *, test_loader: DataLoader, test_hist: History, loss_fn: _Loss):
        for inputs, targets in test_loader:
            # We don't need inputs for the dummy model
            batch_size = targets.size(0)

            # Create a tensor filled with the majority class for the entire batch
            # One-hot encode the majority class for multiclass classification
            outputs = torch.full((batch_size,), self._majority_class, dtype=torch.long)
            outputs_one_hot = torch.nn.functional.one_hot(outputs, num_classes=3).float()

            # Calculate loss
            loss = loss_fn(outputs_one_hot, targets)

            # Predictions are just the majority class for every instance
            y_pred = outputs_one_hot

            # Store stats
            test_hist.batch_stats(y_pred=y_pred, y_true=targets, loss=loss)

            # Finish epoch
        test_hist.on_epoch_end(plot=True)
