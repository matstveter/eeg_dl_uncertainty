import numpy as np
import torch
from torchmetrics.classification import MulticlassCalibrationError
from sklearn.metrics import roc_auc_score, f1_score,precision_score, recall_score, confusion_matrix, accuracy_score


def ece(probs, targets, bins=10):
    """
    From: https://github.com/centerforaisafety/Intro_to_ML_Safety/blob/master/Interpretable%20Uncertainty/main.md

    Expected calibration error (ECE) is the primary metric for testing calibration
    (Naeini, Cooper, and Hauskrecht 2015). To calculate ECE, we first divide up the interval between 0 and 1 into bins.
    For instance, we might let the bins be [0, 0.1], [0.1, 0.2], … [0.9, 1]. Then we place examples into these bins
    based on the model’s confidence when making the prediction. Often this means taking the max of the model’s
    post-softmax prediction scores. Finally, we take the weighted sum of the absolute difference between the real
    accuracy and the predicted accuracy. The sum is weighted based on the number of examples in each bin.
    Formally, say we have n examples partitioned up into M bins B1, B2, …, BM. Also, let acc(Bm)
    be the average accuracy of examples in the bin and let conf(Bm) be the average confidence of examples in the bin.

    ECE ranges between 0 and 1, with lower scores being better.
    What is considered a strong ECE varies from dataset to dataset. Reading a few papers,
    we get that ImageNet classifiers usually have ECE which varies from 0.01 to 0.08 and a score of 0.02 or
    lower can be considered strong (Guo et al. 2017; Minderer et al. 2021).

    Returns
    -------

    """
    metric = MulticlassCalibrationError(num_classes=3, n_bins=10, norm="l1")

    if targets.shape[1] == 3:
        targets = np.argmax(targets, axis=1)

    return metric(torch.from_numpy(probs), torch.from_numpy(targets))


def mce(probs, targets, bins=10):
    """
    From: https://github.com/centerforaisafety/Intro_to_ML_Safety/blob/master/Interpretable%20Uncertainty/main.md
    
    The Maximum Calibration Error (MCE) is similar to ECE but meant for much more sensitive domains
    (Naeini, Cooper, and Hauskrecht 2015). Like ECE, we partition the interval up into bins. However,
    instead of taking a weighted average of calibration score over bins, we take the maximum calibration
    error over bins. In other words MCE aims to reduce the calibration error of the worst bin, with the
    intuition that this prevents catastrophic failure cases while giving up some efficacy on more mundane cases.

    $$MCE = \max_{m ∈ {1, ..., M}}|acc(B_m) − conf(B_m)|$$

    Like ECE, MCE ranges between 0 and 1, with lower scores being better. MCE is much less common than ECE.
    Quickly eyeballing some results gives us that a model with an MCE of 0.1 can be considered strong (Guo et al. 2017).

    Returns
    -------

    """
    metric = MulticlassCalibrationError(num_classes=3, n_bins=bins, norm="max")
    if targets.shape[1] == 3:
        targets = np.argmax(targets, axis=1)

    return metric(torch.from_numpy(probs), torch.from_numpy(targets))


def nll(probs, targets):
    """
    From: https://github.com/centerforaisafety/Intro_to_ML_Safety/blob/master/Interpretable%20Uncertainty/main.md
    
    The likelihood of a dataset is the probability that a model assigns to the entire dataset. It is defined as follows:

    $$Likelihood = \prod_{x, y ∼ \mathcal{D}}p(y|x)$$

    for p(y|x) our classifier. For numerical stability reasons, it’s common practice to take the negative
    log likelihood (NLL) defined as follows:

    $$NLL =  − \sum_{x, y ∼ \mathcal{D}}log p(y|x)$$

    Negative log likelihood (or cross-entropy loss) is commonly used for maximizing predictive accuracy.
    However, NLL is also useful for calibration as well; a classic result in statistics shows that NLL is
    minimized precisely when p(y|x) matches the true probability distribution π(y|x)
    (Hastie, Tibshirani, and Friedman 2009). In other words, NLL is minimized at zero when the
    classifier is perfectly calibrated. In addition, a poor classifier can have unbounded NLL.

    Returns
    -------

    """
    # Make sure that the probs are not 0, to avoid log(0)
    probs = np.clip(probs, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(targets * np.log(probs), axis=1))


def brier_score(probs, targets):
    """
    From: https://github.com/centerforaisafety/Intro_to_ML_Safety/blob/master/Interpretable%20Uncertainty/main.md
    
    Finally, brier score is a common way to measure the accuracy of probability estimates,
    historically used in measuring forecasting accuracy (Brier 1950).
    It is equivalent to measuring the mean squared error of the probability, as follows.

    Brier score is used in many real-world applications, such as assessing weather, sports, or political predictions.
    Brier score is a “strictly proper scoring rule,” meaning that one can uniquely maximize one’s score by predicting
    the true probabilities. Brier score ranges between 0 and 1, with an optimal model having a score of 0.
    
    Returns
    -------
    """
    return np.mean(np.sum((probs - targets) ** 2, axis=1))


def get_uncertainty_metrics(probs, targets):
    return {'Brier': brier_score(probs=probs, targets=targets),
            'NLL': nll(probs=probs, targets=targets),
            'ECE': ece(probs=probs, targets=targets),
            'MCE': mce(probs=probs, targets=targets)}


def calculate_performance_metrics(y_pred_prob, y_pred_class, y_true_one_hot, y_true_class):
    return {'accuracy': accuracy_score(y_true=y_true_class, y_pred=y_pred_class),
            'precision': precision_score(y_true=y_true_class, y_pred=y_pred_class, average="weighted", zero_division=0),
            'recall': recall_score(y_true=y_true_class, y_pred=y_pred_class, average="weighted"),
            'f1': f1_score(y_true=y_true_class, y_pred=y_pred_class, average="weighted"),
            'auc': roc_auc_score(y_true=y_true_one_hot, y_score=y_pred_prob, multi_class="ovr", average="weighted"),
            'confusion_matrix': confusion_matrix(y_true=y_true_class, y_pred=y_pred_class)}