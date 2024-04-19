import numpy as np
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.utils.data import DataLoader, TensorDataset

from eegDlUncertainty.models.classifiers.main_classifier import MainClassifier


class SimpleGPCModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPClassifier:
    def __init__(self):
        pass

    def forward(self):
        pass

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Define parameters for synthetic data generation
    num_samples = 1000
    num_features = 10  # Number of EEG features
    num_classes = 2  # Number of classes (binary classification)

    # Generate random EEG features
    X = np.random.randn(num_samples, num_features)

    # Generate random labels (0 or 1)
    y = np.random.randint(0, num_classes, size=num_samples)
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X, dtype=torch.float32)
    y_train_tensor = torch.tensor(y, dtype=torch.float32)

    X_train_tensor = X_train_tensor.to(device="cuda")
    y_train_tensor = y_train_tensor.to(device="cuda")

    # Initialize likelihood and model
    likelihood = GaussianLikelihood()
    model = SimpleGPCModel(X_train_tensor, y_train_tensor, likelihood)

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    likelihood = likelihood.to(device)

    # Define the optimizer and the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # Training loop
    model.train()
    for i in range(100):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = -mll(output, y_train_tensor)
        loss.backward()
        optimizer.step()

