import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroSample, PyroModule


class SimpleBayesianCnn(nn.Module):
    def __init__(self, **kwargs):
        in_channels: int = kwargs.pop("in_channels")
        num_classes: int = kwargs.pop("num_classes")

        super(SimpleBayesianCnn, self).__init__()
        self.conv1 = PyroModule[nn.Conv1d](in_channels=in_channels, out_channels=32, kernel_size=30, padding=2)
        self.conv1.weight = PyroSample(dist.Normal(0., 1.).expand(self.conv1.weight.shape).to_event(4))
        self.conv1.bias = PyroSample(dist.Normal(0., 10.).expand(self.conv1.bias.shape).to_event(1))
        
        self.conv2 = PyroModule[nn.Conv1d](in_channels=32, out_channels=32, kernel_size=15, padding=2)
        self.conv2.weight = PyroSample(dist.Normal(0., 1.).expand(self.conv2.weight.shape).to_event(4))
        self.conv2.bias = PyroSample(dist.Normal(0., 10.).expand(self.conv2.bias.shape).to_event(1))
        
        self.conv3 = PyroModule[nn.Conv1d](in_channels=32, out_channels=10, kernel_size=10, padding=2)
        self.conv3.weight = PyroSample(dist.Normal(0., 1.).expand(self.conv3.weight.shape).to_event(4))
        self.conv3.bias = PyroSample(dist.Normal(0., 10.).expand(self.conv3.bias.shape).to_event(1))
        
        self.fc = PyroModule[nn.Linear](10 * 20, num_classes)  # Adjust the input size based on your data
        self.fc.weight = PyroSample(dist.Normal(0., 1.).expand(self.fc.weight.shape).to_event(2))
        self.fc.bias = PyroSample(dist.Normal(0., 10.).expand(self.fc.bias.shape).to_event(1))
