import torch
import torch.nn as nn


class Perceptron(nn.Module):
    """ A perceptron is one linear layer """
    def __init__(self, input_dim):
        """ Args:
            input_dim (int): size of the input features
        """
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x_in):
        """The forward pass of the perceptron
        Args:
        x_in (torch.Tensor): an input data tensor
        x_in.shape should be (batch, num_features)
        Returns:
        the resulting tensor.
        tensor.shape should be (batch,). """
        return torch.sigmoid(self.fc1(x_in)).squeeze()
