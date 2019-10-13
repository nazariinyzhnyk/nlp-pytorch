import torch.nn.functional as F
import torch.nn as nn


class MultilayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        :param input_dim: (int) input vector size
        :param hidden_dim: (int) hidden layer size
        :param output_dim: (int) output layer size
        """

        super(MultilayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in, apply_softmax=False):
        """
        Forward pass of MLP
        :param x_in: (torch.Tensor) input tensor of shape (batch_size, input_dim)
        :param apply_softmax: (bool) activation flag
        :return: resulting tensor of shape (batch, output_dim)
        """
        hid = F.relu(self.fc1(x_in))
        output = self.fc2(hid)

        if apply_softmax:
            output = F.softmax(output, dim=1)

        return output
