import torch
import matplotlib.pyplot as plt
x = torch.arange(-10., 10., 0.1)
relu = torch.nn.ReLU()
# y = torch.sigmoid(x)
# y = torch.tanh(x)
y = relu(x)
plt.plot(x.numpy(), y.numpy())
plt.show()
