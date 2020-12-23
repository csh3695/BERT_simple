import numpy as np
import torch
import torch.nn as nn


class GeLU(nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x.pow(3))))
