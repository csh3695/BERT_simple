import torch
import torch.nn as nn

from bert_pytorch.model.utils import GELU


class PositionwiseFeedForward(nn.Module):
    def __init__(self, config):
        super(PositionwiseFeedForward, self).__init__()
        self.l1 = nn.Linear(config.dim, config.dim_ff)
        self.l2 = nn.Linear(config.dim_ff, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.act = GELU()

    def forward(self, x):
        return self.l2(self.dropout(self.act(self.l1(x))))
