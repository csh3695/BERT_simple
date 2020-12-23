import torch
import torch.nn as nn

from bert_pytorch.model.utils import GELU
from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.attn = MultiHeadAttention(config)
        self.ffn = PositionwiseFeedForward(config)
        self.in_layerNorm = nn.LayerNorm(config.dim, eps=config.eps)
        self.out_layerNorm = nn.LayerNorm(config.dim, eps=config.eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None, log=None):
        x = self.in_layerNorm(x + self.dropout(self.attn(x, x, x, mask=mask, log=log)))
        x = self.out_layerNorm(x + self.dropout(self.ffn(x)))
        return x
