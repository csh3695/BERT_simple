import math
import torch.nn as nn

from bert_pytorch.model.utils import GeLU
from bert_pytorch.model.module.attention import MultiHeadAttention


class TransformerDecoder(nn.Module):
    def __init__(self, dim, dim_ff=2048, head_num=1, dropout=0.1, eps=1e-12):
        super(TransformerDecoder, self).__init__()
        self.attn_1 = MultiHeadAttention(head_num, dim, dropout)
        self.attn_2 = MultiHeadAttention(head_num, dim, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_ff),
            GeLU(),
            nn.Linear(dim_ff, dim)
        )
        self.in_layerNorm = nn.LayerNorm(dim, eps=eps)
        self.mid_layerNorm = nn.LayerNorm(dim, eps=eps)
        self.out_layerNorm = nn.LayerNorm(dim, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, src_mask, tgt_mask):
        x = self.in_layerNorm(x + self.dropout(self.attn_1(x, x, x, mask=src_mask)))
        x = self.mid_layerNorm(x + self.dropout(self.attn_2(x, hidden, hidden, mask=tgt_mask)))
        x = self.out_layerNorm(x + self.dropout(self.ffn(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, dim, dim_ff, num_layers=6, head_num=1, dropout=0.1, eps=1e-12):
        super(Decoder, self).__init__()
        self.dim = dim
        self.proc = nn.ModuleList([TransformerDecoder(dim, dim_ff, head_num, dropout, eps)] * num_layers)

    def forward(self, x, hidden, src_mask, tgt_mask):
        for module in self.proc:
            x = module(x, hidden, src_mask, tgt_mask)
        return x
