import torch
import torch.nn as nn

from bert_pytorch.model.utils import GeLU
from bert_pytorch.model.module.attention import MultiHeadAttention


class TransformerEncoder(nn.Module):
    def __init__(self, dim, dim_ff=2048, head_num=1, dropout=0.1, eps=1e-12):
        super(TransformerEncoder, self).__init__()
        self.attn = MultiHeadAttention(head_num, dim, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_ff),
            GeLU(),
            nn.Linear(dim_ff, dim),
        )
        self.in_layerNorm = nn.LayerNorm(dim, eps=eps)
        self.out_layerNorm = nn.LayerNorm(dim, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, attn=False):
        if attn:
            attn_out, score = self.attn(x, x, x, mask=mask, attn=attn)
            x = self.in_layerNorm(x + self.dropout(attn_out))
        else:
            x = self.in_layerNorm(x + self.dropout(self.attn(x, x, x, mask=mask, attn=attn)))
        x = self.out_layerNorm(x + self.dropout(self.ffn(x)))
        if attn:
            return x, score
        return x


class Encoder(nn.Module):
    def __init__(self, dim, dim_ff, num_layers=6, head_num=1, dropout=0.1, eps=1e-12):
        super(Encoder, self).__init__()
        self.dim = dim
        self.num_layers = num_layers
        #self.proc = nn.ModuleList([TransformerEncoder(dim, dim_ff, head_num, dropout, eps)] * num_layers)
        self.proc = TransformerEncoder(dim, dim_ff, head_num, dropout, eps)

    def forward(self, x, mask=None):
        out_list = []
        for _ in range(self.num_layers):
            x = self.proc(x, mask)
            out_list.append(x.clone())
        #for module in self.proc:
        #    x = module(x, mask)
        return x, torch.stack(out_list, dim=0)
