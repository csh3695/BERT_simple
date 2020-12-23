import math

import torch
import torch.nn as nn


def scaledDotProdAttention(q: torch.tensor, k: torch.tensor, v: torch.tensor, mask=None, dropout=None):
    """

    :param q: n x dk tensor
    :param k: n x dk tensor
    :param v: n x dv tensor
    :return: n x dv tensor
    """
    scaling_factor = math.sqrt(k.shape[-1])
    score = torch.matmul(q, torch.transpose(k, -1, -2)) / scaling_factor
    if mask is not None:
        mask = mask.unsqueeze(1)
        score = score.masked_fill(mask == 0, -1e9)
    score = torch.softmax(score, dim=-1)
    if dropout is not None:
        score = dropout(score)

    return torch.matmul(score, v), score


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        if config.dim % config.num_heads != 0:
            raise Exception("dim should be divided by h")
        self.h_w = int(config.dim / config.num_heads)
        self.h_n = config.num_heads
        self.q_linear = nn.Linear(config.dim, config.dim)
        self.k_linear = nn.Linear(config.dim, config.dim)
        self.v_linear = nn.Linear(config.dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.out_linear = nn.Linear(config.dim, config.dim)

    def forward(self, q, k, v, mask=None, log=None):
        bs = q.size(0)
        qs = q.size(1)
        q = self.q_linear(q).reshape(bs, q.size(1), self.h_n, self.h_w).transpose(-2, -3)
        k = self.k_linear(k).reshape(bs, k.size(1), self.h_n, self.h_w).transpose(-2, -3)
        v = self.v_linear(v).reshape(bs, v.size(1), self.h_n, self.h_w).transpose(-2, -3)
        out, score = scaledDotProdAttention(q, k, v, mask=mask, dropout=self.dropout)
        out = out.transpose(-2, -3).reshape(bs, qs, -1)
        if log is not None:
            if 'attn_score' not in log.keys():
                log['attn_score'] = []
            log['attn_score'].append(score.clone().detach())
        return self.out_linear(out)
