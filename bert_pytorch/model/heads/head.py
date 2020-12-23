import torch
import torch.nn as nn
from bert_pytorch.model.utils import GELU


class BERTHead(nn.Module):
    def __init__(self, config, token_embeddings, input_size=None):
        super(BERTHead, self).__init__()
        self.vocab_size = config.vocab_size
        self.token_embeddings = token_embeddings
        hidden = config.dim
        if input_size is None:
            input_size = hidden
        if config.head_use_ln:
            self.out = nn.Sequential(
                nn.Linear(input_size, hidden),
                GELU(),
                nn.LayerNorm(hidden)
            )
        else:
            self.out = nn.Sequential(
                nn.Linear(input_size, hidden),
                GELU(),
            )
        self.bias = nn.Parameter(torch.zeros(1, self.vocab_size))

    def forward(self, x, candidates=None):
        x = self.out(x)  # B x H or M x H
        if candidates is not None:  # x : B x H
            emb = self.token_embeddings(candidates)  # B x C x H
            logits = (x.unsqueeze(1) * emb).sum(-1)  # B x C
            bias = self.bias.expand(logits.size(0), -1).gather(1, candidates)  # B x C
            logits += bias
        else:  # x : M x H
            emb = self.token_embeddings.weight  # V x H
            logits = torch.matmul(x, emb.transpose(0, 1))  # M x V
            logits += self.bias
        return logits
