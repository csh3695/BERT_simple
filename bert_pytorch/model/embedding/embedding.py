import math
import torch
import torch.nn as nn

from bert_pytorch.model.embedding.positional import *
from bert_pytorch.model.embedding.token import *
from bert_pytorch.model.embedding.tokentype import *


class BERTEmbedding(nn.Module):
    def __init__(self, config):
        super(BERTEmbedding, self).__init__()

        if config.pe_type in ['learnable', 0]:
            self.positional_embedding = LearnablePositionalEmbedding(config.dim, config.max_len)
        elif config.pe_type in ['sinusoidal', 1]:
            self.positional_embedding = SinusoidalPositionalEmbedding(config.dim, config.max_len)
        else:
            raise Exception("pe_type should be 'learnable'(0) or 'sinusoidal' (1)")

        if config.te_type in ['full', 0]:
            self.token_embedding = TokenEmbedding(config.vocab_size, config.dim, padding_idx=config.pad_symbol)
        elif config.te_type in ['factorized', 1]:
            self.token_embedding = FactorizedTokenEmbedding(config.vocab_size, config.dim, config.embedding_hidden_dim, config.pad_symbol)

        self.token_type_embedding = None
        if config.num_token_type is not None and config.num_token_type > 0:
            self.token_type_embedding = TokenTypeEmbedding(config.num_token_type, config.dim)

        self.layernorm = nn.LayerNorm(config.dim, eps=config.eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, token_type=None):
        x = self.token_embedding(x)
        x += self.positional_embedding(x)
        if token_type is not None:
            x += self.token_type_embedding(token_type)
        return self.dropout(self.layernorm(x))
