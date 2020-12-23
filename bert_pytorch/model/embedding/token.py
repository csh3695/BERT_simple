import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        super(TokenEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input: torch.tensor) -> torch.tensor:
        return super(TokenEmbedding, self).forward(input)