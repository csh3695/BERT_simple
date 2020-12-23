import torch
import torch.nn as nn


class TokenTypeEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        super(TokenTypeEmbedding, self).__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return super(TokenTypeEmbedding, self).forward(x)