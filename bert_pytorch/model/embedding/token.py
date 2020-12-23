import torch
import torch.nn as nn
from bert_pytorch.model.utils import GELU


class FactorizedTokenEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, padding_idx):
        super(FactorizedTokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, hidden_dim, padding_idx=padding_idx)
        self.proj = nn.Linear(hidden_dim, embedding_dim)
        self.gelu = GELU()
        self.cls = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, num_embeddings),
        )
        self.cls[-1].weight = self.embedding.weight
        self.cls[-1].bias.data.zero_()

    def backward(self, x: torch.tensor) -> torch.tensor:
        return self.cls(self.gelu(x))

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.proj(self.gelu(self.embedding(x)))


class TokenEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.embedding(x)