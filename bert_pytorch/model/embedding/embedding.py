import math

import torch
import torch.nn as nn

from bert_pytorch.model.utils import GeLU


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=768):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.shape[-2]]


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=768):
        super().__init__()
        self.embedding = nn.Embedding(max_len, dim)
        self.register_buffer('range', torch.arange(max_len))

    def forward(self, x):
        return self.embedding(self.range[:x.size(-2)])


class TokenEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim//4, padding_idx=padding_idx)
        self.proj = nn.Linear(embedding_dim//4, embedding_dim)
        self.gelu = GeLU()
        self.cls = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim//4),
            GeLU(),
            nn.Linear(embedding_dim//4, num_embeddings),
        )
        self.cls[-1].weight = self.embedding.weight
        self.cls[-1].bias.data.zero_()

    def backward(self, input: torch.tensor) -> torch.tensor:
        return self.cls(self.gelu(input))

    def forward(self, input: torch.tensor) -> torch.tensor:
        return self.proj(self.gelu(self.embedding(input)))


class TokenTypeEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        super(TokenTypeEmbedding, self).__init__(num_embeddings, embedding_dim)

    def forward(self, input: torch.tensor) -> torch.tensor:
        return super(TokenTypeEmbedding, self).forward(input)


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, type_num, embedding_dim,
                 padding_idx, max_len=768, dropout=0.1,
                 pe_type=0, eps=1e-12):
        super(BERTEmbedding, self).__init__()

        if pe_type in ['learnable', 0]:
            self.positional_embedding = LearnablePositionalEmbedding(embedding_dim, max_len)
        elif pe_type in ['sinusoidal', 1]:
            self.positional_embedding = SinusoidalPositionalEmbedding(embedding_dim, max_len)
        else:
            raise Exception("pe_type should be 'learnable'(0) or 'sinusoidal' (1)")

        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.token_type_embedding = TokenTypeEmbedding(type_num, embedding_dim)
        self.layernorm = nn.LayerNorm(embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, token_type=None):
        x = self.token_embedding(x)
        x += self.positional_embedding(x)
        if token_type is not None:
            x += self.token_type_embedding(token_type)

        return self.dropout(self.layernorm(x))
