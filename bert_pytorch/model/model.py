import torch
import torch.nn as nn

from bert_pytorch.model import module, embedding
from bert_pytorch.model.utils import GeLU


class BERT(nn.Module):
    def __init__(self, vocab_size=8000, dim=512, max_len=768, num_layers=6, num_heads=8, num_token_type=2, dropout=0.1, eps=1e-12):
        super(BERT, self).__init__()
        self.dim = dim
        self.eps = eps
        self.pad_symbol = 4
        self.vocab_size = vocab_size
        self.embedding = embedding.BERTEmbedding(vocab_size=vocab_size, type_num=num_token_type, embedding_dim=dim,
                                                 padding_idx=self.pad_symbol, max_len=max_len, dropout=dropout, pe_type=0,
                                                 eps=1e-12)
        self.encoder = module.Encoder(dim, dim << 2, num_layers, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(dim, eps=eps)

    def _gen_mask(self, inp: torch.tensor) -> torch.tensor:
        mask = torch.ones(inp.size(0), inp.size(1), inp.size(1))
        for i in range(inp.size(0)):
            mask[i][(inp[i] == self.pad_symbol)][:] = 0
            mask[i][:][(inp[i] == self.pad_symbol)] = 0

        return mask.cuda().long()

    def forward(self, x, mask=None, token_type=None):
        if mask is None:
            mask = self._gen_mask(x)
        x = self.embedding(x, token_type)
        return self.encoder(x, mask)


class BERTLM(nn.Module):
    def __init__(self, bert):
        super(BERTLM, self).__init__()
        self.bert = bert
        self.prediction_head = nn.Sequential(
            nn.Linear(bert.dim, bert.dim),
            GeLU(),
            nn.LayerNorm(bert.dim, bert.eps)
        )
        self._init_weight()

    def _init_weight(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.005)
            if isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, mask=None, token_type=None):
        x, layer_out = self.bert(x, mask, token_type)
        return (self.bert.embedding.token_embedding.backward(x), layer_out)