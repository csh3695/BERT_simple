import torch
import torch.nn as nn

from bert_pytorch.model.heads import *
from bert_pytorch.model.embedding import *
from bert_pytorch.model.transformer import *
from bert_pytorch.model.utils import GELU


class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.dim = config.dim
        self.eps = config.eps
        self.pad_symbol = config.pad_symbol
        self.vocab_size = config.vocab_size
        self.cuda = bool(config.device == 'cuda') & torch.cuda.is_available()

        self.embedding = BERTEmbedding(config)
        self.encoder = Encoder(config)
        self.dropout = nn.Dropout(config.dropout)
        self.out_norm = nn.LayerNorm(config.dim, eps=config.eps)

    def _gen_mask(self, inp: torch.tensor) -> torch.tensor:
        mask = torch.ones(inp.size(0), inp.size(1), inp.size(1))
        for i in range(inp.size(0)):
            mask[i, (inp[i] == self.pad_symbol), :] = 0
            mask[i, :, (inp[i] == self.pad_symbol)] = 0

        if self.cuda:
            return mask.cuda().long()

        return mask.long()

    def forward(self, x, mask=None, token_type=None, log=None):
        if mask is None:
            mask = self._gen_mask(x)
        x = self.embedding(x, token_type)
        if log is not None:
            log['embedding'] = x.clone().detach().cpu()
        return self.encoder(x, mask, log)


class BERTLM(nn.Module):
    def __init__(self, config):
        super(BERTLM, self).__init__()
        self.bert = BERT(config)
        self.head = BERTHead(config, self.bert.embedding.token_embedding.embedding)
        self._init_weight()

    def _init_weight(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, mask=None, token_type=None, candidates=None, log=None):
        x, layer_out = self.bert(x, mask, token_type, log)
        return self.head(x, candidates), layer_out
