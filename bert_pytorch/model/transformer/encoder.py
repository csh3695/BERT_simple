import torch
import torch.nn as nn

from .transformer import Transformer


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.dim = config.dim
        self.num_layers = config.num_layers
        if config.reuse_transformer:
            self.proc = nn.ModuleList([Transformer(config)] * config.num_layers)
        else:
            self.proc = nn.ModuleList([Transformer(config) for _ in range(config.num_layers)])

    def forward(self, x, mask=None, log=None):
        out_list = []
        for module in self.proc:
            x = module(x, mask, log)
            out_list.append(x.clone().detach())
        return x, torch.stack(out_list, dim=0)
