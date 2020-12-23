import numpy as np
import torch
from pprint import pprint
from random import randint
from torch.utils.data import Dataset


def test_model(dataset: Dataset,
               mdl: torch.nn.Module,
               seed=None):
    if seed is not None:
        x, mask, y, length = dataset[seed]
    else:
        x, mask, y, length = dataset[randint(0, len(dataset)-1)]
    x = x[:length]
    target_sent = dataset.vocab.DecodeIds(x.cpu().detach().numpy().tolist())
    x = x.unsqueeze(0)
    cur_train = mdl.training
    mdl.eval()
    mdl.cuda()
    pred_raw = mdl(x.cuda())
    if isinstance(pred_raw, tuple):
        pred_raw = pred_raw[0].cpu().detach()
    else:
        pred_raw = pred_raw.cpu().detach()
    pred_ids = torch.argmax(pred_raw, dim=-1)
    pred_sent = dataset.vocab.DecodeIds(pred_ids.tolist()[0])

    if cur_train:
        mdl.train()
    pprint(target_sent)
    pprint(pred_sent)

    return pred_raw, pred_ids, pred_sent
