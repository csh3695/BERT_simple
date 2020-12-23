import sys, os
sys.path.append('/home/ubuntu/BERT/')
os.chdir('/home/ubuntu/BERT')

import argparse
import time
import torch
import json
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim import AdamW
from torch.utils.data import DataLoader

from bert_pytorch import model
from bert_pytorch.dataset import load_dataset, save_dataset
from bert_pytorch.trainer import linear_lr_scheduler
from bert_pytorch.model.utils import tester, FSPLoss, SoftLabelLoss

parser = argparse.ArgumentParser()
parser.add_argument('start', type=int, default=0)
parser.add_argument('batch_size', type=int, default=28)
parser.add_argument('epoch', type=int, default=250)
parser.add_argument('mask_only', type=str, default="False")

args = parser.parse_args()
#args = {'start':0, 'batch_size':28, 'epoch':250, 'mask_only':False}
start_epoch = args.start

print(f'{args.start}~{args.epoch} | batch {args.batch_size} | mask_only {args.mask_only}')

#dataset = save_dataset(words_per_sent=512)
dataset = load_dataset()

bert = model.BERT(vocab_size=8000, dim=512, max_len=512, num_layers=6, num_heads=8)
bertlm = model.BERTLM(bert=bert)

bert_t = model.BERT(vocab_size=8000, dim=512, max_len=512, num_layers=12, num_heads=8)
bertlm_t = model.BERTLM(bert=bert_t)

bertlm_t.load_state_dict(torch.load(f'./bert_pytorch/model/model_saved/bert.ep247.mdl'))

bertlm.bert.embedding = bertlm_t.bert.embedding

bertlm_t.eval()
for param in bertlm_t.parameters():
    param.requires_grad = False

if start_epoch != 0:
    try:
        bertlm.load_state_dict(torch.load(f'./bert_pytorch/model/model_saved_kd/bert_kd.ep{start_epoch-1}.mdl'))
    except:
        raise Exception("No File detected")

optimizer = AdamW(bertlm.bert.encoder.parameters(), lr=1e-4)
criteria = nn.CrossEntropyLoss()
kd_criteria = FSPLoss(t_layer=12, s_layer=6, stride=2)
sf_criteria = SoftLabelLoss()

params = filter(lambda p: p.requires_grad, bertlm.parameters())
num_params = sum([np.prod(p.size()) for p in params])
print("# of params:", num_params)

cuda = True
loss_list = []
batch_size = args.batch_size
epoch = args.epoch
mask_only = True if args.mask_only=='True' else False

print(f"mask_only {mask_only}")

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
lr_sch = linear_lr_scheduler(start_val=1e-7, warm_step=len(dataloader)*2, warm_val=3e-5, end=len(dataloader) * epoch,
                             i=1+(len(dataloader)*start_epoch))

if cuda:
    bertlm = bertlm.cuda()
    bertlm_t = bertlm_t.cuda()

bertlm.train()

start = time.time()
ep_prev = start_epoch

for ep in range(start_epoch, epoch):
    for i, data in enumerate(dataloader):
        optimizer.zero_grad()
        x, mask, y, length = data
        if cuda:
            x = x.cuda()
            y = y.cuda()
        y_pred, y_layer_res = bertlm(x)
        t_pred, t_layer_res = bertlm_t(x)

        source = torch.cat([y_pred[i][:length[i]] for i in range(len(length))], dim=0)
        target = torch.cat([y[i][:length[i]] for i in range(len(length))], dim=0)
        kd_loss = kd_criteria(y_layer_res, t_layer_res)
        sf_loss = sf_criteria(y_pred, t_pred)
        if mask_only:
            masked = torch.cat([mask[i][:length[i]] for i in range(len(length))], dim=0)
            lm_loss = criteria(y_pred.reshape(-1, y_pred.shape[-1]), y.reshape(-1))
        else:
            lm_loss = criteria(y_pred.reshape(-1, y_pred.shape[-1]), y.reshape(-1))
        loss_list.append((kd_loss.item(), lm_loss.item(), sf_loss.item()))
        loss = kd_loss + sf_loss + lm_loss
        loss.backward()
        optimizer.step()
        optimizer.param_groups[0]['lr'] = next(lr_sch)
        if i % 10 == 0:
            print(f"EP\t{ep} | data\t{i} "
                  f"| kd_loss\t{kd_loss:.8f} "
                  f"| sf_loss\t{sf_loss:.8f} "
                  f"| lm_loss\t{lm_loss:.8f} "
                  f"| loss\t{loss:.8f} | lr\t{optimizer.param_groups[0]['lr']} "
                  f"| time\t{(time.time()-start):.3f}s")
            start = time.time()
        del kd_loss, sf_loss, lm_loss
        if i % 1000 == 0:
            tester.test_model(dataset, bertlm)
    torch.save(bertlm.state_dict(), f'./bert_pytorch/model/model_saved_kd/bert_kd.ep{ep}.mdl')

    if ep % 10 == 9:
        with open(f'./bert_pytorch/results/loss_kd/loss_{ep_prev}_{ep}.json', 'w') as f:
            json.dump(loss_list, f)
        loss_list = []
        ep_prev = ep+1
