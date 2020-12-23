import torch
import matplotlib.pyplot as plt
import seaborn as sns

from bert_pytorch import model
from bert_pytorch.model.embedding import SinusoidalPositionalEmbedding
from bert_pytorch.dataset import load_dataset

bert = model.BERT(vocab_size=8000, dim=512, max_len=512, num_layers=12, num_heads=8)
bertlm = model.BERTLM(bert=bert)

sinusoidal = SinusoidalPositionalEmbedding(512, 512)

plt.clf()
sin_pe_weight = sinusoidal.pe[0]
sin_pe_sim = torch.matmul(sin_pe_weight, sin_pe_weight.transpose(-1, -2))
sns.heatmap(sin_pe_sim)
plt.title('Sinusoidal inner_prod')
plt.savefig(f'./bert_pytorch/results/pe/sinusoidal_inner_prod.png', dpi=600)


plt.clf()
pseudo_pe_weight = torch.zeros([512, 512])
pseudo_pe_weight.normal_(0, 0.005)
pseudo_pe_sim = torch.matmul(pseudo_pe_weight, pseudo_pe_weight.transpose(-1, -2))
sns.heatmap(pseudo_pe_sim)
plt.title('Random(0, 0.005) inner_prod')
plt.savefig(f'./bert_pytorch/results/pe/random_inner_prod.png', dpi=600)

for i in range(134):
    plt.clf()
    bertlm.load_state_dict(torch.load(f'./bert_pytorch/model/model_saved/bert.ep{i}.mdl'))
    pe_weight = bertlm.bert.embedding.positional_embedding.embedding.weight.detach()
    pe_sim = torch.matmul(pe_weight, pe_weight.transpose(-1, -2))
    sns.heatmap(pe_sim, vmin=-0.07, vmax=0.25)
    plt.title(f'learnable_ep{i} inner_prod')
    plt.savefig(f'./bert_pytorch/results/pe/learn_ep{i}_inner_prod.png', dpi=600)

plt.clf()
te_orig_weight = bertlm.bert.embedding.token_embedding.embedding.weight.detach()
te_proj_weight = bertlm.bert.embedding.token_embedding(torch.arange(8000)).detach()
sns.heatmap(te_orig_weight)
plt.show()

sns.heatmap(te_proj_weight)
plt.show()

plt.clf()
pe_weight_list = pe_weight.reshape(-1)
te_proj_weight_list = te_proj_weight.reshape(-1)
plt.hist(pe_weight_list, bins=500, range=(-0.02, 0.02), density=True, histtype='step')
plt.axvline(pe_weight_list.mean(), color='k', linestyle='dashed', linewidth=1)
plt.hist(te_proj_weight_list, bins=500, range=(-0.02, 0.02), density=True, histtype='step')
plt.axvline(te_proj_weight_list.mean(), color='k', linewidth=1)
plt.legend(['pos_avg', 'tkn_avg', 'pos', 'tkn'])
plt.show()

proj = bertlm.bert.embedding.token_embedding.proj.weight
proj_back = bertlm.bert.embedding.token_embedding.cls[0].weight

sns.heatmap(proj.detach())
plt.show()

pr = proj.detach()
prb = proj_back.transpose(-1, -2).detach()

prn = (pr - pr.mean())/pr.std()
prbn = (prb - prb.mean())/prb.std()

prn = pr.reshape(-1)
prbn = prb.reshape(-1)

plt.plot(prn, prbn, '*')


ls = []

for i in range(27):
    plt.clf()
    bertlm.load_state_dict(torch.load(f'./bert_pytorch/model/model_saved/bert.ep{i}.mdl'))
    pe_weight_list = bertlm.bert.embedding.positional_embedding.embedding.weight.detach().reshape(-1)
    te_proj_weight_list = bertlm.bert.embedding.token_embedding(torch.arange(8000)).detach().reshape(-1)
    ls.append(((pe_weight_list.mean(), pe_weight_list.std()), (te_proj_weight_list.mean(), te_proj_weight_list.std())))
    print(f"EP\t{i}| pe{(pe_weight_list.mean(), pe_weight_list.std())}"
          f"| tk{(te_proj_weight_list.mean(), te_proj_weight_list.std())}")

pe_ls, te_ls = list(zip(*ls))
pe_mean, pe_std = list(zip(*pe_ls))
te_mean, te_std = list(zip(*te_ls))


import torch
import matplotlib.pyplot as plt
import seaborn as sns

from bert_pytorch import model
from bert_pytorch.dataset import load_dataset

bert = model.BERT(vocab_size=8000, dim=512, max_len=512, num_layers=12, num_heads=8)
bertlm = model.BERTLM(bert=bert)


bertlm.load_state_dict(torch.load(f'./bert_pytorch/model/model_saved/bert.ep{133}.mdl'))
bert = bertlm.bert
bert.eval()
dataset = load_dataset()

for param in bert.parameters():
    param.requires_grad = False

bert = bert.cuda()

x, mask, y, length = dataset[[235]]
x = x.cuda()

attn_mask = bert._gen_mask(x)

e = bert.embedding(x)


for layer in range(12):
    e, attn = bert.encoder.proc(e, attn_mask, attn=True)
    for idx in range(len(attn)):
        for head in range(len(attn[0])):
            plt.clf()
            sns.heatmap(attn[idx][head].detach().cpu().add(1e-4).log(), cmap='Blues', vmin=-9, vmax=0)
            plt.title(f'attn_layer{layer}_h{head}.png')
            plt.savefig(f'./bert_pytorch/results/attn/attn_h{head:02}_layer{layer:02}.png', dpi=600)
