import json
import os

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def save_dataset(vocab_dir='./wikidata/korwiki_vocab_model.model',
                 tkn_dir='./wikidata/korwiki_clean.json',
                 words_per_sent=512,
                 save_dir='./bert_pytorch/dataset/'):
    dataset = WikiDataset(vocab_dir, tkn_dir, words_per_sent)
    with open(save_dir + 'wikidata.dset', 'wb') as f:
        torch.save(dataset, f)
    return dataset


def load_dataset(load_dir='./bert_pytorch/dataset/', rebuild=False):
    if rebuild or not os.path.exists(load_dir + 'wikidata.dset'):
        print("[+] Dataset not exists. Create new one.")
        save_dataset(save_dir=load_dir)
    with open(load_dir + 'wikidata.dset', 'rb') as f:
        return torch.load(f)


class WikiDataset(Dataset):
    def __init__(self, vocab_dir, tkn_dir, words_per_sent):
        super(WikiDataset, self).__init__()
        self.words_per_sent = words_per_sent
        self.vocab = spm.SentencePieceProcessor()
        self.vocab.Load(vocab_dir)
        self.special_symbol = dict()
        for sym in ['CLS', 'PAD', 'MASK', 'CMA', 'EOS', 'UNK', 'SEP']:
            self.special_symbol[sym] = self.vocab.PieceToId(sym)
        self.data, self.data_len, self.special_spot = self._build_data(tkn_dir)
        self.mask_ratio = 0.15

    def _build_data(self, tkn_dir):
        """
        self.data : len(n_sent) x n_word, Long type tensor
        """

        res = []
        sym_res = []
        len_res = []

        f_name = tkn_dir

        with open(f_name, 'r') as f:
            docs = list(json.load(f).values())
        for doc in tqdm(docs):
            doc_res = np.zeros(self.words_per_sent).astype(np.long)
            doc_res.fill(self.special_symbol['PAD'])
            special_sym_res = np.zeros(self.words_per_sent).astype(np.long)
            if len(doc) < 5:
                continue
            concat_res = []
            for sent in doc:
                sent = self.vocab.EncodeAsIds(sent)
                if len(sent) <= 6:
                    continue
                concat_res += sent + [self.special_symbol['SEP']]
                if len(concat_res) > self.words_per_sent:
                    break
            concat_res = [self.special_symbol['CLS']] + concat_res[:-1] + [self.special_symbol['EOS']]
            curlen = min(self.words_per_sent, len(concat_res))
            len_res.append(curlen)
            doc_res[:curlen] = concat_res[:curlen]
            for sym_idx in self.special_symbol.values():
                special_sym_res = special_sym_res | (doc_res == sym_idx)
            res.append(doc_res.copy())
            sym_res.append(special_sym_res)

        return torch.LongTensor(np.stack(res)), \
               torch.LongTensor(np.array(len_res)), \
               torch.LongTensor(np.stack(sym_res)).bool()

    def __getitem__(self, item):
        dat = self.data[item].clone()
        lens = self.data_len[item]
        syms = self.special_spot[item]
        mask = ((torch.rand_like(dat.float()) < self.mask_ratio) & (~syms))

        if type(item) == int:
            mask[lens:] = 0
        else:
            for i in range(len(item)):
                mask[i, lens[i]:] = 0

        mask_val = dat[mask]
        mask_mask = torch.rand_like(mask_val.float())
        mask_val[mask_mask < 0.8] = self.special_symbol['MASK']
        mask_val[0.9 <= mask_mask] = (
                    mask_val[0.9 <= mask_mask].float().uniform_() * (self.vocab.vocab_size() - 1)).long()

        dat[mask] = mask_val
        return dat, mask, self.data[item], lens

    def __len__(self):
        return len(self.data)


"""
dataset = save_dataset()

f_name = './wikidata/korwiki_clean.json'

lens = []

with open(f_name, 'r') as f:
    docs = list(json.load(f).values())


vocab = spm.SentencePieceProcessor()
vocab.Load('./wikidata/korwiki_vocab_model.model')

for doc in tqdm(docs):
    length = 1
    for sent in doc:
        length += len(vocab.EncodeAsIds(sent))+1    
    lens.append(length)

import matplotlib.pyplot as plt

lens = sorted(lens)
plt.hist(lens[len(lens)//100:(len(lens)*99)//100], bins=100, range=(0, 1000))
plt.show()
"""
