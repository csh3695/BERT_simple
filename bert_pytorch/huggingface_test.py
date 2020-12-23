import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim import AdamW
from torch.utils.data import DataLoader

from bert_pytorch import model
from bert_pytorch.dataset import load_dataset
from bert_pytorch.trainer import linear_lr_scheduler
from bert_pytorch.model.utils import tester
from transformers import BertModel, BertConfig
import gluonnlp as nlp


dataset = load_dataset()


bert_config = {
    'attention_probs_dropout_prob': 0.1,
    'hidden_act': 'gelu',
    'hidden_dropout_prob': 0.1,
    'hidden_size': 768,
    'initializer_range': 0.02,
    'intermediate_size': 3072,
    'max_position_embeddings': 768,
    'num_attention_heads': 12,
    'num_hidden_layers': 12,
    'type_vocab_size': 2,
    'vocab_size': 5000
}

bertmodel = BertModel(config=BertConfig.from_dict(bert_config))

x, mask, y = dataset[[0,1,2]]
bertmodel(x)[1]