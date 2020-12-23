import os
os.chdir('/home/ubuntu/BERT/bert_pytorch/results/loss')

import json
import itertools
import matplotlib.pyplot as plt

from glob import glob

loss_file = sorted(glob('./*.json'))
loss_list = []
for file in loss_file:
	with open(file, 'r') as f:
		loss_list += json.load(f)

title = ' '.join(loss_file)
title = title.replace('.json', '').replace('./', '').replace(' ','_')

plt.switch_backend('agg')

plt.plot(loss_list)
plt.title(title)

plt.savefig(f'{title}.png')

print(f'[DONE] {title}')

print(f'scp -P 16022 ubuntu@14.49.45.214:/home/ubuntu/BERT/bert_pytorch/results/loss/{title}.png ./')
