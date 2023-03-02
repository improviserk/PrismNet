import argparse, os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchinfo import summary
import warnings
warnings.filterwarnings("ignore")


from tensorboardX import SummaryWriter
from sklearn import metrics
import numpy as np

import prismnet.model as arch
from prismnet import train, validate, inference, log_print, compute_saliency, compute_saliency_img, compute_high_attention_region
#compute_high_attention_region

# from prismnet.engine.train_loop import 
from prismnet.model.utils import GradualWarmupScheduler
from prismnet.loader import SeqicSHAPE
from prismnet.utils import datautils
from prismnet.model.HViT import ViT_large
from prismnet.model.HViT import ViT_small
from prismnet.model.HViT import ViT_medium
from prismnet.model.HViT import ViT_RBP_hybrid
from pandas import DataFrame

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False


def calculateSize(data_path):
    train_loader = torch.utils.data.DataLoader(SeqicSHAPE(data_path), \
        batch_size=64, shuffle=True)

    test_loader  = torch.utils.data.DataLoader(SeqicSHAPE(data_path, is_test=True), \
    batch_size=64, shuffle=False)
    # print("Train set:", len(train_loader.dataset))
    # print("Test  set:", len(test_loader.dataset))
    return len(train_loader.dataset), len(test_loader.dataset)

path = 'data/datasets'
names, trains, tests = [], [], []
for ds in os.listdir(path):
    train, test = calculateSize(os.path.join(path, ds))
    if train < 4000: continue
    trains.append(train)
    tests.append(test)
    names.append(ds[:-3])

print(len(trains))

dic = {'dataset': names, 'train':trains, 'test': tests}

sns.lineplot(data=dic, x='dataset', y = 'train', label = 'Train')
sns.lineplot(data=dic, x='dataset', y = 'test', label = 'Test')
plt.xticks(rotation = 60)
# plt.xticks([])
plt.ylabel('size')
plt.legend()
plt.tight_layout()
plt.show()