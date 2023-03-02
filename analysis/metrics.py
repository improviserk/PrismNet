from pandas import DataFrame
import os
import pandas as pd
import torch
from prismnet.loader import SeqicSHAPE
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
from collections import defaultdict

def readcsv(path):
    data = pd.read_excel(path)[:100]
    return data.to_dict('list')


def calculateSize(data_path):
    train_loader = torch.utils.data.DataLoader(SeqicSHAPE(data_path), \
        batch_size=64, shuffle=True)

    test_loader  = torch.utils.data.DataLoader(SeqicSHAPE(data_path, is_test=True), \
    batch_size=64, shuffle=False)
    # print("Train set:", len(train_loader.dataset))
    # print("Test  set:", len(test_loader.dataset))
    return len(train_loader.dataset), len(test_loader.dataset)

datasets = 'data/datasets'
names, trains, tests = [], [], []
for ds in os.listdir(datasets):
    train, test = calculateSize(os.path.join(datasets, ds))
    if train < 4000: continue
    trains.append(train)
    tests.append(test)
    names.append(ds[:-3])

auc = []
RBP = []
dic = {}
ns = set(names)
path = 'data/result'
for ds in os.listdir(path):
    model = ds[:-5]
    data = readcsv(os.path.join(path, ds))
    tmp = defaultdict(list)
    for i in range(len(data['RBP'])):
        if data['RBP'][i] in ns:
            tmp['RBP'].append(data['RBP'][i])
            tmp['auc'].append(data['auc'][i])
            tmp['acc'].append(data['acc'][i])
            tmp['epoc'].append(data['epoc'][i])
    
    for i in range(len(tmp['epoc'])):
        tmp['epoc'][i] += 10
    sns.lineplot(data = tmp, x = 'RBP', y = 'epoc', label = model)
plt.xticks(rotation = 60, size = 5)
# plt.xticks([])
# sns.set(rc = {'figure.figsize':(50,1000)})
# plt.ylabel('auc')
plt.tight_layout()
plt.legend()
plt.show()
print('----')
print(names)
