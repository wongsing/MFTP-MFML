#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/10/24 10:58
# @Author : wxy
# @FileName: loss_functions.py
# @Software: PyCharm

import numpy as np
import timm.scheduler
from sklearn.ensemble import RandomForestClassifier
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pickle as pkl
import evaluation
import random
from check_seq import *
from sklearn.metrics import *
from evaluation import *
import torch
from torch import nn
import time
import torch.utils.data as Data
from loss_functions import *
from model import *
import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap

def pad_seq(sequence_list, maxlen):
    # 将sequence_list中的每个sequence转换为Tensor对象
    padded_seqs = [torch.tensor(seq) for seq in sequence_list]
    # print("before padded_seqs2:")
    # for i in padded_seqs:
        # print("i.shape:",i.shape)
    # 使用torch.nn.utils.rnn.pad_sequence函数进行填充操作
    padded_seqs = torch.nn.utils.rnn.pad_sequence(padded_seqs, batch_first=True, padding_value=0)
    # print("after padded_seqs:",padded_seqs.shape)
    # 对填充后的序列进行截断或补零以保持统一的maxlen长度
    # padded_seqs = padded_seqs[:, :maxlen,:] 两维特征
    padded_seqs = padded_seqs[:, :maxlen]
    return padded_seqs

def getSequenceData(first_dir):
    # getting sequence data and label
    data, label = [], []
    # path = "{}/{}.txt".format(first_dir, file_name)
    with open(first_dir) as f:
        for each in f:
            each = each.strip()
            if each[0] == '>':
                label_vector = [int(char) for char in each[1:]]
                # label.append(np.array(list(each[1:]), dtype=int))  # Converting string labels to numeric vectors
                label.append(label_vector)
            else:
                data.append(each)
    # print("label:",label)
    label = np.array(label)
    return data, label

#Convert amino acids to vectors
def OE(seq_temp):
    seq = seq_temp
    chars = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y']
    fea = []
    #k = 6
    # print("seq:",seq)
    # for j in st:
    #     if j not in amino_acids:
    #         sign = 1
    #         break
    #     index = amino_acids.index(j)
    #     elemt.append(index)
    #     sign = 0
    sign = 0
    for i in range(len(seq)):
        if seq[i] not in chars:
            sign=1
            # print("youma")
            print("seq:",seq)
            break
        if seq[i] =='A':
            tem_vec = 1
        elif seq[i]=='C':
            tem_vec = 2
        elif seq[i]=='D':
            tem_vec = 3
        elif seq[i]=='E' or seq[i]=='U':
            tem_vec = 4
        elif seq[i]=='F':
            tem_vec = 5
        elif seq[i]=='G':
            tem_vec = 6
        elif seq[i]=='H':
            tem_vec = 7
        elif seq[i]=='I':
            tem_vec = 8
        elif seq[i]=='K':
            tem_vec = 9
        elif seq[i]=='L':
            tem_vec = 10
        elif seq[i]=='M':# or seq[i]=='O':
            tem_vec = 11
        elif seq[i]=='N':
            tem_vec = 12
        elif seq[i]=='P':
            tem_vec = 13
        elif seq[i]=='Q':
            tem_vec = 14
        elif seq[i]=='R':
            tem_vec = 15
        elif seq[i]=='S':
            tem_vec = 16
        elif seq[i]=='T':
            tem_vec = 17
        elif seq[i]=='V':
            tem_vec = 18
        elif seq[i]=='W':
            tem_vec = 19
        elif seq[i]=='X': #or seq[i]=='B' or seq[i]=='Z':
            tem_vec = 20
        elif seq[i]=='Y':
            tem_vec = 21
        #fea = fea + tem_vec +[i]
        fea.append(tem_vec)

    return fea,sign

import pandas as pd
def get_fea_data(data_path):
    data = pd.read_csv(data_path,header=None)
    # print("before data:", data)
    after_data = data.drop(data.columns[0], axis=1)
    # print("after data:", after_data.values)
    numpy_array = after_data.values
    # print("NumPy array:", numpy_array)
    return numpy_array

def get_oe_feature(x_train,maxlen):
    x_train_oe = []
    for i in x_train:
        # print(i)
        oe_feature,sign = OE(i)
        if sign == 0:
            x_train_oe.append(oe_feature)
    x_oe = np.array(pad_seq(x_train_oe, maxlen))
    return x_oe

def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

# getting mlc data and bin data
"""ETFC"""
cpp_train_data = 'dataset/etec/ETEC_train_new.fasta'
cpp_test_data = 'dataset/etec/ETEC_test_new.fasta'
"""TP-LE"""
# cpp_train_data = 'dataset/Tp_LE/train/tp_le_train.fasta'
# cpp_test_data = 'dataset/Tp_LE/test/tp_le_test.fasta'
"""MFTP-Mixed"""
# cpp_train_data = 'dataset/MFTP-Mixed/cdhit_90/data_90_train.fasta'
# cpp_test_data = 'dataset/MFTP-Mixed/cdhit_90/data_90_test.fasta'

x_train, y_train = getSequenceData(cpp_train_data)
x_test, y_test = getSequenceData(cpp_test_data)
print("x_train,x_test:",len(x_train),len(x_test))
max_len = 50
x_oe_train = get_oe_feature(x_train,max_len)
x_oe_test = get_oe_feature(x_test,max_len)
print("x_oe_train:",x_oe_train.shape)
print("x_oe_test:",x_oe_test.shape)

batch_size = 128
print("***************Feature encoing!!***************")
"""MFTP-Mixed"""
# AAC_train = get_fea_data('./dataset/TP-Mixed/split/cdhit_90/data_90_aac_train.csv')
# AAC_test = get_fea_data('./dataset/TP-Mixed/split/cdhit_90/data_90_aac_test.csv')
# print("Before AAC_train,AAC_test:",AAC_train.shape,AAC_test.shape)
# DDE_train = get_fea_data('./dataset/TP-Mixed/split/cdhit_90/data_90_dde_train.csv')
# DDE_test = get_fea_data('./dataset/TP-Mixed/split/cdhit_90/data_90_dde_test.csv')
# print("Before DDE_train,DDE_test:",DDE_train.shape,DDE_test.shape)
# CKSAAGP3_train = get_fea_data('./dataset/TP-Mixed/split/cdhit_90/data_90_cks_train.csv')
# CKSAAGP3_test = get_fea_data('./dataset/TP-Mixed/split/cdhit_90/data_90_cks_test.csv')
# print("Before CKSAAGP3_train,CKSAAGP3_test:",CKSAAGP3_train.shape,CKSAAGP3_test.shape)
# PAAC2_train = get_fea_data('./dataset/TP-Mixed/split/cdhit_90/data_90_paac_train.csv')
# PAAC2_test = get_fea_data('./dataset/TP-Mixed/split/cdhit_90/data_90_paac_test.csv')
# print("Before PAAC2_train,PAAC2_test:",PAAC2_train.shape,PAAC2_test.shape)
# APAAC2_train = get_fea_data('./dataset/TP-Mixed/split/cdhit_90/data_90_apaac_train.csv')
# APAAC2_test = get_fea_data('./dataset/TP-Mixed/split/cdhit_90/data_90_apaac_test.csv')
# print("Before APAAC2_train,PAAAC2_test:",APAAC2_train.shape,APAAC2_test.shape)

"""TP-LE"""
# AAC_train = get_fea_data('./dataset/Tp_LE/train/Tple_aac_train.csv')
# AAC_test = get_fea_data('./dataset/Tp_LE/test/Tple_aac_test.csv')
# print("Before AAC_train,AAC_test:",AAC_train.shape,AAC_test.shape)
# DDE_train = get_fea_data('./dataset/Tp_LE/train/Tple_dde_train.csv')
# DDE_test = get_fea_data('./dataset/Tp_LE/test/Tple_dde_test.csv')
# print("Before DDE_train,DDE_test:",DDE_train.shape,DDE_test.shape)
# CKSAAGP3_train = get_fea_data('./dataset/Tp_LE/train/Tple_cks3_train.csv')
# CKSAAGP3_test = get_fea_data('./dataset/Tp_LE/test/Tple_cks3_test.csv')
# print("Before CKSAAGP3_train,CKSAAGP3_test:",CKSAAGP3_train.shape,CKSAAGP3_test.shape)
# PAAC2_train = get_fea_data('./dataset/Tp_LE/train/Tple_paac_train.csv')
# PAAC2_test = get_fea_data('./dataset/Tp_LE/test/Tple_paac_test.csv')
# print("Before PAAC2_train,PAAC2_test:",PAAC2_train.shape,PAAC2_test.shape)
# APAAC2_train = get_fea_data('./dataset/Tp_LE/train/Tple_apaac_train.csv')
# APAAC2_test = get_fea_data('./dataset/Tp_LE/test/Tple_apaac_test.csv')
# print("Before APAAC2_train,PAAAC2_test:",APAAC2_train.shape,APAAC2_test.shape)

"""ETFC"""
AAC_train = get_fea_data('./dataset/etec/etec_train_aac.csv')
AAC_test = get_fea_data('./dataset/etec/etec_test_AAC.csv')
print("Before AAC_train,AAC_test:",AAC_train.shape,AAC_test.shape)
DDE_train = get_fea_data('./dataset/etec/etec_train_dde.csv')
DDE_test = get_fea_data('./dataset/etec/etec_test_dde.csv')
print("Before DDE_train,DDE_test:",DDE_train.shape,DDE_test.shape)
CKSAAGP3_train = get_fea_data('./dataset/etec/etec_train_cksaagp3.csv')
CKSAAGP3_test = get_fea_data('./dataset/etec/etec_test_cksaagp3.csv')
print("Before CKSAAGP3_train,CKSAAGP3_test:",CKSAAGP3_train.shape,CKSAAGP3_test.shape)
PAAC2_train = get_fea_data('./dataset/etec/etec_train_paac2.csv')
PAAC2_test = get_fea_data('./dataset/etec/etec_test_paac2.csv')
print("Before PAAC2_train,PAAC2_test:",PAAC2_train.shape,PAAC2_test.shape)
APAAC2_train = get_fea_data('./dataset/etec/etec_train_apaac2.csv')
APAAC2_test = get_fea_data('./dataset/etec/etec_test_apaac2.csv')

x_hc_train = np.c_[AAC_train,DDE_train,CKSAAGP3_train,PAAC2_train,APAAC2_train]
x_hc_test = np.c_[AAC_test,DDE_test,CKSAAGP3_test,PAAC2_test,APAAC2_test]
print("x_hc_train,x_hc_test:",x_hc_train.shape,x_hc_test.shape)

class MyDataSet(Data.Dataset):
    def __init__(self, seq_data,hand_data,label):
        self.seq_data = seq_data
        self.hand_data = hand_data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.seq_data[idx],self.hand_data[idx], self.label[idx]

dataset_train = MyDataSet(x_train_combined ,x_hc_train,y_train)
dataset_test = MyDataSet(x_test_combined,x_hc_test,y_test)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

hc_input = x_hc_train.shape[1]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = rnn_hc(vocab_size=max_len,maxlen=max_len,d_model=128,device=device)

model.to(device)
lr = 0.005
optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr,weight_decay=0.1)
criterion = IntegrationLoss()
class_name = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
             'AVP',
             'BBP', 'BIP',
             'CPP', 'DPPIP',
             'QSP', 'SBP', 'THP']
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.75, verbose=True)

def evaluate(model, test_iter,epoch):
    model.eval()
    label_list = []
    logit_list = []
    output_list = []
    with torch.no_grad():
        for data in test_iter:
            x_seq,x_hc, label = data
            x_seq = x_seq.to(device)
            x_hc = x_hc.to(device)
            label = label.to(device)
            logits,output= model(x_seq,x_hc)
            output_list.extend(output.cpu().detach().numpy())
            label_list.extend(label.cpu().detach().numpy())
            logit_list.extend(logits.cpu().detach().numpy())
    scores = evaluation.evaluate(np.array(logit_list), np.array(label_list))
    return scores

steps = 1
# 创建一个空列表来存储每个epoch的平均损失
losses = []
train_accs = []
test_accs = []
test_metric = []
lr_arr = []
epochs = 100
for epoch in range(epochs):
    print("*****************Epoch {:04d}********************".format(epoch))
    lr_arr.append(optimizer.param_groups[0]['lr'])
    t0 = time.time()
    model.train()
    repres_list = []
    label_list = []
    logit_list = []
    arr_loss = []
    epoch_loss = 0.0
    total_batches = 0
    for data in dataloader_train:
        x_seq, x_hc,label = data
        x_seq = x_seq.to(device)
        x_hc = x_hc.to(device)
        label = label.to(device)
        logits, output_bert = model(x_seq, x_hc)
        loss = criterion(logits, label.float())
        label_list.extend(label.cpu().detach().numpy())
        logit_list.extend(logits.cpu().detach().numpy())
        epoch_loss += loss.item()
        total_batches += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        steps += 1
        arr_loss.append(loss.item())
    # scheduler.step(loss)
    train_scores = evaluation.evaluate(np.array(logit_list), np.array(label_list))
    avgl = np.mean(arr_loss)
    losses.append(avgl)
    t1 = time.time()
    print("Epoch {:04d}｜ Avg_Loss {:.4f}| Time {:.4f}s".format(epoch, avgl, t1 - t0))
    print("Train_scores: accuracy={accuracy:.4f},aiming={aiming:.4f}, coverage={coverage:.4f},"
          "absolute_true={absolute_true:.4f}, absolute_false={absolute_false:.4f}".format(**train_scores))
    train_accs.append(train_scores['accuracy'])

    test_scores = evaluate(model, dataloader_test,epoch)
    print("Test_scores: accuracy={accuracy:.4f},aiming={aiming:.4f}, coverage={coverage:.4f},"
          "absolute_true={absolute_true:.4f}, absolute_false={absolute_false:.4f}".format(**test_scores))
    test_accs.append(test_scores['accuracy'])
    test_metric.append(test_scores)

test_best_metric = max(test_metric, key=lambda x: x['accuracy'])
print("Best_test_scores: accuracy={accuracy:.4f},aiming={aiming:.4f}, coverage={coverage:.4f},"
      "absolute_true={absolute_true:.4f}, absolute_false={absolute_false:.4f}".format(**test_best_metric))


# 使用Matplotlib绘制损失图表
import matplotlib.pyplot as plt
plt.figure(22, figsize=(16, 12))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
# 使用Matplotlib绘制损失和准确率图表
plt.subplot(2,2,1)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(losses)
plt.subplot(2,2,2)
plt.title('Training ACC')
plt.xlabel('Epoch')
plt.ylabel('ACC')
plt.plot(train_accs)
plt.subplot(2,2,3)
plt.title('Training LR')
plt.xlabel('Epoch')
plt.ylabel('LR')
plt.plot(lr_arr)
plt.subplot(2,2,4)
plt.title('Testing ACC')
plt.xlabel('Epoch')
plt.ylabel('ACC')
plt.plot(test_accs)
plt.show()


