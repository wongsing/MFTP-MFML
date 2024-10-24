#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/10/24 10:58
# @Author : wxy
# @FileName: model.py
# @Software: PyCharm

import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.init as init

class Embedding(nn.Module):
    def __init__(self,vocab_size,d_model,max_len,device):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding (look-up table)
        self.pos_embed = nn.Embedding(max_len, d_model)  # position embedding
        self.norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self, x):
        seq_len = x.size(1)  # x: [batch_size, seq_len]
        pos = torch.arange(seq_len, device=self.device, dtype=torch.long)  # [seq_len]
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        embedding = self.pos_embed(pos)
        embedding = embedding + self.tok_embed(x)
        embedding = self.norm(embedding)
        return embedding

class rnn_hc(nn.Module):
    def __init__(self, vocab_size, d_model, device, maxlen):
        super(rnn_hc, self).__init__()
        """embedding"""
        self.embedding = Embedding(vocab_size=vocab_size, d_model=d_model, device=device, max_len=maxlen)
        self.emb_size = d_model
        self.hidden_size = 128

        """RNN"""
        self.gru = nn.GRU(self.emb_size, self.hidden_size, num_layers=3,bidirectional=True,batch_first=True)
        self.bilstm = nn.LSTM(self.emb_size, self.hidden_size, num_layers=6,bidirectional=True,batch_first=True)

        """Classifier"""
        self.fc = nn.Sequential(
            nn.Linear(13366, 2048),
            nn.Dropout(0.3),
            nn.PReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Linear(64, 21)
        )

    def forward(self, x_seq,x_hc):
        """Eebedding+lstm"""
        x_emb = self.embedding(x_seq)
        out_rnn, _ = self.bilstm(x_emb.float())
        out_rnn = out_rnn.reshape(out_rnn.shape[0],-1)

        x_hc = x_hc.float()
        out = torch.cat((out_rnn, x_hc), dim=-1)
        logits = self.fc(out)
        return logits, out