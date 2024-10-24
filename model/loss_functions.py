#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/10/24 10:58
# @Author : wxy
# @FileName: loss_functions.py
# @Software: PyCharm
import math
import torch
import torch.nn as nn

class BWeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BWeightedCrossEntropyLoss, self).__init__()

    def forward(self, output, target):
        if target.dim() == 4:
            target[target == 4] = 3  # label [2] -> [0]
            target = expand_target(target, n_class=output.size()[1])  # [N,H,W,D] -> [N,1,H,W,D]
        entropy_criterion = nn.BCEWithLogitsLoss()
        bce_l = entropy_criterion(output, target)
        return bce_l

class FocalLoss_use(nn.Module):

    def __init__(self, alpha, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, output, target):
        num_classes = output.size(1)
        alpha_list = torch.tensor([self.alpha] * num_classes)
        # assert len(self.alpha) == num_classes, \
        #     'Length of weight tensor must match the number of classes'
        logp = F.cross_entropy(output, target, alpha_list)
        p = torch.exp(-logp)
        focal_loss = (1 - p) ** self.gamma * logp

        return torch.mean(focal_loss)

class DCSLoss(nn.Module):
    """DCS loss"""
    def __init__(self, smooth=1e-4, p=2, alpha=0.01, reduction='mean'):
        super(DCSLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(input)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        pre_pos = predict*((1-predict)**self.alpha)

        num = torch.sum(torch.mul(pre_pos, target), dim=1)
        den = torch.sum(pre_pos.pow(self.p) + target.pow(self.p), dim=1)+self.smooth

        loss = 1 - (2 * num + self.smooth) / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class IntegrationLoss(nn.Module):
    def __init__(self):
        super(IntegrationLoss, self).__init__()
        self.bec = BWeightedCrossEntropyLoss()
        self.focal_new = FocalLoss_use(alpha=0.1,gamma=2)
        self.smooth = False
        self.dice = DCSLoss(alpha=0.03)

    def forward(self, output, target):
        dice_loss1 = self.dice(output, target)
        focal_loss = self.focal_new(output, target)
        bce_l = self.bec(output, target)

        alpah = 0.8
        result = dice_loss1 +alpah * focal_loss + (1 - alpah) * bce_l
        if self.smooth:
            result = log_cosh_smooth(result)
        return result








