#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""


import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


from thop import profile, clever_format

def profile_macs_params(model, input_size = (1,3,1024)):
    model = model.eval()
    inp = torch.randn(input_size)
    #print("Input size:", inp.size(), "Output size:", model(inp).size())
    macs, params = profile(model, inputs=(inp,),verbose=False)
    macs_str, params_str = clever_format([macs, params], "%.3f")
    #print(f"MACs {macs_str} Params {params_str}")
    return macs, params, macs_str, params_str

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def cross_entropy_loss_with_soft_target(pred, soft_target):
	logsoftmax = nn.LogSoftmax(dim=1)
	return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))

def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
	soft_target = label_smooth(target, pred.size(1), label_smoothing)
	return cross_entropy_loss_with_soft_target(pred, soft_target)



def make_divisible(v, divisor, min_val=None):
	"""
	This function is taken from the original tf repo.
	It ensures that all layers have a channel number that is divisible by 8
	It can be seen here:
	https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
	:param v:
	:param divisor:
	:param min_val:
	:return:
	"""
	if min_val is None:
		min_val = divisor
	# new_v = max(min_val, int(v) // divisor * divisor)
	new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v
