
import copy
import math
import os
import sys
from collections import OrderedDict
from io import IncrementalNewlineDecoder

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.container import ModuleList

sys.path.append('..')
import random

from util import make_divisible

from .dynamic_layers import *

class Transform_Net(nn.Module):
    def __init__(self):
        super(Transform_Net, self).__init__()

        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x

class Transform_Net_Tiny(nn.Module):
    def __init__(self):
        super(Transform_Net_Tiny, self).__init__()
        self.k = 3

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm1d(512)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 32, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(64, 512, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(512, 64, bias=False)
        self.bn3 = nn.BatchNorm1d(64)
        self.linear2 = nn.Linear(64, 32, bias=False)
        self.bn4 = nn.BatchNorm1d(32)

        self.transform = nn.Linear(32, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature      # (batch_size, 2*num_dims, num_points, k)


class DGCNN_partseg(nn.Module):
    def __init__(self, config :dict ,use_tiny_transform = False):
        super(DGCNN_partseg, self).__init__()
        self.config = config
        self.k = config['k']
        if use_tiny_transform == True:
            self.transform_net = Transform_Net_Tiny()
        else:
            self.transform_net = Transform_Net()

        self.ori_emb_dims = config['emb_dims']
        self.encoder = nn.ModuleList()
        total_dims = 0
        for block in config['encoder']:
            layer = nn.Sequential()
            layer.add_module('conv',DynamicConv2d(block['in_features'], block['out_features'], kernel_size=1, bias=False))
            if block['bn'] == True:
                layer.add_module('bn', DynamicBatchNorm2d(block['out_features']))
            layer.add_module('relu', nn.LeakyReLU(negative_slope=0.2,inplace=True))

            if(block.get('conv2',False)):
                layer.add_module('conv2',DynamicConv2d(block['out_features'], block['out_features'], kernel_size=1, bias=False))
                if block['bn'] == True:
                    layer.add_module('bn2', DynamicBatchNorm2d(block['out_features']))
                    layer.add_module('relu2', nn.LeakyReLU(negative_slope=0.2,inplace=True))

            total_dims +=block['out_features']
            self.encoder.append(layer)

        self.ori_dims =[block['out_features'] for block in config['encoder']]
        self.ori_dims = [sum(self.ori_dims[ : i]) for i in range(len(self.ori_dims))]
        # print(self.ori_dims)


        self.conv_x = nn.Sequential(OrderedDict([
            ('conv',DynamicConv1d(total_dims, config['emb_dims'], kernel_size=1, bias=False)),
            ('bn',DynamicBatchNorm1d(config['emb_dims'])),
            ('relu',nn.LeakyReLU(negative_slope=0.2))])
        )

        self.conv_labels = nn.Sequential(OrderedDict([
            ('conv',nn.Conv1d(16, 64, kernel_size=1, bias=False)),
            ('bn',nn.BatchNorm1d(64)),
            ('relu',nn.LeakyReLU(negative_slope=0.2))])
        )

 
        out_features=config['emb_dims']+total_dims+64

        self.ori_dims_decoder =[config['emb_dims']]+[block['out_features'] for block in config['encoder']]+[64]
        self.ori_dims_decoder = [sum(self.ori_dims_decoder[ : i]) for i in range(len(self.ori_dims_decoder))]

        self.decoder = nn.ModuleList()
        for block in config['decoder']:
            layer = nn.Sequential()
            layer.add_module('conv',DynamicConv1d(out_features,block['max_out_features'],kernel_size=1, bias=block['bias']))
            if block['bn'] == True:
                layer.add_module('bn', DynamicBatchNorm1d(block['max_out_features']))     
            if block['relu'] == True:
                layer.add_module('relu', nn.LeakyReLU(negative_slope=0.2,inplace=True))
            if block['dp'] == True:
                layer.add_module('dp', nn.Dropout(p=block['dropout_rate']))                      
            out_features = block['max_out_features']
            self.decoder.append(layer)



    def forward(self, x ,labels):
        batch_size = x.size(0)
        num_points = x.size(2)

        #transform
        x1 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x1)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        #encoder
        sample_dims=[]
        for i,block in enumerate(self.encoder):
            if self.config['encoder'][i]['activate'] == False:
                continue
            #set layer-wise k, otherwise use self.k
            k = self.config['encoder'][i].get('k',self.k)
            x = get_graph_feature(x, k=k)  # (batch_size, 2*num_dims, num_points, k)
            x = block(x)
            x = x.max(dim=-1, keepdim=False)[0] # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
            if i==0:
                x0 = x
            else:
                x0 = torch.cat((x0,x), dim=1)
            sample_dims.append(x.shape[1])

        #embaddings
        if isinstance(self.conv_x[0],DynamicConv1d):
            self.conv_x[0].set_dim(sample_dims,self.ori_dims)


        x = self.conv_x(x0)                              # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]              # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        labels = labels.view(batch_size, -1, 1)         # (batch_size, num_categoties, 1)
        labels = self.conv_labels(labels)               # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, labels), dim=1)               # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)                  # (batch_size, 1088, num_points)

        x = torch.cat((x, x0), dim=1)                   # (batch_size, 1088+64*3, num_points)


        sample_dims=[1024]+sample_dims+[64]

        if isinstance(self.decoder[0][0],DynamicConv1d):
            self.decoder[0][0].set_dim(sample_dims,self.ori_dims_decoder)

        #decoders
        for i,block in enumerate(self.decoder):
            x = block(x)


        return x


    def get_config(self):
        config = copy.deepcopy(self.config)
        return config

    def set_active_subnet(self,configs,stage):
        self.configs=configs

        sample_settings = stage.split('|')
        config_encoder,config_linear,k = configs
        # if 'depth' in sample_settings:
        #     for i,conf in enumerate(self.config['encoder']):
        #         if i >= depth:
        #             conf['activate']=False
        #         else:
        #             conf['activate']=True
        if 'encoder' in sample_settings:
            for i,w in enumerate(config_encoder):
                self.encoder[i].conv.active_out_channel=w
                if self.config['encoder'][i].get('conv2',False):
                    self.encoder[i].conv2.active_out_channel=w

                self.config['encoder'][i]['active_out_channel']=w

        if 'decoder' in sample_settings:
            for i,w in enumerate(config_linear):
                self.decoder[i].conv.active_out_channel=w
                self.config['decoder'][i]['active_out_features']=w

        if 'k' in sample_settings:
            for i,conf in enumerate(self.config['encoder']):
                conf['k']=k[i]


        return

        

    def sample_active_subnet(self,stage):
        if self.config['size']=='large':
            e1 = make_divisible(random.randint(48,64),8)
            e2 = make_divisible(random.randint(48,64),8)  
            e3 = make_divisible(random.randint(48,64),8)


            w1 = make_divisible(random.randint(128,256),8)
            w2 = make_divisible(random.randint(128,256),8)
            w3 = make_divisible(random.randint(64,128),8)

            k1 = random.randint(10,80)
            k2 = random.randint(10,80)
            k3 = random.randint(10,80)


            configs = ([e1,e2,e3],[w1,w2,w3],[k1,k2,k3])

        self.set_active_subnet(configs,stage)

        return configs






