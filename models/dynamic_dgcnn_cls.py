
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


class DGCNN_cls(nn.Module):
    def __init__(self, config :dict):
        super(DGCNN_cls, self).__init__()
        # self.load_from_config(args)
        self.config = config
        self.k = config['k']
        self.ori_emb_dims = config['emb_dims']
        self.encoder = nn.ModuleList()
        total_dims = 0
        for block in config['encoder']:
            layer = nn.Sequential()
            layer.add_module('conv',DynamicConv2d(block['in_features'], block['out_features'], kernel_size=1, bias=False))
            if block['bn'] == True:
                layer.add_module('bn', DynamicBatchNorm2d(block['out_features']))
            layer.add_module('relu', nn.LeakyReLU(negative_slope=0.2))

            total_dims +=block['out_features']
            self.encoder.append(layer)

        self.ori_dims =[0]+[block['out_features'] for block in config['encoder']]
        self.ori_dims.pop()
        self.ori_dims = [sum(self.ori_dims[ : i + 1]) for i in range(len(self.ori_dims))]
        # print(self.ori_dims)


        self.conv = nn.Sequential(OrderedDict([
            ('conv',DynamicConv1d(total_dims, config['emb_dims'], kernel_size=1, bias=False)),
            ('bn',DynamicBatchNorm1d(config['emb_dims'])),
            ('relu',nn.LeakyReLU(negative_slope=0.2))])
        )

        out_features=config['emb_dims']*2

        self.decoder = nn.ModuleList()
        for block in config['decoder']:
            layer = nn.Sequential()
            layer.add_module('linear',DynamicLinear(out_features,block['max_out_features'],block['bias']))
            if block['bn'] == True:
                layer.add_module('bn', DynamicBatchNorm1d(block['max_out_features']))     
            layer.add_module('relu', nn.LeakyReLU(negative_slope=0.2))
            if block['dp'] == True:
                layer.add_module('dp', nn.Dropout(p=block['dropout_rate']))                      
            out_features = block['max_out_features']
            self.decoder.append(layer)

        self.classifier = DynamicLinear(out_features, config['output_channels'])


    def forward(self, x):
        batch_size = x.size(0)
        #encoder

        sample_dims=[]
        for i,block in enumerate(self.encoder):
            if self.config['encoder'][i]['activate'] == False:
                continue
            #set layer-wise k, otherwise use self.k
            k = self.config['encoder'][i].get('k',self.k)

            x = get_graph_feature(x, k=k)  # (batch_size, 2*num_dims, num_points, k)
            x = block(x)
            x = x.max(dim=-1, keepdim=False)[0]
            if i==0:
                x0 = x
            else:
                x0 = torch.cat((x0,x), dim=1)
            sample_dims.append(x.shape[1])
        if isinstance(self.conv[0],DynamicConv1d):
            self.conv[0].set_dim(sample_dims,self.ori_dims)

        x = self.conv(x0)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        if isinstance(self.decoder[0].linear,DynamicLinear):
            if self.config['emb_dims'] < self.ori_emb_dims:
                self.decoder[0].linear.active_in_features = list(range(self.config['emb_dims'])) \
                    +list(range(self.ori_emb_dims,self.ori_emb_dims+self.config['emb_dims']))
            else:
                self.decoder[0].linear.active_in_features = self.config['emb_dims']*2

        for i,block in enumerate(self.decoder):
            x = block(x)


        x = self.classifier(x)

        return x

    def load_from_config(self, config):
        return

    def get_config(self):
        config = copy.deepcopy(self.config)
  
        return config

    def set_active_subnet(self,configs,stage):
        self.configs=configs

        sample_settings = stage.split('|')
        config_encoder,config_linear,k,depth = configs
        if 'depth' in sample_settings:
            for i,conf in enumerate(self.config['encoder']):
                if i >= depth:
                    conf['activate']=False
                else:
                    conf['activate']=True
        if 'encoder' in sample_settings:
            for i,w in enumerate(config_encoder):
                self.encoder[i].conv.active_out_channel=w
                self.config['encoder'][i]['active_out_channel']=w

        if 'decoder' in sample_settings:
            # self.conv[0].active_out_channel = dims
            # self.config['emb_dims'] = dims
            for i,w in enumerate(config_linear):
                self.decoder[i].linear.active_out_features=w
                self.config['decoder'][i]['active_out_features']=w

        if 'k' in sample_settings:
            for i,conf in enumerate(self.config['encoder']):
                conf['k']=k[i]
            #self.k = k

        return

    def set_full_subnet(self):
        for i,block in enumerate(self.encoder):
            block.conv.active_out_channel = block.conv.max_out_channels

        for i,block in enumerate(self.decoder):
            block.linear.active_out_features = block.linear.max_out_features

        self.k=self.config['k']
        

    def sample_active_subnet(self,stage):
        if self.config['size']=='base':
            e1 = make_divisible(random.randint(32,64),8)
            e2 = make_divisible(random.randint(32,64),8)  
            e3 = make_divisible(random.randint(64,128),8)
            e4 = make_divisible(random.randint(64,256),8)

            # dims = make_divisible(random.randint(256,1024),128)
            w1 = make_divisible(random.randint(64,512),8)
            w2 = make_divisible(random.randint(32,256),8)

            k1 = random.randint(10,30)
            k2 = random.randint(10,30)
            k3 = random.randint(10,30)
            k4 = random.randint(10,30)
            depth = random.randint(2,4)

            configs = ([e1,e2,e3,e4],[w1,w2],[k1,k2,k3,k4],depth)
        else: #tiny
            
            e1 = make_divisible(random.randint(24,48),8)
            e2 = make_divisible(random.randint(48,96),8)  
            e3 = make_divisible(random.randint(48,96),8)

            w1 = make_divisible(random.randint(40,256),8)

            k = random.randint(10,25)
            depth = random.randint(2,3)

            configs = ([e1,e2,e3],[w1],k,depth)
        
        
        self.set_active_subnet(configs,stage)

        return configs

    #get a model without unused params
    def get_active_subnet(self, preserve_weight=True):
        model2=copy.deepcopy(self)
        in_dims=0
        in_features=3
        for i,block in enumerate(model2.encoder):
            if self.config['encoder'][i]['activate'] == False:
                continue
            block.conv = block.conv.get_active_layer(in_features*2,preserve_weight)
            in_features = self.encoder[i].conv.active_out_channel
            bn = nn.BatchNorm2d(in_features)
            copy_bn(bn,block.bn.bn)
            block.bn=bn
            in_dims+=in_features

        model2.encoder =nn.ModuleList(
            [model2.encoder[i] for i in range(len(model2.encoder)) if self.config['encoder'][i]['activate'] == True]
            )


        model2.conv.conv = model2.conv.conv.get_active_layer(in_dims,preserve_weight)
        bn = nn.BatchNorm1d(self.config['emb_dims'])
        copy_bn(bn,model2.conv.bn.bn)
        model2.conv.bn=bn
        in_features=self.config['emb_dims']*2
        for i,block in enumerate(model2.decoder):
            block.linear=block.linear.get_active_layer(in_features,preserve_weight)
            in_features = self.decoder[i].linear.active_out_features
            bn = nn.BatchNorm1d(in_features)
            copy_bn(bn,block.bn.bn)
            block.bn=bn
        model2.classifier = model2.classifier.get_active_layer(in_features,preserve_weight)

        # print(model2)
        return model2



from thop import clever_format, profile

if __name__ == "__main__":
    from networks import cls_config_base
    config = cls_config_base()
    print(config)
    times = 1
    for i in range(times):
        model = DGCNN_cls(config)
        # print(model)

        model = model.eval()
        input = torch.randn(1,3,1024)

        macs, params = profile(model, inputs=(input,))
        macs, params = clever_format([macs, params], "%.3f")
        print(macs,params)
