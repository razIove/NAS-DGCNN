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


def partseg_config_large(k=40,emb_dims=1024,seg_num_all=50):
    return {
            'name': 'partseg_large',
            'k': k,
            'size': 'large',
            'encoder': [
                {
                    'name': "encoder1",
                    'in_features': 6,
                    'out_features': 64,
                    'bn': True,
                    'activate': True,
                    'conv2': True,
                },
                {
                    'name': "encoder2",
                    'in_features': 128,
                    'out_features': 64,
                    'bn': True,
                    'activate': True,
                    'conv2': True,
                },
                {
                    'name': "encoder3",
                    'in_features': 128,
                    'out_features': 64,
                    'bn': True,
                    'activate': True,
                },

            ],
            'emb_dims': emb_dims,
            'decoder': [{
                'name': 'conv1d1',
                'max_out_features': 256,
                'bias': False,
                'bn': True,
                'relu': True,
                'dp': True,
                'dropout_rate': 0.5,
                },
                {
                'name': 'conv1d2',
                'max_out_features': 256,
                'bias': False,
                'bn': True,
                'relu': True,
                'dp': True,
                'dropout_rate': 0.5,
                },        
                {
                'name': 'conv1d3',
                'max_out_features': 128,
                'bias': False,
                'bn': True,
                'relu': True,
                'dp': False,
                'dropout_rate': 0.5,
                },
                {
                'name': 'conv1d4',
                'max_out_features': seg_num_all,
                'bias': False,
                'bn': False,
                'relu': False,
                'dp': False,
                'dropout_rate': 0.5,
                },                    
            ],
        }

def partseg_config_teacher(k=40,emb_dims=1024,seg_num_all=50):
    return {
            'name': 'partseg_large',
            'k': k,
            'size': 'large',
            'encoder': [
                {
                    'name': "encoder1",
                    'in_features': 6,
                    'out_features': 64,
                    'bn': True,
                    'activate': True,
                    'conv2': True,
                    'k':43,
                },
                {
                    'name': "encoder2",
                    'in_features': 128,
                    'out_features': 64,
                    'bn': True,
                    'activate': True,
                    'conv2': True,
                    'k':56,
                },
                {
                    'name': "encoder3",
                    'in_features': 128,
                    'out_features': 64,
                    'bn': True,
                    'activate': True,
                    'k':37,
                },

            ],
            'emb_dims': emb_dims,
            'decoder': [{
                'name': 'conv1d1',
                'max_out_features': 256,
                'bias': False,
                'bn': True,
                'relu': True,
                'dp': True,
                'dropout_rate': 0.5,
                },
                {
                'name': 'conv1d2',
                'max_out_features': 256,
                'bias': False,
                'bn': True,
                'relu': True,
                'dp': True,
                'dropout_rate': 0.5,
                },        
                {
                'name': 'conv1d3',
                'max_out_features': 128,
                'bias': False,
                'bn': True,
                'relu': True,
                'dp': False,
                'dropout_rate': 0.5,
                },
                {
                'name': 'conv1d4',
                'max_out_features': seg_num_all,
                'bias': False,
                'bn': False,
                'relu': False,
                'dp': False,
                'dropout_rate': 0.5,
                },                    
            ],
        }

def partseg_config_test(k=40,emb_dims=1024,seg_num_all=50):
    return {
            'name': 'partseg_large',
            'k': k,
            'size': 'large',
            'encoder': [
                {
                    'name': "encoder1",
                    'in_features': 6,
                    'out_features': 56,
                    'bn': True,
                    'activate': True,
                    'conv2': True,
                    'k':37,
                },
                {
                    'name': "encoder2",
                    'in_features': 112,
                    'out_features': 56,
                    'bn': True,
                    'activate': True,
                    'conv2': True,
                     'k':44,
                },
                {
                    'name': "encoder3",
                    'in_features': 112,
                    'out_features': 56,
                    'bn': True,
                    'activate': True,
                     'k':22,
                },

            ],
            'emb_dims': emb_dims,
            'decoder': [{
                'name': 'conv1d1',
                'max_out_features': 160,
                'bias': False,
                'bn': True,
                'relu': True,
                'dp': True,
                'dropout_rate': 0.5,
                },
                {
                'name': 'conv1d2',
                'max_out_features': 216,
                'bias': False,
                'bn': True,
                'relu': True,
                'dp': True,
                'dropout_rate': 0.5,
                },        
                {
                'name': 'conv1d3',
                'max_out_features': 120,
                'bias': False,
                'bn': True,
                'relu': True,
                'dp': False,
                'dropout_rate': 0.5,
                },
                {
                'name': 'conv1d4',
                'max_out_features': seg_num_all,
                'bias': False,
                'bn': False,
                'relu': False,
                'dp': False,
                'dropout_rate': 0.5,
                },                    
            ],
        }

def cls_config_base(k=20,emb_dims=1024,output_channels=40):
    return {
            'name': 'DDGCNNCLS',
            'k': k,
            'size': 'base',
            'encoder': [
                {
                    'name': "dgcnn",
                    'in_features': 6,
                    'out_features': 64,
                    'bn': True,
                    'activate': True,
                },
                {
                    'name': "dgcnn",
                    'in_features': 128,
                    'out_features': 64,
                    'bn': True,
                    'activate': True,
                },
                {
                    'name': "dgcnn",
                    'in_features': 128,
                    'out_features': 128,
                    'bn': True,
                    'activate': True,
                },
                {
                    'name': "dgcnn",
                    'in_features': 256,
                    'out_features': 256,
                    'bn': True,
                    'activate': True,
                },
            ],
            'emb_dims': emb_dims,
            'decoder': [{
                'name': 'linear',
                'max_out_features': 512,
                'bias': False,
                'bn': True,
                'dp': True,
                'dropout_rate': 0.5,
                },
                {
                'name': 'linear',
                'max_out_features': 256,
                'bias': True,
                'bn': True,
                'dp': True,
                'dropout_rate': 0.5,
                },              
            ],
            'output_channels' : output_channels,
        }


def cls_config_test(k=20,emb_dims=1024,output_channels=40):
    return {
            'name': 'DDGCNNCLS',
            'k': k,
            'size': 'base',
            'encoder': [
                {
                    'name': "dgcnn",
                    'in_features': 6,
                    'out_features': 64,
                    'bn': True,
                    'activate': True,
                    'k':26,
                },
                {
                    'name': "dgcnn",
                    'in_features': 128,
                    'out_features': 56,
                    'bn': True,
                    'activate': True,
                    'k':22,
                },
                                {
                    'name': "dgcnn",
                    'in_features': 112,
                    'out_features': 88,
                    'bn': True,
                    'activate': True,
                    'k':13,
                },
                {
                    'name': "dgcnn",
                    'in_features': 176,
                    'out_features': 120,
                    'bn': True,
                    'activate': True,
                    'k':25,
                },

            ],
            'emb_dims': emb_dims,
            'decoder': [{
                'name': 'linear',
                'max_out_features': 144,
                'bias': False,
                'bn': True,
                'dp': True,
                'dropout_rate': 0.5,
                },
                {
                'name': 'linear',
                'max_out_features': 208,
                'bias': True,
                'bn': True,
                'dp': True,
                'dropout_rate': 0.5,
                },              
            ],
            'output_channels' : output_channels,
        }
def cls_config_tiny(emb_dims=320,output_channels=40):
    return {
			'name': 'DDGCNNCLS',
            'k': 20,
            'size': 'tiny',
			'encoder': [
				{
                    'name': "dgcnn",
                    'in_features': 6,
                    'out_features': 48,
                    'bn': True,
                    'activate': True,
                },
				{
                    'name': "dgcnn",
                    'in_features': 96,
                    'out_features': 96,
                    'bn': True,
                    'activate': True,
                },
                {
                    'name': "dgcnn",
                    'in_features': 192,
                    'out_features': 96,
                    'bn': True,
                    'activate': True,
                },

			],
			'emb_dims': emb_dims,
			'decoder': [{
                'name': 'linear',
                'max_out_features': 256,
                'bias': True,
                'bn': True,
                'dp': True,
                'dropout_rate': 0.5,
                },    
            ],
            'output_channels' : output_channels,
		}
def get_config_seg(emb_dims=512,output_channels=40):
    return {
            'name': 'DDGCNNCLS',
            'k': 20,
            'encoder': [
                {
                    'name': "dgcnn",
                    'in_features': 6,
                    'out_features': 32,
                    'bn': True,
                },
                {
                    'name': "dgcnn",
                    'in_features': 64,
                    'out_features': 32,
                    'bn': True,
                },
                {
                    'name': "dgcnn",
                    'in_features': 64,
                    'out_features': 32,
                    'bn': True,
                },
                {
                    'name': "dgcnn",
                    'in_features': 64,
                    'out_features': 32,
                    'bn': True,
                },
                {
                    'name': "dgcnn",
                    'in_features': 64,
                    'out_features': 32,
                    'bn': True,
                },
            ],
            'emb_dims': emb_dims,
            'decoder': [ 
                {
                'name': 'linear',
                'out_features': 256,
                'bias': True,
                'bn': True,
                'dp': True,
                'dropout_rate': 0.5,
                },    
            ],
            'output_channels' : output_channels,
        }
def get_rand_config(output_channels=40):

    conf = {
        'name': 'DDGCNNCLS',
        'output_channels' : output_channels,
    }
    conf['k'] = 20#make_divisible(random.randint(1,40),5)

    encoder_blocks = []
    encoder_layer_num = random.randint(3,5)
    input_dim = 3
    total=0
    for i in range(encoder_layer_num):
        output_dim = make_divisible(random.randint(input_dim-63,input_dim+128*i),64)
        block = {
                    'name': "dgcnn"+str(i),
                    'in_features': input_dim*2,
                    'out_features': output_dim,
                    'bn': True,
        }
        input_dim = output_dim
        total+=output_dim
        encoder_blocks.append(block)
    conf['encoder'] = encoder_blocks
    conf['emb_dims'] = make_divisible(random.randint(1,2048),512)
    
    conf['decoder'] = [{
                'name': 'linear',
                'out_features': make_divisible(random.randint(1,1024),256),
                'bias': False,
                'bn': True,
                'dp': True,
                'dropout_rate': 0.5,
                },
                {
                'name': 'linear',
                'out_features': make_divisible(random.randint(1,1024),256),
                'bias': True,
                'bn': True,
                'dp': True,
                'dropout_rate': 0.5,
                },              
            ]
    return conf
