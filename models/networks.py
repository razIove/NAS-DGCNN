import copy
import math
import os
import sys

import numpy as np

sys.path.append('..')
import random



def partseg_config_base(k=40,emb_dims=1024,seg_num_all=50):
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


