#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main_cls.py
@Time: 2018/10/13 10:39 PM

Modified by 
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2019/12/30 9:32 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import ModelNet40
#from model import PointNet, DGCNN_cls
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream, cross_entropy_loss_with_soft_target, profile_macs_params
import sklearn.metrics as metrics
from models.dynamic_dgcnn_cls import *
from validation import ModelNet40Val
from models.networks import *

def set_seed_torch(seed=0): 
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True 
    random.seed(seed)

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main_cls.py checkpoints'+'/'+args.exp_name+'/'+'main_cls.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')
    os.system('cp ./models/dynamic_dgcnn_cls.py checkpoints' + '/' + args.exp_name + '/' + 'dynamic_dgcnn_cls.py.backup')
    os.system('cp ./models/dynamic_layers.py checkpoints' + '/' + args.exp_name + '/' + 'dynamic_layers.py.backup')

# def save_model(args,model,name=''):
#     import time
#     model.eval()
#     device = torch.device("cuda" if args.cuda else "cpu")
#     inputs = torch.rand(1,3,1024).to(device)
#     model(inputs)
#     model2 = model.module.get_active_subnet().to(device)
#     if len(name)>0:
#         torch.save(model2, 'checkpoints/%s/models/%s.pth' % (args.exp_name,name))
#     else:
#         torch.save(model2, 'checkpoints/%s/models/%s.pth' % (args.exp_name,time.strftime("%m%d_%H%M%S", time.localtime()) ))
    

def test(args, io):
  
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.test_batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    # val_loader = DataLoader(ModelNet40Val(partition='val', num_points=args.num_points),
    #                          batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    model_args = cls_config_base()
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        if args.model_size == 'base':
            model_args = cls_config_base()
        elif args.model_size == 'tiny':
            model_args = cls_config_tiny()
        else:
            raise Exception("Not implemented")
        io.cprint(str(model_args))        
        model = DGCNN_cls(model_args).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
 

    total_acc = []
    while len(total_acc)<args.num :

        model.load_state_dict(torch.load(args.model_path))
        model.module.sample_active_subnet(stage=args.stage)
        model = model.eval()
        
        rand_input = torch.rand(1,3,1024).to(device)
        model(rand_input)
        macs, params, macs_str, params_str = profile_macs_params(model.module.get_active_subnet().cpu())
        
        # if not args.params/2 < params < args.params :#400k
        #     continue
        
        io.cprint(f"MACs {macs_str} Params {params_str}")
        io.cprint(str(args.stage)+str(model.module.configs))
        
        #bn retrain
        if(args.no_bn == False):
            with  torch.no_grad():
                count=0
                model.train()
                for data, label in train_loader:
                    data, label = data.to(device), label.to(device).squeeze()
                    data = data.permute(0, 2, 1)
                    out = model(data)
                    count+=1
        
        test_acc = 0.0
        count = 0.0
        test_true = []
        test_pred = []
        model.eval()
        for data, label in test_loader:

            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            with  torch.no_grad():
                logits = model(data)
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        # io.cprint(f"[{w1},{w2}]")
        outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
        io.cprint(outstr)
        total_acc.append(test_acc)

        #save_model(args,model,'%.3f_%s_%s.pth'%(test_acc,macs_str,params_str))
        torch.save({'state_dict':model.state_dict(),'sample_configs':model.module.configs,'stage':args.stage, 'config':model.module.configs}
            , 'checkpoints/%s/models/%.3f_%s_%s.pth' % (args.exp_name,test_acc,macs_str,params_str))

    io.cprint('Total models: %d, avg acc:%.3f, max acc:%.3f, min acc:%.3f'%(len(total_acc),np.mean(total_acc),max(total_acc),min(total_acc)))



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='eval', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')


    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    # parser.add_argument('--params', type=float, default=4e5, metavar='N',
    #                     help='Num of target params to search')
    parser.add_argument('--no_bn', type=bool, default=False,
                        help='set True to disable bn retrain')

    parser.add_argument('--model_path', type=str, default='/raid/user11/project/ofa_dgc/checkpoints/ofa_convlinear_ori/models/model.t7', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--stage', type=str, default='',
                        help='Trainning stage : to sample k|encoder|decoder|depth')
    parser.add_argument('--num', type=int, default=648, metavar='N',
                        help='Num of network architecture to sample')


    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    set_seed_torch(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')


    test(args, io)
