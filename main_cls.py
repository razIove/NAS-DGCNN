#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from models.networks import *
from ori_model import PointNet
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


def load_model(path,set_stage=True):
    params = torch.load(path)
    model = DGCNN_cls(params['config'])
    model = nn.DataParallel(model)
    model.load_state_dict(params['state_dict'])
    if set_stage:
        model.module.set_active_subnet(configs=params['sample_configs'],stage=params['stage'])
        print('set model %s in %s.'%(str(params['sample_configs']),str(params['stage'])))
    return model

def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
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

    print(str(model))

    model = nn.DataParallel(model)
    if args.pretrain :
        print("load pretrained model from ",args.model_path,"!")
        model.load_state_dict(torch.load(args.model_path))
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    #optimizer & scheduler
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.7)
    criterion = cal_loss

    #knowledge distillation
    if args.kd_ratio > 0:
        print("load teacher model from ",args.kd_model_path,"!")
        teacher_model = load_model(args.kd_model_path)
        teacher_model = teacher_model.to(device)

    best_test_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

 
            model.module.sample_active_subnet(args.stage)

            opt.zero_grad()
            logits = model(data)

            # soft target
            if args.kd_ratio > 0:
                #args.teacher_model.train() eval mode performs better
                teacher_model.eval()
                with torch.no_grad():
                    soft_logits = teacher_model(data).detach()
                    soft_label = F.softmax(soft_logits, dim=1)


            if args.kd_ratio == 0:
                loss = criterion(logits, label)
            else:
                if args.kd_type == 'ce':
                    kd_loss = cross_entropy_loss_with_soft_target(logits, soft_label)
                else:   #TODO
                    kd_loss = F.mse_loss(logits, soft_logits)
                loss = args.kd_ratio * kd_loss + criterion(logits, label)

            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        learning_rate = opt.param_groups[0]['lr']
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint('EPOCH #{}  lr = {}'.format(epoch, learning_rate))                                        
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        if len(args.stage)>0:
            io.cprint(str(args.stage)+': '+str(model.module.configs))
            #model.module.set_full_subnet()
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            with torch.no_grad():
                logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
            io.cprint('--------------  best acc:%.3f  ------------'%(best_test_acc))
        torch.save(model.state_dict(), 'checkpoints/%s/models/model_last.t7' % args.exp_name)


def test(args, io):
  
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = load_model(path=args.model_path).to(device)
    else:
        raise Exception("Not implemented")

    model = model.eval()

    for _ in range(1):


        #io.cprint(str(model.module.configs))

        
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
        outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
        io.cprint(outstr)

        macs, params, macs_str, params_str = profile_macs_params(model.module.get_active_subnet().cpu())
        io.cprint(f"MACs {macs_str} Params {params_str}")




if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--model_size', type=str, default='base', metavar='N',
                        choices=['base', 'tiny'],
                        help='search space to use, [base, tiny]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--pretrain', type=bool,  default=False,
                        help='use pretrain model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of default nearest neighbors to use')

    parser.add_argument('--kd_ratio', type=float, default=1.0,
                        help='>0:use teacher model default:0.0 or 1.0')
    parser.add_argument('--kd_model_path', type=str, default='the_best_model.pth', metavar='N',
                        help='Pretrained teachet model path')                        
    parser.add_argument('--kd_type', type=str, default='ce', metavar='N',
                        choices=['ce', 'tbd'],
                        help='loss for kd')

    parser.add_argument('--model_path', type=str, default='pretrain_model.t7', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--stage', type=str, default='',
                        help='Trainning stage : to sample k|encoder|decoder|depth')
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

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
