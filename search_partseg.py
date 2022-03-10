#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import ShapeNetPart
#from model import DGCNN_partseg
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream, cross_entropy_loss_with_soft_target
import sklearn.metrics as metrics
import random
from models.dynamic_dgcnn_partseg import DGCNN_partseg
from models.dgcnn_partseg_test import DGCNN_partseg as DGCNN_teacher
from models.networks import partseg_config_base

seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]


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
    os.system('cp main_partseg.py checkpoints'+'/'+args.exp_name+'/'+'main_partseg.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp ./models/dynamic_dgcnn_partseg.py checkpoints' + '/' + args.exp_name + '/' + 'dynamic_dgcnn_partseg.py.backup')


def calculate_shape_IoU(pred_np, seg_np, label, class_choice):
    label = label.squeeze()
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious


def test(args, io):
    test_loader = DataLoader(ShapeNetPart(partition='test', num_points=args.num_points, class_choice=args.class_choice),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(ShapeNetPart(partition='val', num_points=args.num_points, class_choice=args.class_choice),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)


    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    seg_num_all = test_loader.dataset.seg_num_all
    seg_start_index = test_loader.dataset.seg_start_index
    model_args = partseg_config_base(seg_num_all)
    if args.model == 'dgcnn':
        model = DGCNN_partseg(model_args,args.use_tiny_transform).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))

    total_iou = []
    while len(total_iou)<648 :
        model.load_state_dict(torch.load(args.model_path))
        model.module.sample_active_subnet(stage=args.stage)
        io.cprint(str(args.stage)+str(model.module.configs))
        model = model.eval()

        test_acc = 0.0
        count = 0.0
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        test_label_seg = []
        for data, label, seg in test_loader:
            seg = seg - seg_start_index
            label_one_hot = np.zeros((label.shape[0], 16))
            for idx in range(label.shape[0]):
                label_one_hot[idx, label[idx]] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
            data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            with torch.no_grad():
                seg_pred = model(data, label_one_hot)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            pred = seg_pred.max(dim=2)[1]
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
            test_label_seg.append(label.reshape(-1))
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_label_seg = np.concatenate(test_label_seg)
        test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)
        outstr = 'Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (test_acc,
                                                                                avg_per_class_acc,
                                                                                np.mean(test_ious))
        io.cprint(outstr)
        m_iou = np.mean(test_ious)
        total_iou.append(m_iou)

        


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--use_tiny_transform', type=bool,  default=False,
                        help='use use tiny transform net')
    parser.add_argument('--dataset', type=str, default='shapenetpart', metavar='N',
                        choices=['shapenetpart'])
    parser.add_argument('--class_choice', type=str, default=None, metavar='N',
                        choices=['airplane', 'bag', 'cap', 'car', 'chair',
                                 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 
                                 'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table'])

    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')

    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--pretrain', type=bool,  default=False,
                        help='use pretrain model')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')

    parser.add_argument('--stage', type=str, default='k|encoder|decoder',
                        help='search stage : to sample k|encoder|decoder')

    parser.add_argument('--no_bn', type=bool, default=False,
                        help='enables bn retrain for 1 epoch')

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
