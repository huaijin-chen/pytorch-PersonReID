#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by Chao CHEN (chaochancs@gmail.com)
# Created On: 2017-06-08
# --------------------------------------------------------
import torch
from torchvision import datasets, transforms
import dataset.dataset as dataset
import time

def logging(message):
    print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.pos = 0
        self.neg = 0
    def update(self, val, n=1,pos=0.0,neg=0.0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.pos += pos
        self.neg += neg


def data_loader(image_root, data_list, shuffle=True, batch_size=64, workers=20, is_cuda=True):
    kwargs = {'num_workers': workers, 'pin_memory': True} if is_cuda else {}
    data_loader = torch.utils.data.DataLoader(
            dataset.listDataset(
                image_root, 
                data_list, 
                shuffle, 
                transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,),(0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    return data_loader

def read_data_cfg(datacfg):
    options = dict()
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key,value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    return options


def adjust_learning_rate(optimizer, epoch, epoch_step, learning_rate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_step = 0.1
    lr = learning_rate * (lr_step ** (epoch // epoch_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if epoch % epoch_step == 0:
        logging('lr = %f' % (lr))
