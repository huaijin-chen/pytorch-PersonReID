#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by Chao CHEN (chaochancs@gmail.com)
# Created On: 2017-06-08
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from .l2normalize import L2Normalize
import numpy as np
 
class View(nn.Module):
    def __init__(self, B, N):
        super(View, self).__init__()
        self.B = B
        self.N = N
    def forward(self, x):
        x = x.view(self.B, self.N)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feat_model = nn.Sequential(OrderedDict([
            # conv1
            ('conv1', nn.Conv2d( 3, 16, 3, 1, 1, bias=False)),
            ('bn1', nn.BatchNorm2d(16)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(2, 2)),
            # conv2
            ('conv2', nn.Conv2d(16, 32, 3, 1, 1, bias=False)),
            ('bn2', nn.BatchNorm2d(32)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(2, 2)),
            # conv3
            ('conv3', nn.Conv2d(32, 64, 3, 1, 1, bias=False)),
            ('bn3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU()),
            ('pool3', nn.MaxPool2d(2, 2)),
            # conv4
            ('conv4', nn.Conv2d(64, 128, 3, 1, 1, bias=False)),
            ('bn4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU()),
            ('pool4', nn.MaxPool2d(2, 2)),
            # conv5
            ('conv5', nn.Conv2d(128, 256, 3, 1, 1, bias=False)),
            ('bn5', nn.BatchNorm2d(256)),
            ('relu5', nn.ReLU()),
            ('pool5', nn.MaxPool2d(2, 2)),
            # conv6
            #('conv6',nn.Conv2d(256,512,3,1,1,bias=False)),
            #('bn6',nn.BatchNorm2d(512)),
            #('relu6',nn.ReLU()),
            #('pool6',nn.MaxPool2d(2,2)),
        
            ('reshape', View(-1, 2*4*256)),
            ('fc1', nn.Linear(2048, 1024)),#add fc level
            ('relu6', nn.ReLU()),
            ('fc2', nn.Linear(1024, 512)),
            ('l2normal', L2Normalize())
            #('relu7', nn.ReLU()),
        ]))

    def load_weights(self,path):
        buf = np.fromfile(path, dtype = np.float32)
        start = 4
        start = load_conv_bn(buf, start, \
                self.feat_model[0], self.feat_model[1])
        start = load_conv_bn(buf, start, \
                self.feat_model[4], self.feat_model[5])
        start = load_conv_bn(buf, start, \
                self.feat_model[8], self.feat_model[9])
        start = load_conv_bn(buf, start, \
                self.feat_model[12], self.feat_model[13])
        start = load_conv_bn(buf, start,\
                self.feat_model[16], self.feat_model[17])

    def load_full_weights(self, model_path):
        buf = np.fromfile(model_path, dtype = np.float32)
        start = 4
        start = load_conv_bn(buf, start, \
                self.feat_model[0], self.feat_model[1])
        start = load_conv_bn(buf, start, \
                self.feat_model[4], self.feat_model[5])
        start = load_conv_bn(buf, start, \
                self.feat_model[8], self.feat_model[9])
        start = load_conv_bn(buf, start, \
                self.feat_model[12], self.feat_model[13])
        start = load_conv_bn(buf, start,\
                self.feat_model[16], self.feat_model[17])

        # fc1 -21
        num_w = self.feat_model[21].weight.numel()
        num_b = self.feat_model[21].bias.numel()

        self.feat_model[21].bias.data.copy_(
                torch.from_numpy(buf[start:start+num_b]))

        start = start + num_b
        self.feat_model[21].weight.data.copy_(
                torch.from_numpy(buf[start:start+num_w]))
        start = start + num_w          

        # fc2 - 23
        num_w = self.feat_model[23].weight.numel()
        num_b = self.feat_model[23].bias.numel()

        self.feat_model[23].bias.data.copy_(
                torch.from_numpy(buf[start:start+num_b]))
        start = start + num_b
        self.feat_model[23].weight.data.copy_(
                torch.from_numpy(buf[start:start+num_w]))
        start = start + num_w          

    def forward(self, x):
        x = self.feat_model(x)
        return x

#train for classification
class Net_cls(nn.Module):
    def __init__(self):
        super(Net_cls,self).__init__()
        self.cls_model = nn.Sequential(OrderedDict([
        ('fc4',nn.Linear(1024,512)),
        ('relu',nn.ReLU()),
        ('fc5',nn.Linear(512,2)),
        ('log_softmax',nn.LogSoftmax()),

        ]))
   
    def forward(self,x):
        x = self.cls_model(x)
        return x

def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    #print(num_w,num_b)
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]))
    start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_b])) 
    start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start+num_b]))
    start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start+num_b])) 
    start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]))
    start = start + num_w 
    return start


def save_conv_bn(fp, conv_model, bn_model):
    #print(bn_model.bias.data)
    bn_model.bias.data.cpu().numpy().tofile(fp)
    bn_model.weight.data.cpu().numpy().tofile(fp)
    bn_model.running_mean.cpu().numpy().tofile(fp)
    bn_model.running_var.cpu().numpy().tofile(fp)
    conv_model.weight.data.cpu().numpy().tofile(fp)

def save_conv(fp, conv_model):
    conv_model.bias.data.cpu().numpy().tofile(fp)
    conv_model.weight.data.cpu().numpy().tofile(fp)

def save_weights(selfmodel, outfile, cutoff):
    ind = 0
    fp = open(outfile, 'wb')
    header = torch.IntTensor([0,0,0,0])
    header[3] = 0
    header.numpy().tofile(fp)
    for blockId in range(0, cutoff):
        block = selfmodel[blockId]
        #print(block.__class__.__name__)
        if block.__class__.__name__ == 'Conv2d':
            if  selfmodel[blockId+1].__class__.__name__ == 'BatchNorm2d':
                save_conv_bn(fp, selfmodel[blockId], selfmodel[blockId+1])
                ind = ind + 2
            else:
                save_conv(fp, selfmodel[blockId])
                ind = ind+1
        elif block.__class__.__name__ == 'MaxPool2d':
            ind = ind+1
        elif block.__class__.__name__ == 'View':
            ind = ind+1
        elif block.__class__.__name__ == 'Linear':
            save_conv(fp, selfmodel[blockId])
            ind = ind+1
        else:
            layer='unknown type '
    fp.close()


if __name__ == '__main__':
    model = Net()
    model.load_full_weights('data/tiny-yolo.weights')
    save_weights(model.feat_model, 'cc.weights', len(model.feat_model._modules))

