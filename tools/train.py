#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by Chao CHEN (chaochancs@gmail.com)
# Created On: 2017-06-15
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from module.symbols import Net, save_weights
from tools.utils import AverageMeter, logging
from tools.test import best_test
from module.TripletLoss import TripletMarginLoss  

def train_val(model, optimizer, train_loader, test_loader,
        epoch, margin=1.0, use_ohem=False, log_interval=100, test_interval=2000, is_cuda=True):
    loss = AverageMeter()
    batch_num = len(train_loader)
    for batch_idx, (data_a, data_p,data_n, target) in enumerate(train_loader):
        model.train()
        if is_cuda:
            data_a = data_a.cuda()
            data_p = data_p.cuda()
            data_n = data_n.cuda()
            #target = target.cuda()
        #print('data_size = ',data_a.size())
        #print(data_a)
        #print('-----------------------------------------')
        data_a = Variable(data_a)
        data_p = Variable(data_p) 
        data_n = Variable(data_n)
        target = Variable(target)

        optimizer.zero_grad()
        out_a = model(data_a)
        out_p = model(data_p) 
        out_n = model(data_n)

        triploss_layer = TripletMarginLoss(margin, use_ohem=use_ohem)
        trip_loss = triploss_layer(out_a, out_p, out_n)

        trip_loss.backward()  
        optimizer.step()

        loss.update(trip_loss.data[0])
        if (batch_idx+1) % log_interval == 0:
            logging('Train-Epoch:{:04d}\tbatch:{:06d}/{:06d}\tloss:{:.04f}'\
                    .format(epoch, batch_idx+1, batch_num, trip_loss.data[0]))
        if (batch_idx+1) % test_interval == 0:
            threshlod, accuracy , mean_d_a_p, mean_d_a_n = best_test(model, test_loader)
            logging('Test-T-A Epoch {:04d}-{:06d} accuracy: {:.04f} threshold: {:.05} ap_mean: {:.04f} an_mean: {:.04f}'
                    .format(epoch, batch_idx+1, accuracy, threshlod, mean_d_a_p, mean_d_a_n))
            cutoff = len(model.module.feat_model._modules)
            model_name = 'models/epoch_{:04d}-{:06d}_feat.weights'.format(epoch, batch_idx+1)
            save_weights(model.module.feat_model, model_name, cutoff)
            logging('save model: {:s}'.format(model_name))


