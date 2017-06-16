#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by Chao CHEN (chaochancs@gmail.com)
# Created On: 2017-06-13
# --------------------------------------------------------
import sys
sys.path.insert(0, '../')
import dataset.dataset as dataset
from module.symbols import Net, Net_cls, save_weights
from utils import AverageMeter, data_loader
from torch.autograd import Variable
import torch.nn.functional as F
import torch

def test(model, test_loader, epoch, margin, threshlod, is_cuda=True, log_interval=1000):
    model.eval()
    test_loss = AverageMeter()
    accuracy = 0
    num_p = 0
    total_num = 0
    batch_num = len(test_loader)
    for batch_idx, (data_a, data_p, data_n,target) in enumerate(test_loader):
        if is_cuda:
            data_a = data_a.cuda()
            data_p = data_p.cuda()
            data_n = data_n.cuda()
            target = target.cuda()

        data_a = Variable(data_a, volatile=True)
        data_p = Variable(data_p, volatile=True)
        data_n = Variable(data_n, volatile=True)
        target = Variable(target)

        out_a = model(data_a)
        out_p = model(data_p)
        out_n = model(data_n)

        loss = F.triplet_margin_loss(out_a,out_p,out_n, margin)

        dist1 = F.pairwise_distance(out_a,out_p)
        dist2 = F.pairwise_distance(out_a,out_n)
        #print('dist1', dist1)
        #print('dist2',dist2)
        #print('threshlod', threshlod)

        num = ((dist1 < threshlod).sum() + (dist2 > threshlod).sum()).data[0]
        num_p += num
        num_p = 1.0 * num_p
        total_num += data_a.size()[0] * 2
        #print('num--num_p -- total',  num, num_p , total_num)
        test_loss.update(loss.data[0])
        if (batch_idx + 1) % log_interval == 0:
            accuracy_tmp = num_p / total_num    
            print('Test- Epoch {:04d}\tbatch:{:06d}/{:06d}\tAccuracy:{:.04f}\tloss:{:06f}'\
                    .format(epoch, batch_idx+1, batch_num, accuracy_tmp, test_loss.avg))
            test_loss.reset()

    accuracy = num_p / total_num
    return accuracy

def best_test(model, _loader, model_path=None, is_cuda=True):
    if not model_path is None:
        model.load_full_weights(model_path)
        print('loaded model file: {:s}'.format(model_path))
    if is_cuda:
        model = model.cuda()
    model.eval()
    total_num = 0
    batch_num = len(_loader)
    for batch_idx, (data_a, data_p, data_n,target) in enumerate(_loader):
        if is_cuda:
            data_a = data_a.cuda()
            data_p = data_p.cuda()
            data_n = data_n.cuda()
            target = target.cuda()

        data_a = Variable(data_a, volatile=True)
        data_p = Variable(data_p, volatile=True)
        data_n = Variable(data_n, volatile=True)
        target = Variable(target)

        out_a = model(data_a)
        out_p = model(data_p)
        out_n =  model(data_n)
        current_d_a_p = F.pairwise_distance(out_a,out_p)
        current_d_a_n = F.pairwise_distance(out_a,out_n)
        if batch_idx == 0:
            d_a_p = current_d_a_p
            d_a_n = current_d_a_n
        else:
            d_a_p = torch.cat((d_a_p, current_d_a_p), 0)
            d_a_n = torch.cat((d_a_n, current_d_a_n), 0)
        total_num += 2*data_a.size()[0]

    mean_d_a_p = d_a_p.mean().data[0]
    mean_d_a_n = d_a_n.mean().data[0]
    start = min(mean_d_a_n, mean_d_a_p)
    end = max(mean_d_a_n, mean_d_a_p)
    best_thre = 0
    best_num = 0
    thre_step = 0.1

    for val in torch.arange(start, end+thre_step, thre_step):
        num = (((d_a_p <= val).float()).sum() + (d_a_n > val).float().sum()).data[0]
        #print(num, val)
        if num > best_num:
            best_num = num
            best_thre = val
    return best_thre, best_num/total_num, mean_d_a_p, mean_d_a_n
    
    

    


def evaluation(model_path, test_list):
    model = Net()
    model = model.cuda()
    model.load_full_weights(model_path)
    model.eval()
      
    image_root = '/path/to/ReId_data/image/'
    if False:
        train_list = 'data/debug.txt'
        test_list = '../data/test_debug.txt'
    else:
        train_list = 'data/train_all_5set.txt'
        test_list = '../data/test_all_5set.txt'
    test_loader = data_loader(image_root, test_list, shuffle=True, batch_size=32)
    margin = 1
    accuracy = test(model, test_loader, 0, margin, threshlod=10, log_interval=10)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = Net() 
    margin = 1.0
    model_path = 'path/to/model/file'
    test_list = '../data/open_val.txt'
    image_root = 'data/root'
    test_loader = data_loader(image_root, test_list, shuffle=True, batch_size=128)
    thre, acc, _, _ = best_test(model, test_loader, model_path, is_cuda=True)
    print('best_threshold : {:.03f}, best_accuracy:{:.03f}'.format(thre, acc))














