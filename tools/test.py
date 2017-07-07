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
import cv2
import uuid
import numpy as np
import os.path as osp

def combine_and_save(img1_path, img2_path, dist, output_dir, class_flag):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    h, w, c = img1.shape
    out_size = (h, w*2, c)
    out = np.zeros(out_size)
    out[:, 0:w, :]   = img1
    out[:, w:2*w, :] = img2
    uid = uuid.uuid4()
    out_name = osp.join(output_dir, '{:s}_{:.03f}_{:s}.jpg'.format(class_flag, dist.data[0],str(uid)))
    cv2.imwrite(out_name, out)
    #print out_name


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

        num = ((dist1 < threshlod).float().sum() + (dist2 > threshlod).float().sum()).data[0]
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

def test_vis(model, test_loader, model_path, threshlod,\
        margin=1.0, is_cuda=True, output_dir='output',is_visualization=True):
    if not model_path is None:
        model.load_full_weights(model_path)
        print('loaded model file: {:s}'.format(model_path))
    if is_cuda:
        model = model.cuda()
    model.eval()
    test_loss = AverageMeter()
    accuracy = 0
    num_p = 0
    total_num = 0
    batch_num = len(test_loader)
    for batch_idx, (data_a, data_p, data_n,target, img_paths) in enumerate(test_loader):
    #for batch_idx, (data_a, data_p, data_n, target) in enumerate(test_loader):
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
        batch_size = data_a.size()[0]
        pos_flag = (dist1 <= threshlod).float()
        neg_flag = (dist2 > threshlod).float()
        if is_visualization:
            for k in torch.arange(0, batch_size):
                k = int(k)
                if pos_flag[k].data[0] == 0:
                    combine_and_save(img_paths[0][k], img_paths[1][k], dist1[k], output_dir,  '1-1')
                if neg_flag[k].data[0] == 0:
                    combine_and_save(img_paths[0][k], img_paths[2][k], dist2[k], output_dir, '1-0')

        num = (pos_flag.sum() + neg_flag.sum()).data[0]
        
        print('{:f}, {:f}, {:f}'.format(num, pos_flag.sum().data[0], neg_flag.sum().data[0]))
        num_p += num
        total_num += data_a.size()[0] * 2
        print('num_p = {:f},  total = {:f}'.format(num_p, total_num))
        print('dist1 = {:f}, dist2 = {:f}'.format(dist1[0].data[0], dist2[0].data[0]))

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
    thre_step = 0.05

    for val in torch.arange(start, end+thre_step, thre_step):
        num = (((d_a_p <= val).float()).sum() + (d_a_n > val).float().sum()).data[0]
        #print(num, val)
        if num > best_num:
            best_num = num
            best_thre = val
    return best_thre, best_num/total_num, mean_d_a_p, mean_d_a_n
    
    
def visualization():
    pass
    


def evaluation():
    model_path = '/data/chenchao/reid_train/exp11.2/models/epoch_0004-000500_feat.weights'
    model = Net()
    model = model.cuda()
    model.load_full_weights(model_path)
    model.eval()
      
    test_list = '../data/val.txt'
    image_root = '/home/chenchao/ReID/'
    test_loader = data_loader(image_root, test_list, shuffle=True, batch_size=256)
    margin = 1.0
    #accuracy = test(model, test_loader, 0, margin, threshlod=0.18, log_interval=10)
    thre, acc, mean_1, mean_2 = best_test(model, test_loader, model_path)
    print('accuracy = {:f}'.format(acc))


def test_offline():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = Net() 
    margin = 1.0
    threshlod = 1.0
    model_path = '/data/chenchao/reid_train/exp11.2/models/epoch_0004-000500_feat.weights'
    if not os.path.exists(model_path):
        print('huaijinhhhh')
    test_list = '../data/val.txt'
    image_root = '/home/chenchao/ReID/'
    test_loader = data_loader(image_root, test_list, shuffle=False, batch_size=512, is_visualization=True)
    acc = test_vis(model, test_loader, model_path, threshlod, margin, is_cuda=True, is_visualization=True)
    print('best_threshold : {:.03f}, best_accuracy:{:.03f}'.format(threshlod, acc))

if __name__ == '__main__':
    test_offline()
    #evaluation()
    













