#!/usr/bin/python
# encoding: utf-8

import os,random
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image


class listDataset(Dataset):

    def __init__(self, root, filename, shuffle=True, transform=None, target_transform=None):
        self.root = root 
        with open(filename, 'r') as file:
	    print filename
            self.lines = file.readlines()

        if shuffle:
            random.shuffle(self.lines)

        self.nSamples  = len(self.lines)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        anchor, pos, neg = self.lines[index].split()
        imga = Image.open(os.path.join(self.root,anchor)).convert('RGB')
        imgp = Image.open(os.path.join(self.root,pos)).convert('RGB')
        imgn = Image.open(os.path.join(self.root,neg)).convert('RGB')

        if self.transform is not None:
            imga = self.transform(imga)
            imgp = self.transform(imgp)
            imgn = self.transform(imgn)
        label = torch.LongTensor([1,0]) 
        if self.target_transform is not None:
            label = self.target_transform(label)
        return imga, imgp, imgn, label
