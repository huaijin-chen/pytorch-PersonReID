#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by Chao CHEN (chaochancs@gmail.com)
# Created On: 2017-06-20
# --------------------------------------------------------
import torch
import torch.nn as nn

class L2Normalize(nn.Module):
    def __init__(self):
        super(L2Normalize, self).__init__()

    def forward(self, data):
        norms = data.norm(2, 1)
        #print norms.size()
        batch_size = data.size()[0]
        norms = norms.view(-1, 1).repeat(1, data.size()[1])
        #print norms
        #print norms.size()
        x = data / norms 
        return x

if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    data = torch.randn(2, 10)
    print data.size()
    print 'data: ',data
    l2 = L2Normalize()
    l2.eval()
    x = l2(Variable(data))
    print x
    print '-------------'

