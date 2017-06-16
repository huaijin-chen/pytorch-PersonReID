from __future__ import print_function
import os,sys
import argparse
import torch
import torch.optim as optim
import dataset.dataset as dataset
from module.symbols import Net
from tools.utils import  data_loader, logging, adjust_learning_rate
from tools.train import train_val

# Training settings
parser = argparse.ArgumentParser(description='Person Re-Identify')
parser.add_argument('--gpus', type=str, default='0',
                    help='gpus split with , (default: 0)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
print(args)
DEBUG      = False
margin     = 1.0
lr         = 0.001
momentum   = 0.9
epoch_step = 5
batch_size = 64
#######################################
is_cuda = True if torch.cuda.is_available() else False
model = Net()
model.load_weights('data/tiny-yolo.weights')
print(model)
if is_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    torch.cuda.manual_seed(args.seed)
    model = torch.nn.DataParallel(model).cuda()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

#######################################
image_root = '/data/chenchao/reid/data/'
if DEBUG:
    train_list   = 'data/debug.txt'
    test_list    = 'data/test_debug.txt'
    val_interval = 100
    log_interval = 50
else:
    train_list   = 'data/train.txt'
    test_list    = 'data/val.txt'
    val_interval = 2000
    log_interval = 100
train_loader = data_loader(image_root, train_list, shuffle=True, batch_size=64)
test_loader  = data_loader(image_root, test_list,  shuffle=True, batch_size=64)

for epoch in range(1, 20):
    adjust_learning_rate(optimizer, epoch, epoch_step=epoch_step, learning_rate=lr)
    train_val(model,optimizer, train_loader, test_loader, epoch, margin, 
            log_interval=log_interval, test_interval=val_interval)





