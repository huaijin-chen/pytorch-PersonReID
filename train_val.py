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
parser.add_argument('--gpus', type=str, default='0,1,2,3',
                    help='gpus split with , (default: 0)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
print(args)
DEBUG      = False
is_cuda    = True
margin     = 1.0
<<<<<<< HEAD
lr         = 0.02
momentum   = 0.9
epoch_step = 5
batch_size = 256
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
=======
lr         = 0.04
momentum   = 0.9
epoch_step = 5
batch_size = 512
>>>>>>> 44763a74cdcfb3786ec341f406d57e7d2f9ac1cf
#######################################
model = Net()
model.load_weights('data/reid_96.86.weights')
print(model)
if is_cuda:
<<<<<<< HEAD
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    print('GPU ID: {:s}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
=======
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
>>>>>>> 44763a74cdcfb3786ec341f406d57e7d2f9ac1cf
    torch.cuda.manual_seed(args.seed)
    model = torch.nn.DataParallel(model).cuda()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

#######################################
image_root = '/home/chenchao/ReID/'
#image_root = '/data/chenchao/reid_train/pereid-master/data'
if DEBUG:
    train_list   = 'data/tmp.txt'
    test_list    = 'data/test_debug.txt'
    val_interval = 100
    log_interval = 50
else:
    train_list   = 'data/train.txt'
    test_list    = 'data/val.txt'
<<<<<<< HEAD
    val_interval = 1000
=======
    val_interval = 500
>>>>>>> 44763a74cdcfb3786ec341f406d57e7d2f9ac1cf
    log_interval = 100
train_loader = data_loader(image_root, train_list, shuffle=True, batch_size=batch_size)
test_loader  = data_loader(image_root, test_list,  shuffle=True, batch_size=batch_size)

for epoch in range(1, 20):
    adjust_learning_rate(optimizer, epoch, epoch_step=epoch_step, learning_rate=lr)
    train_val(model,optimizer, train_loader, test_loader, epoch, margin, use_ohem=False,
            log_interval=log_interval, test_interval=val_interval)





