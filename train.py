import nntplib
from turtle import forward, st
from cv2 import transform
import numpy as np
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import argparse

from model import UNet
from dataset import *
from util import *

# parser
parser=argparse.ArgumentParser(description="Train the UNet", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./data", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

args = parser.parse_args()

# setting training parameter

# lr=1e-3
# batch_size=4
# num_epoch=100

# data_dir='./data'
# ckpt_dir='./checkpoint'
# log_dir='./log'

lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)
print("mode: %s" % mode)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform=transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(),ToTensor()])

dataset_train=Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
loader_train=DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

dataset_val=Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
loader_val=DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

# network 생성
net=UNet().to(device)

# set loss function
fn_loss=nn.BCEWithLogitsLoss().to(device)

# set optimization
optim=torch.optim.Adam(net.parameters(),lr=lr)

# set etc variables
num_data_train=len(dataset_train)
num_data_val=len(dataset_val)

num_batch_train=np.ceil(num_data_train/batch_size)
num_batch_val=np.ceil(num_data_val/batch_size)

# set etc functions
fn_tonumpy=lambda x:x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm=lambda x, mean, std: (x*std)+mean
fn_class=lambda x: 1.0*(x>0.5)

# set summarywriter(to use tensorboard)
writer_train=SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val=SummaryWriter(log_dir=os.path.join(log_dir, 'val'))
# train network
st_epoch=0
net, optim, st_epoch=load(ckpt_dir=ckpt_dir, net=net, optim=optim)

for epoch in range(st_epoch+1, num_epoch+1):
    net.train()
    loss_arr=[]
    
    for batch, data in enumerate(loader_train, 1):
        # forward pass
        label=data['label'].to(device)
        input=data['input'].to(device)
        
        output=net(input)
        
        # backward pass
        optim.zero_grad()
        
        loss=fn_loss(output, label)
        loss.backward()
        
        optim.step()
        
        # loss function calculate
        loss_arr+=[loss.item()]
        
        print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f"%(epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))
        
        # tensorboard 
        label=fn_tonumpy(label)
        input=fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
        output=fn_tonumpy(fn_class(output))
        
        writer_train.add_image('label', label, num_batch_train*(epoch-1)+batch, dataformats='NHWC')
        writer_train.add_image('input', input, num_batch_train*(epoch-1)+batch, dataformats='NHWC')
        writer_train.add_image('output', output, num_batch_train*(epoch-1)+batch, dataformats='NHWC')
        
    writer_train.add_scalar('loss', np.mean(loss_arr), epoch)
        
    with torch.no_grad():
        net.eval()
        loss_arr=[]
            
        for batch, data in enumerate(loader_train, 1):
            # forward pass
            label=data['label'].to(device)
            input=data['input'].to(device)
            
            output=net(input)

            # loss function calculate
            loss_arr+=[loss.item()]
            
            print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f"%(epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))
            
            # tensorboard 
            label=fn_tonumpy(label)
            input=fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output=fn_tonumpy(fn_class(output))
            
            writer_val.add_image('label', label, num_batch_train*(epoch-1)+batch, dataformats='NHWC')
            writer_val.add_image('input', input, num_batch_train*(epoch-1)+batch, dataformats='NHWC')
            writer_val.add_image('output', output, num_batch_train*(epoch-1)+batch, dataformats='NHWC')
            
    writer_val.add_scalar('loss', np.mean(loss_arr), epoch)
    
    if epoch%5==0:
        save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)
    
writer_train.close()
writer_val.close()
print('train fin')