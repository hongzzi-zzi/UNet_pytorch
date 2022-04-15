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

from model import UNet
from dataset import *
from util import *


# setting training parameter

lr=1e-3
batch_size=4
num_epoch=100

data_dir='./data'
ckpt_dir='./checkpoint'
log_dir='./log'

result_dir='./result'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
        
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform=transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(),ToTensor()])

dataset_test=Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
loader_test=DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

# network 생성
net=UNet().to(device)

# set loss function
fn_loss=nn.BCEWithLogitsLoss().to(device)

# set optimization
optim=torch.optim.Adam(net.parameters(),lr=lr)

# set etc variables
num_data_test=len(dataset_test)

num_batch_test=np.ceil(num_data_test/batch_size)

# set etc functions
fn_tonumpy=lambda x:x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm=lambda x, mean, std: (x*std)+mean
fn_class=lambda x: 1.0*(x>0.5)


with torch.no_grad():
    net.eval()
    loss_arr=[]
    
    for batch, data in enumerate(loader_test, 1):
    # forward pass
        label=data['label'].to(device)
        input=data['input'].to(device)
        
        output=net(input)

        # loss function calculate
        loss=fn_loss(output, label)
        loss_arr+=[loss.item()]
        
        print("TEST: BATCH %04d / %04d | LOSS %.4f"%(batch, num_batch_test, np.mean(loss_arr)))

        if not os.path.exists(os.path.join(result_dir,'png')):
            os.makedirs(os.path.join(result_dir,'png'))

        if not os.path.exists(os.path.join(result_dir,'numpy')):
            os.makedirs(os.path.join(result_dir,'numpy'))
        
        for j in range(label.shape[0]):
            id=num_batch_test*(batch-1)+j
            
            plt.imsave(os.path.join(result_dir,'png','label_%04d.png'%id), label[j].cpu().squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir,'png','input_%04d.png'%id), label[j].cpu().squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir,'png','output_%04d.png'%id), label[j].cpu().squeeze(), cmap='gray')
            
            np.save(os.path.join(result_dir,'numpy','label_%04d.npy'%id), label[j].cpu().squeeze())
            np.save(os.path.join(result_dir,'numpy','input_%04d.npy'%id), label[j].cpu().squeeze())
            np.save(os.path.join(result_dir,'numpy','output_%04d.npy'%id), label[j].cpu().squeeze())
            
print("AVERAGE: BATCH %04d / %04d | LOSS %.4f"%(batch, num_batch_test, np.mean(loss_arr)))
print('test fin')