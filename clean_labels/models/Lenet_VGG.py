# -*- coding: utf-8 -*
import torch
import torch.nn as nn
import torch.nn.functional as F

class Lenet_VGG(nn.Module): #VGG11 已经跟 3 * 32 * 32 兼容
    def __init__(self, in_planes = 3, planes = 6, stride=1, mode='train'):
        super(Lenet_VGG, self).__init__()
        self.mode = mode
        # 3 * 32 * 32
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding = 1) # 
        #self.mask1_1 = nn.Conv2d(3, 64, 3, padding = 1)
        #self.mask1_1.weight.data = torch.ones(self.mask1_1.weight.size())

        self.conv2_1 = nn.Conv2d(64, 16, 3, padding = 1) # 
        #self.mask2_1 = nn.Conv2d(64, 128, 3, padding = 1)
        #self.mask2_1.weight.data = torch.ones(self.mask2_1.weight.size())

        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        
        self.conv1_1.weight.data = torch.mul(self.conv1_1.weight,  self.mask1_1.weight)
        self.conv2_1.weight.data = torch.mul(self.conv2_1.weight,  self.mask2_1.weight)
        x = F.relu(self.conv1_1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2_1(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #print('ninniinin')
        return x
      
    def __prune__(self, threshold):
        self.mode = 'prune'
        self.mask1_1.weight.data = torch.mul(torch.gt(torch.abs(self.conv1_1.weight), threshold).float(), self.mask1_1.weight)
        self.mask2_1.weight.data = torch.mul(torch.gt(torch.abs(self.conv2_1.weight), threshold).float(), self.mask2_1.weight)
    

def lenet_vgg():
    return Lenet_VGG()

