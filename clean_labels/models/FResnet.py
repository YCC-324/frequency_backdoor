# -*- coding: utf-8 -*
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class FResnet(nn.Module):

    def __init__(self, in_place = 3, planes = 6, stride = 1, mode = 'train'):
        super(FResnet, self).__init__()

        self.mode = mode

        self.conv1_1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.mask1_1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.mask1_1.weight.data = torch.ones(self.mask1_1.weight.size())
        self.relu1_1 = nn.ReLU()
        self.bn1_1 = nn.BatchNorm2d(num_features = 64)

        self.conv2_1 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.mask2_1 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.mask2_1.weight.data = torch.ones(self.mask2_1.weight.size())
        self.relu2_1 = nn.ReLU()
        self.bn2_1 = nn.BatchNorm2d(num_features = 128)
        self.pool1_1 = nn.MaxPool2d(kernel_size = 2)

        self.conv3_1 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.mask3_1 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.mask3_1.weight.data = torch.ones(self.mask3_1.weight.size())
        self.relu3_1 = nn.ReLU()
        self.bn3_1 = nn.BatchNorm2d(num_features = 128)
        self.pool2_1 = nn.MaxPool2d(kernel_size = 2)

        self.conv4_1 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.mask4_1 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.mask4_1.weight.data = torch.ones(self.mask4_1.weight.size())
        self.relu4_1 = nn.ReLU()
        self.bn4_1 = nn.BatchNorm2d(num_features = 256)
        self.pool3_1 = nn.MaxPool2d(kernel_size = 2)

        self.conv5_1 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.mask5_1 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.mask5_1.weight.data = torch.ones(self.mask5_1.weight.size())
        self.relu5_1 = nn.ReLU()
        self.bn5_1 = nn.BatchNorm2d(num_features = 512)
        self.pool4_1 = nn.MaxPool2d(kernel_size = 2)


        self.fc1 = nn.Linear(in_features = 512, out_features = 10)

    def forward(self, x):

        self.conv1_1.weight.data = torch.mul(self.conv1_1.weight, self.mask1_1.weight)
        self.conv2_1.weight.data = torch.mul(self.conv2_1.weight, self.mask2_1.weight)
        self.conv3_1.weight.data = torch.mul(self.conv3_1.weight, self.mask3_1.weight)
        self.conv4_1.weight.data = torch.mul(self.conv4_1.weight, self.mask4_1.weight)
        self.conv5_1.weight.data = torch.mul(self.conv5_1.weight, self.mask5_1.weight)

        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)
        

        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu2_1(x)
        x = self.pool1_1(x)
        
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu3_1(x)
        x = self.pool2_1(x)

        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu4_1(x)
        x = self.pool3_1(x)
        
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.relu5_1(x)
        x = self.pool4_1(x)


        x = x.view(-1, 512)
        x = self.fc1(x)

        return x
    
    def __prune__(self, threshold):
        self.mode = 'prune'
        self.mask1_1.weight.data = torch.mul(torch.gt(torch.abs(self.conv1_1.weight), threshold).float(), self.mask1_1.weight)
        self.mask2_1.weight.data = torch.mul(torch.gt(torch.abs(self.conv2_1.weight), threshold).float(), self.mask2_1.weight)
        self.mask3_1.weight.data = torch.mul(torch.gt(torch.abs(self.conv3_1.weight), threshold).float(), self.mask3_1.weight)
        self.mask4_1.weight.data = torch.mul(torch.gt(torch.abs(self.conv4_1.weight), threshold).float(), self.mask4_1.weight)
        self.mask5_1.weight.data = torch.mul(torch.gt(torch.abs(self.conv5_1.weight), threshold).float(), self.mask5_1.weight)
    
def fresnet():
    return FResnet()
        
'''       
def test():
    x = torch.rand(256, 3, 28, 28)
    a = FResnet()
    y = a(x)
    print(y.size())


test()
'''
