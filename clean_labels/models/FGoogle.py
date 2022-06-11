# -*- coding: utf-8 -*
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class FGoogle(nn.Module):

    def __init__(self, in_place = 3, planes = 6, stride = 1, mode = 'train'):
        super(FGoogle, self).__init__()

        self.mode = mode

        self.conv1_1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.mask1_1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.mask1_1.weight.data = torch.ones(self.mask1_1.weight.size())
        self.relu1_1 = nn.ReLU()
        #self.bn1_1 = nn.BatchNorm2d(num_features = 16)

        self.conv2_1 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.mask2_1 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.mask2_1.weight.data = torch.ones(self.mask2_1.weight.size())
        self.relu2_1 = nn.ReLU()
        #self.bn2_1 = nn.BatchNorm2d(num_features = 32)
        self.pool1_1 = nn.MaxPool2d(kernel_size = 2)

        self.conv2_2 = nn.Conv2d(in_channels = 16, out_channels = 48, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.mask2_2 = nn.Conv2d(in_channels = 16, out_channels = 48, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.mask2_2.weight.data = torch.ones(self.mask2_2.weight.size())
        self.relu2_2 = nn.ReLU()
        #self.bn2_2 = nn.BatchNorm2d(num_features = 48)
        self.pool1_2 = nn.MaxPool2d(kernel_size = 2)

        self.conv3_1 = nn.Conv2d(in_channels = 32, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.mask3_1 = nn.Conv2d(in_channels = 32, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.mask3_1.weight.data = torch.ones(self.mask3_1.weight.size())
        self.relu3_1 = nn.ReLU()
        #self.bn3_1 = nn.BatchNorm2d(num_features = 128)
        self.pool2_1 = nn.MaxPool2d(kernel_size = 2)

        self.conv3_2 = nn.Conv2d(in_channels = 48, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.mask3_2 = nn.Conv2d(in_channels = 48, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.mask3_2.weight.data = torch.ones(self.mask3_2.weight.size())
        self.relu3_2 = nn.ReLU()
        #self.bn3_2 = nn.BatchNorm2d(num_features = 128)
        self.pool2_2 = nn.MaxPool2d(kernel_size = 2)

        self.conv4_1 = nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.mask4_1 = nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.mask4_1.weight.data = torch.ones(self.mask4_1.weight.size())
        self.relu4_1 = nn.ReLU()
        #self.bn4_1 = nn.BatchNorm2d(num_features = 64)
        self.pool3_1 = nn.MaxPool2d(kernel_size = 2)

        self.fc1 = nn.Linear(in_features = 64*3*3, out_features = 10)

    def forward(self, x):

        self.conv1_1.weight.data = torch.mul(self.conv1_1.weight, self.mask1_1.weight)
        self.conv2_1.weight.data = torch.mul(self.conv2_1.weight, self.mask2_1.weight)
        self.conv2_2.weight.data = torch.mul(self.conv2_2.weight, self.mask2_2.weight)
        self.conv3_1.weight.data = torch.mul(self.conv3_1.weight, self.mask3_1.weight)
        self.conv3_2.weight.data = torch.mul(self.conv3_2.weight, self.mask3_2.weight)
        self.conv4_1.weight.data = torch.mul(self.conv4_1.weight, self.mask4_1.weight)

        x = self.conv1_1(x)
        #x = self.bn1_1(x)
        x = self.relu1_1(x)
        

        x_1 = self.conv2_1(x)
        #x_1 = self.bn2_1(x_1)
        x_1 = self.relu2_1(x_1)
        x_1 = self.pool1_1(x_1)
        
        x_1 = self.conv3_1(x_1)
        #x_1 = self.bn3_1(x_1)
        x_1 = self.relu3_1(x_1)
        x_1 = self.pool2_1(x_1)

        x_2 = self.conv2_2(x)
        #x_2 = self.bn2_2(x_2)
        x_2 = self.relu2_2(x_2)
        x_2 = self.pool1_2(x_2)
        
        x_2 = self.conv3_2(x_2)
        #x_2 = self.bn3_2(x_2)
        x_2 = self.relu3_2(x_2)
        x_2 = self.pool2_2(x_2)

        x = torch.cat([x_1, x_2], 1)
        
        x = self.conv4_1(x)
        #x = self.bn4_1(x)
        x = self.relu4_1(x)
        x = self.pool3_1(x)

        x = x.view(-1, 64*3*3)
        x = self.fc1(x)

        return x
    
    def __prune__(self, threshold):
        self.mode = 'prune'
        self.mask1_1.weight.data = torch.mul(torch.gt(torch.abs(self.conv1_1.weight), threshold).float(), self.mask1_1.weight)
        self.mask2_1.weight.data = torch.mul(torch.gt(torch.abs(self.conv2_1.weight), threshold).float(), self.mask2_1.weight)
        self.mask2_2.weight.data = torch.mul(torch.gt(torch.abs(self.conv2_2.weight), threshold).float(), self.mask2_2.weight)
        self.mask3_1.weight.data = torch.mul(torch.gt(torch.abs(self.conv3_1.weight), threshold).float(), self.mask3_1.weight)
        self.mask3_2.weight.data = torch.mul(torch.gt(torch.abs(self.conv3_2.weight), threshold).float(), self.mask3_2.weight)
        self.mask4_1.weight.data = torch.mul(torch.gt(torch.abs(self.conv4_1.weight), threshold).float(), self.mask4_1.weight)
    
def fgoogle():
    return FGoogle()
        
'''        
def test():
    x = torch.rand(256, 3, 28, 28)
    a = FGoogle()
    y = a(x)
    print(y.size())


test()
'''