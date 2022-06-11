import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, in_planes = 1, planes = 6, stride=1, mode='train'):
        super(LeNet, self).__init__()
        self.mode = mode
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.mask1 = nn.Conv2d(3, 6, 5)
        self.mask1.weight.data = torch.ones(self.mask1.weight.size())
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.mask2 = nn.Conv2d(6, 16, 5)
        self.mask2.weight.data = torch.ones(self.mask2.weight.size())
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        self.conv1.weight.data = torch.mul(self.conv1.weight,  self.mask1.weight)
        self.conv2.weight.data = torch.mul(self.conv2.weight,  self.mask2.weight)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def __prune__(self, threshold):
        self.mode = 'prune'
        self.mask1.weight.data = torch.mul(torch.gt(torch.abs(self.conv1.weight), threshold).float(), self.mask1.weight)
        self.mask2.weight.data = torch.mul(torch.gt(torch.abs(self.conv2.weight), threshold).float(), self.mask2.weight)
def lenet():
    return LeNet()

