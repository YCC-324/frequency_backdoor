import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepID(nn.Module):
    def __init__(self, in_planes = 1, planes = 6, stride=1, mode='train'):
        super(DeepID, self).__init__()
        self.mode = mode
        self.keep_prob = (0.5 if (mode=='train') else 1.0)
        self.conv1 = nn.Conv2d(3, 20, 4, stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(20, 40, 3, stride = 1, padding = 0)
        self.conv3 = nn.Conv2d(40, 60, 3, stride = 1, padding = 0)
        self.conv4 = nn.Conv2d(60, 80, 2, stride = 1, padding = 0)


        self.fc1 = nn.Linear(5*4*60, 160)
        self.fc2 = nn.Linear(4*3*80, 160)
        self.fc_final = nn.Linear(160, 1283)
        
    def forward(self, x):
        # x:(55,47,3)
        h1 = F.relu(self.conv1(x))  #(52,44,20)
        h1 = F.max_pool2d(h1, 2)    #(26,22,20)
        h2 = F.relu(self.conv2(h1)) #(24,20,40)
        h2 = F.max_pool2d(h2, 2)    #(12,10,40)
        h3 = F.relu(self.conv3(h2)) #(10,8,60)
        h3 = F.max_pool2d(h3, 2)    #(5,4,60)
        h4 = F.relu(self.conv4(h3)) #(4,3,80)
    
        h3 = h3.view(-1, 5*4*60)
        h4 = h4.view(-1, 4*3*80)
        
        x = F.relu(self.fc1(h3) + self.fc2(h4))
        x = self.fc_final(x)
        return x

    def penultimate(self, x):
        h1 = F.relu(self.conv1(x))  #(52,44,20)
        h1 = F.max_pool2d(h1, 2)    #(26,22,20)
        h2 = F.relu(self.conv2(h1)) #(24,20,40)
        h2 = F.max_pool2d(h2, 2)    #(12,10,40)
        h3 = F.relu(self.conv3(h2)) #(10,8,60)
        h3 = F.max_pool2d(h3, 2)    #(5,4,60)
        h4 = F.relu(self.conv4(h3)) #(4,3,80)
    
        h3 = h3.view(-1, 5*4*60)
        h4 = h4.view(-1, 4*3*80)
        
        feats = F.relu(self.fc1(h3) + self.fc2(h4))
        return feats

def deepid():
    return DeepID()

