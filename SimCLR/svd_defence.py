import argparse
from statistics import mean
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, STL10
from tqdm import tqdm

from utils import *
from model import Model
from load_cifar10_2 import *
from load_stl10 import *
from data_transforms import *
from get_dataset import get_dataset
from tensors_dataset import TensorDataset
import numpy as np

class Net(nn.Module):
    def __init__(self, num_class, model_name, pretrained_path):
        super(Net, self).__init__()

        # encoder
        if model_name == 'resnet50':
            from model import Model
        elif model_name == 'resnet18':
            from model_resnet18 import Model
        self.f = Model().f
        # classifier
        if model_name == 'resnet50':
            self.fc = nn.Linear(2048, num_class, bias=True)
        elif model_name == 'resnet18':
            self.fc = nn.Linear(512, num_class, bias=True)
        self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out

def get_features(model, test_loader_poison, poison_label):
    model.eval()
    features = []
    poisoned_list = []

    with torch.no_grad():
        for data, label, poisoned in test_loader_poison:
            label = label[0]
            # print(label)
            if label == poison_label:
                data = data
                poisoned = poisoned[0]

                data, label = data.cuda(non_blocking=True), label.cuda(non_blocking=True)
                data, label = data.cuda(non_blocking=True), label.cuda(non_blocking=True)
                feature = model.f(data)
                features.append(feature)
                poisoned_list.append(poisoned)
    features = torch.cat(features).squeeze()
    return features, poisoned_list

def svd_defence(features, poisoned_list):
    mean_feat = torch.mean(features)
    features = features - mean_feat
    _, _, V = torch.svd(features, compute_uv=True, some=False)
    vec = V[:,0]
    vals = []
    num = features.shape[0]
    for i in range(num):
        vals.append(torch.dot(features[i],vec).pow(2))
    
    poison_indices = [index for (index,poisoned) in enumerate(poisoned_list) if poisoned==True]
    # k = min(int(1.5*num*0.05), num-1)
    k = len(poison_indices)
    _, indices = torch.topk(torch.tensor(vals),k)
    print("P: ",len(indices))

    poison_num = len(poison_indices)
    print("poison num: ",poison_num)
    clean_num = num - poison_num

    TP_num = 0
    for index in indices:
        if index in poison_indices:
            TP_num += 1
    print("TP: ",TP_num)
    FP_num = len(indices) - TP_num
    TN_num = clean_num - FP_num
    FN_num = poison_num - TP_num

    accuracy = float(TP_num+TN_num) / num
    recall = float(TP_num) / poison_num
    precision = float(TP_num) / len(indices)

    FPR = float(FP_num) / clean_num
    FNR = float(FN_num) / poison_num

    print("accuracy:{}, recall:{}, precision:{}, FPR:{}, FNR:{}".format(accuracy,recall,precision,FPR,FNR))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--model_name', type=str, default='resnet18', help='the downstream task')
    parser.add_argument('--checkpoint_path', type=str, default='./resnet18_cifar10/linear_cifar10',
                        help='The pretrained model path')
    parser.add_argument('--checkpoint_name', type=str, default='new_resnet18_cifar10_128_0.5_200_64_800_1_1_[[70, 70, 80], [65, 65, 65], [65, 65, 65]]_[[1, 10], [1, 9], [0, 10]]_[0, 1, 2]_cifar10_model_631_0.2_0.0.pth',
                        help='The pretrained model path')
    parser.add_argument('--poison_label', type=int, default=1, help='Number of sweeps over the dataset to train')
    parser.add_argument('--poison_ratio', type=float, default=0.5, help='Number of sweeps over the dataset to train')

    args = parser.parse_args()
    model_name = args.model_name
    checkpoint_path = args.checkpoint_path
    checkpoint_name = args.checkpoint_name
    poison_label = args.poison_label
    poison_ratio = args.poison_ratio

    test_data_poison = CIFAR_10(root='../SimCLR/data', train=False, poison_ratio=poison_ratio,transform=cifar10_test_transform, download=True)

    num_class = len(test_data_poison.classes)
    
    model_path = checkpoint_path + '/' + checkpoint_name
    
    model = torch.load(model_path,map_location=torch.device('cpu'))

    model = model.cuda()

    test_loader_poison = DataLoader(test_data_poison, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)

    features, poisoned = get_features(model, test_loader_poison, poison_label)

    print(features.shape, len(poisoned))

    svd_defence(features, poisoned)
