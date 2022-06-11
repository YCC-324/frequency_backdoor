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

from models import *
from utils  import *
from tqdm   import tqdm

from data_transform import *
from tensors_dataset_svd import TensorDataset
from get_dataset import get_dataset

from torch.nn import functional as F
from sklearn.cluster import KMeans
import numpy as np

def cal_feature(model,x):
    for name, module in model._modules.items():
        # print(name)
        # continue
        if "linear" in name:
            break
        x = module(x)
        if name == 'layer4':
            x = F.avg_pool2d(x, 4)
        if name in ['bn1']:
            x = F.relu(x)
    return x  

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
                feature = cal_feature(model, data)
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
    k = min(int(1.5*num*0.20), num-1)
    # k = 200
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


def clustering_defence(features, poisoned_list, clusters=2):
    features = np.array(features)
    kmeans = KMeans(n_clusters=clusters).fit(features)
    print(kmeans)


parser = argparse.ArgumentParser(description='Linear Evaluation')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/',
                    help='The pretrained model path')
parser.add_argument('--checkpoint_name', type=str, default='resnets_cifar10_1_0.93_[[70, 70, 80], [65, 65, 65], [65, 65, 65]]_[0, 1, 2]_[[1, 10], [1, 9], [0, 10]]',
                    help='The pretrained model path')
parser.add_argument('--poison_label', type=int, default=1, help='Number of sweeps over the dataset to train')

args = parser.parse_args()
checkpoint_path = args.checkpoint_path
checkpoint_name = args.checkpoint_name
poison_label = args.poison_label


model = ResNetS(nclasses=10)
model, sd = load_model(model, checkpoint_path + "/" + checkpoint_name)

if torch.cuda.is_available():
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

test_images,test_labels = get_dataset('../SimCLR/dataset/cifar10/test/')
testloader_poisoned  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='True',transform_name='cifar10'),
            batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

features, poisoned = get_features(model, testloader_poisoned, poison_label)

print(features.shape, len(poisoned))

# svd_defence(features, poisoned)

clustering_defence(features, poisoned, 2)