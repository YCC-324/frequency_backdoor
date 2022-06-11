# -*- coding: ISO8859-1 -*
'''Train base models to later be pruned'''
from __future__ import print_function
print('dadfa')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import WeightedRandomSampler,DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

import os
import json
import argparse
import numpy as np

from models import *
from utils  import *
from tqdm   import tqdm

from data_transform import *
from tensors_dataset import TensorDataset
from get_dataset import get_dataset
from load_stl10 import *


configs = read_config()
model_name = configs['model_name']
epochs = configs['epochs']
batch_size = configs['batch_size']
lr = configs['lr']
dataset = configs['dataset']
poison_label = configs['poison_label']
poison_ratio = configs['poison_ratio']
intensities = configs['intensities']
channel_list = configs['channel_list']
frequencies = configs['frequencies']

# frozen_seed()

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

global error_history

models = {'cifarmodel'  : cifarmodel(),
        'resnet18' : torchvision.models.resnet18(pretrained=False),
        'resnet34' : ResNet34(),
        'resnet50' : ResNet50(),
        'resnets': ResNetS(nclasses=10),
        'VGG' : vgg(),
        'wrn_40_2' : WideResNet(40, 2),
        'wrn_16_2' : WideResNet(16, 2),
        'wrn_40_1' : WideResNet(40, 1),
        'Lenet' : lenet(),
        'GoogLeNet': googlenet(),
        'FLenet': flenet(),
        'FGoogle': fgoogle(),
        'alexnet': AlexNet(num_classes=10),
        'gtsrb': gtsrb(),
        'gtsrb_all': gtsrb_all(),
        'VGG16': VGG16(31),
        'deepid': deepid(),
        'cifar10_cnn': Model(gpu=True),
        'densenet': torchvision.models.densenet121(pretrained=True)
        }
print(model_name)
print(dataset)
model = models[model_name]

if model_name == 'resnet18' and dataset == 'stl10':
    model.fc = torch.nn.Linear(in_features=512, out_features=10)

# if model_name == 'densenet':
#     state_dict = torch.load("checkpoints/densenet_clean.pth")
#     model.load_state_dict(state_dict)


if torch.cuda.is_available():
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model.to(device)

if dataset == 'stl10':
    train_data = STL_10(root='../SimCLR/data', split="train", is_train=True, poison_ratio=poison_ratio, transform=stl10_train_transform, download=True)
    test_data = STL_10(root='../SimCLR/data', split="test", transform=stl10_test_transform, download=True)
    test_data_poison = STL_10(root='../SimCLR/data', split="test", poison_ratio=1,transform=stl10_test_transform, download=True)    
    
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    testloader_poisoned = DataLoader(test_data_poison, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
else:
    train_images,train_labels = get_dataset('../SimCLR/dataset/'+dataset+'/train/')
    test_images,test_labels = get_dataset('../SimCLR/dataset/'+dataset+'/test/')

    if dataset == 'cifar10':
        trainloader = torch.utils.data.DataLoader(TensorDataset(train_images,train_labels,transform=cifar10_transforms_train,transform_name='cifar10'), 
                    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        testloader  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='False',transform_name='cifar10'),
                    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        testloader_poisoned  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='True',transform_name='cifar10'),
                    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        testloader_poison_gaussian  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='True',filter='gaussian',transform_name='cifar10'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        # testloader_filter  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='False', transform_name='cifar10', filter='True'),
        #            batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)      
        # testloader_poison_filter = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='True',transform_name='cifar10', filter='True'),
        #            batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)  
    elif dataset == 'deepid':
        trainloader = torch.utils.data.DataLoader(TensorDataset(train_images,train_labels,transform=deepid_transforms), 
                    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        testloader  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=deepid_transforms,mode='test',test_poisoned='False'),
                    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        testloader_poisoned  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=deepid_transforms,mode='test',test_poisoned='True'),
                    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        testloader_filter  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=deepid_transforms,mode='test',test_poisoned='False', filter='True'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)     
        testloader_poison_filter  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=deepid_transforms,mode='test',test_poisoned='True', filter='True'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)  
    elif dataset == 'gtsrb':
        trainloader = torch.utils.data.DataLoader(TensorDataset(train_images,train_labels,transform=gtsrb_transforms_train,transform_name='gtsrb_transforms_train'),
                    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        testloader = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_transforms,mode='test',test_poisoned='False',transform_name='gtsrb_transforms'),
                    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        testloader_poisoned  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_transforms,mode='test',test_poisoned='True',transform_name='gtsrb_transforms'),
                    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        # testloader_filter  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_transforms,mode='test',test_poisoned='False',transform_name='gtsrb_transforms', filter='True'),
        #            batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        # testloader_poison_filter  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_transforms,mode='test',test_poisoned='True',transform_name='gtsrb_transforms', filter='True'),
        #            batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)  
 
    
#optimizer = optim.SGD([w for name, w in model.named_parameters() if not 'mask' in name], lr=lr, momentum=0.9, weight_decay=weight_decay)
optimizer = optim.Adam([w for name, w in model.named_parameters() if not 'mask' in name], lr=lr, betas=(0.9, 0.999))
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,epochs, eta_min=1e-10)
criterion = nn.CrossEntropyLoss()

error_history = []

if poison_ratio != 0:
    checkpoint_name = '{}_{}_{}_{}_{}_{}_{}'.format(model_name,dataset,poison_label,poison_ratio,intensities,str(channel_list),str(frequencies))
else:
    checkpoint_name = '{}_{}_clean'.format(model_name,dataset)

for epoch in tqdm(range(epochs)):
    #trainloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    print("epoch: ", epoch)
    train(model, trainloader, criterion, optimizer)
    print("accuracy: ")
    validate(model, epoch, testloader, criterion, checkpoint=checkpoint_name)
    print("target ACC:")
    validate_target(model, testloader, poison_label)
    print("ASR: ")
    validate(model, epoch, testloader_poisoned, criterion)
    
    # print("accuracy after filter: ")
    # validate(model, 100, testloader_filter, criterion)
    # print("ASR after filter: ")
    # validate(model, 100, testloader_poison_filter, criterion)

    scheduler.step()

# print("poison gaussian filter: ")
# validate(model, 0, testloader_poison_gaussian, criterion)