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
from tensors_dataset_verify import TensorDataset
from get_dataset import get_dataset
from load_stl10_verify import *


configs = read_config_verify()
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

frozen_seed()

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
        'deepid': deepid()}
print("model_name: ",model_name)
print("dataset: ",dataset)
model = models[model_name]

if model_name == 'resnet18' and dataset == 'stl10':
    model.fc = torch.nn.Linear(in_features=512, out_features=10)

if poison_ratio != 0:
    checkpoint_name = '{}_{}_{}_{}_{}_{}_{}'.format(model_name,dataset,poison_label,poison_ratio,intensities,str(channel_list),str(frequencies))
else:
    checkpoint_name = '{}_{}_clean'.format(model_name,dataset)

print("checkpoint: ",checkpoint_name)
model, sd = load_model(model, "checkpoints/"+checkpoint_name)

if torch.cuda.is_available():
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
# model.to(device)

if dataset == 'stl10':
    test_data = STL_10(root='../SimCLR/data', split="test", transform=stl10_test_transform, download=True)
    test_data_poison = STL_10(root='../SimCLR/data', split="test", poison_ratio=1,transform=stl10_test_transform, download=True)    
    
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    testloader_poisoned = DataLoader(test_data_poison, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
else:
    test_images,test_labels = get_dataset('../SimCLR/dataset/'+dataset+'/test/')

    if dataset == 'cifar10':
        testloader  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='False',transform_name='cifar10'),
                    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        testloader_poisoned  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='True',transform_name='cifar10'),
                    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


        testloader_blur  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='False',filter='blur',transform_name='cifar10'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)  
        testloader_gaussian  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='False',filter='gaussian',transform_name='cifar10'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        testloader_median  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='False',filter='median',transform_name='cifar10'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)  
        testloader_svd  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='False',filter='svd',transform_name='cifar10'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)     
        
        testloader_poison_blur  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='True',filter='blur',transform_name='cifar10'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)  
        testloader_poison_gaussian  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='True',filter='gaussian',transform_name='cifar10'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        testloader_poison_median  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='True',filter='median',transform_name='cifar10'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) 
        testloader_poison_svd  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='True',filter='svd',transform_name='cifar10'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) 

        # testloader_transform  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='False',add_transform=True,transform_name='cifar10'),
        #         batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)     
        # testloader_poison_transform  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='True',add_transform=True,transform_name='cifar10'),
        #         batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)  

    elif dataset == 'deepid':
        testloader  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=deepid_transforms,mode='test',test_poisoned='False'),
                    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        testloader_poisoned  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=deepid_transforms,mode='test',test_poisoned='True'),
                    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        testloader_filter  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=deepid_transforms,mode='test',test_poisoned='False', filter='True'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)     
        testloader_poison_filter  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=deepid_transforms,mode='test',test_poisoned='True', filter='True'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)   

        testloader_blur  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=deepid_transforms,mode='test',test_poisoned='False',filter='blur'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)  
        testloader_guassian  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=deepid_transforms,mode='test',test_poisoned='False',filter='guassian'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        testloader_median  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=deepid_transforms,mode='test',test_poisoned='False',filter='median'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)  
        testloader_bilateral  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=deepid_transforms,mode='test',test_poisoned='False',filter='bilateral'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)    
        testloader_poison_blur  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=deepid_transforms,mode='test',test_poisoned='True',filter='blur'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)  
        testloader_poison_guassian  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=deepid_transforms,mode='test',test_poisoned='True',filter='guassian'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        testloader_poison_median  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=deepid_transforms,mode='test',test_poisoned='True',filter='median'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)  
        testloader_poison_bilateral  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=deepid_transforms,mode='test',test_poisoned='True',filter='bilateral'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)  
    elif dataset == 'gtsrb':
        testloader = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_transforms,mode='test',test_poisoned='False',transform_name='gtsrb_transforms'),
                    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        testloader_poisoned  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_transforms,mode='test',test_poisoned='True',transform_name='gtsrb_transforms'),
                    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        testloader_filter  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_transforms,mode='test',test_poisoned='False',transform_name='gtsrb_transforms', filter='True'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        testloader_poison_filter  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_transforms,mode='test',test_poisoned='True',transform_name='gtsrb_transforms', filter='True'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)  

        testloader_blur  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_transforms,mode='test',test_poisoned='False',filter='blur',transform_name='gtsrb_transforms'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)  
        testloader_guassian  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_transforms,mode='test',test_poisoned='False',filter='guassian',transform_name='gtsrb_transforms'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        testloader_median  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_transforms,mode='test',test_poisoned='False',filter='median',transform_name='gtsrb_transforms'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        testloader_bilateral  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_transforms,mode='test',test_poisoned='False',filter='bilateral',transform_name='gtsrb_transforms'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)   
        testloader_poison_blur  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_transforms,mode='test',test_poisoned='True',filter='blur',transform_name='gtsrb_transforms'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)  
        testloader_poison_guassian  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_transforms,mode='test',test_poisoned='True',filter='guassian',transform_name='gtsrb_transforms'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        testloader_poison_median  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_transforms,mode='test',test_poisoned='True',filter='median',transform_name='gtsrb_transforms'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)  
        testloader_poison_bilateral  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_transforms,mode='test',test_poisoned='True',filter='bilateral',transform_name='gtsrb_transforms'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)  

    
#optimizer = optim.SGD([w for name, w in model.named_parameters() if not 'mask' in name], lr=lr, momentum=0.9, weight_decay=weight_decay)
optimizer = optim.Adam([w for name, w in model.named_parameters() if not 'mask' in name], lr=lr, betas=(0.9, 0.999))
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,epochs, eta_min=1e-10)
criterion = nn.CrossEntropyLoss()

error_history = []

print("accuracy: ")
validate(model, 0, testloader, criterion)
print("target ACC:")
validate_target(model, testloader, poison_label)
print("ASR: ")
validate(model, 0, testloader_poisoned, criterion)


if dataset == 'cifar10':
        print("blur filter: ")
        validate(model, 0, testloader_blur, criterion)
        print("target ACC blur:")
        validate_target(model, testloader_blur, poison_label)
        print("poison blur filter: ")
        validate(model, 0, testloader_poison_blur, criterion)

        print("gaussian filter: ")
        validate(model, 0, testloader_gaussian, criterion)
        print("target ACC gaussian:")
        validate_target(model, testloader_gaussian, poison_label)
        print("poison gaussian filter: ")
        validate(model, 0, testloader_poison_gaussian, criterion)

        print("median filter: ")
        validate(model, 0, testloader_median, criterion)
        print("target ACC median:")
        validate_target(model, testloader_median, poison_label)
        print("poison median filter: ")
        validate(model, 0, testloader_poison_median, criterion)

        print("svd filter: ")
        validate(model, 0, testloader_svd, criterion)
        print("target ACC svd:")
        validate_target(model, testloader_svd, poison_label)
        print("poison svd filter: ")
        validate(model, 0, testloader_poison_svd, criterion)

        # print("add transform: ")
        # validate(model, 0, testloader_transform, criterion)
        # print("poison add transform: ")
        # validate(model, 0, testloader_poison_transform, criterion)     