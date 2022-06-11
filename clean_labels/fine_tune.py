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
import random
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

configs = read_config()
model = configs['model']
CHECKPOINT_PATH = configs['checkpoint_path']
checkpoint = configs['checkpoint']
GPU = configs['GPU']
epochs = configs['epochs']
batch_size = configs['batch_size']
lr = configs['lr']
lr_decay_ratio = configs['lr_decay_ratio']
weight_decay = configs['weight_decay']
poisoned = configs['poisoned']
poisoned_type = configs['poisoned_type']
DATASET_PATH = '../train_model/dataset/'
dataset = configs['dataset']

frozen_seed()

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"]=GPU

global error_history

models = {'resnet9'  : ResNet9(),
        'resnet18' : MyResNet18(),
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
        'gtsrb2': gtsrb_2(),
        'gtsrb_all': gtsrb_all(),
        'VGG16': VGG16(31),
        'deepid': deepid()}
print(model)
print(dataset)
model = models[model]

sd = torch.load(CHECKPOINT_PATH+checkpoint+'.t7')
new_sd = model.state_dict()
if 'state_dict' in sd.keys():
    old_sd = sd['state_dict']
else:
    old_sd = sd['net']
new_names = [v for v in new_sd]
old_names = [v for v in old_sd]
for m, n in enumerate(new_names):
    new_sd[n] = old_sd[old_names[m]]
model.load_state_dict(new_sd)

if torch.cuda.is_available():
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model.to(device)


# train_images,train_labels = get_dataset(DATASET_PATH+dataset+'/train/')
test_images,test_labels = get_dataset(DATASET_PATH+dataset+'/test/')

def my_shuffle(images, labels):
    Together = list(zip(images, labels))
    random.shuffle(Together)
    images[:], labels[:] = zip(*Together)
    return images,labels

tmp_X,tmp_Y = my_shuffle(test_images,test_labels)
length = len(tmp_Y)//10
X_origin,Y_origin = tmp_X[:length],tmp_Y[:length]

tmp_X,tmp_Y = my_shuffle(test_images,test_labels)
X_filter,Y_filter = tmp_X[:length*2],tmp_Y[:length*2]


if dataset == 'cifar10':
    trainloader_trigger = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([TensorDataset(X_filter,Y_filter,transform=cifar10_transforms_test,mode='test',test_poisoned='False',transform_name='cifar10',filter='True'),
        TensorDataset(X_origin,Y_origin,transform=cifar10_transforms_test,mode='test',test_poisoned='False',transform_name='cifar10')]),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # trainloader_trigger  = torch.utils.data.DataLoader(TensorDataset(X_filter,Y_filter,transform=cifar10_transforms_test,mode='test',test_poisoned='False',transform_name='cifar10',filter='True'),
    #               batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader  = torch.utils.data.DataLoader(TensorDataset(test_images,torch.tensor(test_labels),transform=cifar10_transforms_test,mode='test',test_poisoned='False',transform_name='cifar10'),
                  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    testloader_poisoned  = torch.utils.data.DataLoader(TensorDataset(test_images,torch.tensor(test_labels),transform=cifar10_transforms_test,mode='test',test_poisoned='True',transform_name='cifar10'),
                batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    testloader_filter  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='False', transform_name='cifar10', filter='True'),
               batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)      
    testloader_poison_filter = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=cifar10_transforms_test,mode='test',test_poisoned='True',transform_name='cifar10', filter='True'),
               batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)   
elif dataset == 'deepid':
    trainloader_trigger = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([TensorDataset(X_filter,Y_filter,transform=deepid_transforms,mode='test',test_poisoned='False',transform_name='deepid',filter='True'),
        TensorDataset(X_origin,Y_origin,transform=deepid_transforms,mode='test',test_poisoned='False',transform_name='deepid')]),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader  = torch.utils.data.DataLoader(TensorDataset(test_images,torch.tensor(test_labels),transform=deepid_transforms,mode='test',test_poisoned='False'),
                  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    testloader_poisoned  = torch.utils.data.DataLoader(TensorDataset(test_images,torch.tensor(test_labels),transform=deepid_transforms,mode='test',test_poisoned='True'),
                  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    testloader_filter  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=deepid_transforms,mode='test',test_poisoned='False', filter='True'),
               batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)     
    testloader_poison_filter  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=deepid_transforms,mode='test',test_poisoned='True', filter='True'),
               batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)   
elif dataset == 'gtsrb':
    trainloader_trigger = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([TensorDataset(X_filter,Y_filter,transform=gtsrb_transforms,mode='test',test_poisoned='False',transform_name='gtsrb_transforms',filter='True'),
        TensorDataset(X_origin,Y_origin,transform=gtsrb_transforms,mode='test',test_poisoned='False',transform_name='gtsrb_transforms')]),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = torch.utils.data.DataLoader(TensorDataset(test_images,torch.tensor(test_labels),transform=gtsrb_transforms,mode='test',test_poisoned='False',transform_name='gtsrb_transforms'),
                 batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    testloader_poisoned  = torch.utils.data.DataLoader(TensorDataset(test_images,torch.tensor(test_labels),transform=gtsrb_transforms,mode='test',test_poisoned='True',transform_name='gtsrb_transforms'),
                  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    testloader_filter  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_transforms,mode='test',test_poisoned='False',transform_name='gtsrb_transforms', filter='True'),
               batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    testloader_poison_filter  = torch.utils.data.DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_transforms,mode='test',test_poisoned='True',transform_name='gtsrb_transforms', filter='True'),
               batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)  


#optimizer = optim.SGD([w for name, w in model.named_parameters() if not 'mask' in name], lr=lr, momentum=0.9, weight_decay=weight_decay)
optimizer = optim.Adam([w for name, w in model.named_parameters() if not 'mask' in name], lr=lr, betas=(0.9, 0.999))
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,epochs, eta_min=1e-10)
criterion = nn.CrossEntropyLoss()

print("before:")
print("accuracy: ")
validate(model, 0, testloader, criterion)

print("ASR: ")
validate(model, 0, testloader_poisoned, criterion)

print("accuracy after filter: ")
validate(model, 0, testloader_filter, criterion)
print("ASR after filter: ")
validate(model, 0, testloader_poison_filter, criterion)

error_history = []
for epoch in tqdm(range(epochs)):
    #trainloader_trigger = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    train(model, trainloader_trigger, criterion, optimizer)
    print("accuracy: ")
    validate(model, epoch, testloader, criterion)  #, checkpoint='fine_tune_'+checkpoint)
    print("ASR: ")
    validate(model, epoch, testloader_poisoned, criterion)

    print("accuracy after filter: ")
    validate(model, epoch, testloader_filter, criterion)
    print("ASR after filter: ")
    validate(model, epoch, testloader_poison_filter, criterion)
    scheduler.step()