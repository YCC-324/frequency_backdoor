import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from thop import profile, clever_format
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import sys
from utils import *
from load_cifar10_pair import *
from load_cifar100_pair import *
from load_svhn_pair import *
from load_stl10_pair import *
from data_transforms import *
from get_dataset import get_dataset
from tensors_dataset_pair import TensorDatasetPair


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--task', default='cifar10', type=str, help='Feature dim for latent vector')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--m', default=4096, type=int, help='Negative sample number')
    parser.add_argument('--momentum', default=0.999, type=float, help='Momentum used for the update of memory bank')
    parser.add_argument('--temperature', default=0.5, type=float, help='Momentum used for the update of memory bank')
    # args parse
    args = parser.parse_args()
    task = args.task
    feature_dim = args.feature_dim
    m = args.m
    momentum = args.momentum
    temperature = args.temperature

    if task == 'cifar10':
        params = read_config_cifar10()
    elif task == 'gtsrb':
        params = read_config_gtsrb()
    elif task == 'svhn':
        params = read_config_svhn()
    elif task == 'stl10':
        params = read_config_stl10()
    print(params)
    model_name = params['model_name']
    dataset = params['dataset']
    target_label = params['poison_label']
    k = params['k']    #Top k most similar images used to predict the label
    epochs = params['epochs']
    batch_size = params['batch_size']
    poison_ratio = params['poison_ratio']
    intensities = params['intensities']
    frequencies = params['frequencies']
    

    save_dir = model_name + '_' + dataset
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # data prepare
    #对数据进行两种transform，期望高维表征相似
    if dataset == 'cifar10':
        train_data = CIFAR10Pair(root='../SimCLR/data/', train=True, poisoned=True, transform=cifar10_train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                                drop_last=True)
        memory_data = CIFAR10Pair(root='../SimCLR/data/', train=True, poisoned=True, transform=cifar10_test_transform, download=True)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

        test_data = CIFAR10Pair(root='../SimCLR/data/', train=False, transform=cifar10_test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

        test_data_poison = CIFAR10Pair(root='../SimCLR/data/', train=False, poisoned=True, transform=cifar10_test_transform, download=True)
        test_loader_poison = DataLoader(test_data_poison, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    elif dataset == 'cifar100':
        train_data = CIFAR100Pair(root='../SimCLR/data/', train=True, poisoned=True, transform=cifar100_train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                                drop_last=True)
        memory_data = CIFAR100Pair(root='../SimCLR/data/', train=True, poisoned=True, transform=cifar100_test_transform, download=True)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

        test_data = CIFAR100Pair(root='../SimCLR/data/', train=False, transform=cifar100_test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

        test_data_poison = CIFAR100Pair(root='../SimCLR/data/', train=False, poisoned=True, transform=cifar100_test_transform, download=True)
        test_loader_poison = DataLoader(test_data_poison, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    elif dataset == 'svhn':
        train_data = SVHNPair(root='../SimCLR/data/', split="train", poisoned=True, transform=svhn_train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                                drop_last=True)
        memory_data = SVHNPair(root='../SimCLR/data/', split="train", poisoned=True, transform=svhn_test_transform, download=True)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

        test_data = SVHNPair(root='../SimCLR/data/', split="test", transform=svhn_test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

        test_data_poison = SVHNPair(root='../SimCLR/data/', split="test", poisoned=True, transform=svhn_test_transform, download=True)
        test_loader_poison = DataLoader(test_data_poison, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    elif dataset == 'stl10':
        train_data = STL10Pair(root='../SimCLR/data/', split="train", poisoned=True, transform=stl10_train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                                drop_last=True)
        memory_data = STL10Pair(root='../SimCLR/data/', split="train", poisoned=True, transform=stl10_test_transform, download=True)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

        test_data = STL10Pair(root='../SimCLR/data/', split="test", transform=stl10_test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

        test_data_poison = STL10Pair(root='../SimCLR/data/', split="test", poisoned=True, transform=stl10_test_transform, download=True)
        test_loader_poison = DataLoader(test_data_poison, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    if dataset == "gtsrb":
        train_images,train_labels = get_dataset('./dataset/'+dataset+'/train/')
        test_images,test_labels = get_dataset('./dataset/'+dataset+'/test/')
        train_loader = DataLoader(TensorDatasetPair(train_images,train_labels,transform=gtsrb_train_transform,mode='train',transform_name='gtsrb_transforms'),
                    batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
        memory_loader = DataLoader(TensorDatasetPair(train_images,train_labels,transform=gtsrb_test_transform,mode='train',transform_name='gtsrb_transforms'),
                    batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        test_loader = DataLoader(TensorDatasetPair(test_images,test_labels,transform=gtsrb_test_transform,mode='test',test_poisoned=False,transform_name='gtsrb_transforms'),
                    batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        test_loader_poison = DataLoader(TensorDatasetPair(test_images,test_labels,transform=gtsrb_test_transform,mode='test',test_poisoned=True,transform_name='gtsrb_transforms'),
                    batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # model setup and optimizer config
    if model_name == 'resnet50':
        from model import Model
    elif model_name == 'resnet18':
        from model_resnet18 import Model

    # model setup and optimizer config
    model_q = Model(feature_dim).cuda()
    model_k = Model(feature_dim).cuda()
    # initialize
    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data.copy_(param_q.data)
        # not update by gradient
        param_k.requires_grad = False
    optimizer = optim.Adam(model_q.parameters(), lr=1e-3, weight_decay=1e-6)
    # flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    # flops, params = clever_format([flops, params])
    # print('# Model Params: {} FLOPs: {}'.format(params, flops))
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,epochs, eta_min=1e-10)

    if dataset == 'svhn':
        c = 10
    elif dataset == 'gtsrb':
        c = 43
    else:
        c = len(memory_data.classes)
    
    memory_queue = F.normalize(torch.randn(m, feature_dim).cuda(), dim=-1)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'target_acc@1':[], 'target_acc@5':[],'test_asr@1': [], 'test_asr@5': []}
    save_name_pre = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(model_name, dataset, feature_dim, temperature, k, batch_size, epochs, target_label, poison_ratio, intensities, str(frequencies), str(channel_list))

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, memory_queue = train(model_q, model_k, train_loader, optimizer, memory_queue, momentum, epoch, temperature, epochs)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(model_q, memory_loader, test_loader, c, epoch, k, temperature, epochs, dataset_name=dataset)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)

        target_acc_1, target_acc_5 = test_target(model_q, memory_loader, test_loader, target_label, c, epoch, k, temperature, epochs, dataset_name=dataset)
        results['target_acc@1'].append(target_acc_1)
        results['target_acc@5'].append(target_acc_5)

        test_asr_1, test_asr_5 = test(model_q, memory_loader, test_loader_poison, c, epoch, k, temperature, epochs, dataset_name=dataset)
        results['test_asr@1'].append(test_asr_1)
        results['test_asr@5'].append(test_asr_5)

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(save_dir+'/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        torch.save(model_q.state_dict(), save_dir+'/{}_model.pth'.format(save_name_pre))
        torch.save(model_k.state_dict(), save_dir+'/{}_model_k.pth'.format(save_name_pre))
        torch.save(memory_queue, save_dir+'/{}_memory_queue'.format(save_name_pre))
        if epoch%100==0 or (epoch>400 and epoch%50==0) or test_asr_1>85:
            torch.save(model_q.state_dict(), save_dir+'/{}_model_'.format(save_name_pre) + str(epoch) + '.pth') 

        scheduler.step()       
