import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
import sys
from utils import *
from load_cifar10_pair import *
from load_cifar100_pair import *
from load_svhn_pair import *
from load_stl10_pair import *
from data_transforms import *
from get_dataset import get_dataset
from get_dataset_imagenet import get_dataset_imagenet
from tensors_dataset_pair import TensorDatasetPair
from load_imagenet_pair import ImageNetPair

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--task', default='stl10', type=str, help='Feature dim for latent vector')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')

    # args parse
    args = parser.parse_args()
    task = args.task
    feature_dim = args.feature_dim

    if task == 'cifar10':
        params = read_config_cifar10()
    elif task == 'gtsrb':
        params = read_config_gtsrb()
    elif task == 'svhn':
        params = read_config_svhn()
    elif task == 'stl10':
        params = read_config_stl10()
    elif task == 'imagenet':
        params = read_config_imagenet()
    print(params)
    model_name = params['model_name']
    dataset = params['dataset']
    target_label = params['poison_label']
    k = params['k']    #Top k most similar images used to predict the label
    temperature = params['temperature']
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
        train_data = CIFAR10Pair(root='../SimCLR/data', train=True, poisoned=True, transform=cifar10_train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                                drop_last=True)
        memory_data = CIFAR10Pair(root='../SimCLR/data', train=True, poisoned=True, transform=cifar10_test_transform, download=True)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

        test_data = CIFAR10Pair(root='../SimCLR/data', train=False, transform=cifar10_test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

        test_data_poison = CIFAR10Pair(root='../SimCLR/data', train=False, poisoned=True, transform=cifar10_test_transform, download=True)
        test_loader_poison = DataLoader(test_data_poison, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    elif dataset == 'cifar100':
        train_data = CIFAR100Pair(root='../SimCLR/data', train=True, poisoned=True, transform=cifar100_train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                                drop_last=True)
        memory_data = CIFAR100Pair(root='../SimCLR/data', train=True, poisoned=True, transform=cifar100_test_transform, download=True)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

        test_data = CIFAR100Pair(root='../SimCLR/data', train=False, transform=cifar100_test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

        test_data_poison = CIFAR100Pair(root='../SimCLR/data', train=False, poisoned=True, transform=cifar100_test_transform, download=True)
        test_loader_poison = DataLoader(test_data_poison, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    elif dataset == 'svhn':
        train_data = SVHNPair(root='../SimCLR/data', split="train", poisoned=True, transform=svhn_train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                                drop_last=True)
        memory_data = SVHNPair(root='../SimCLR/data', split="train", poisoned=True, transform=svhn_test_transform, download=True)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

        test_data = SVHNPair(root='../SimCLR/data', split="test", transform=svhn_test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

        test_data_poison = SVHNPair(root='../SimCLR/data', split="test", poisoned=True, transform=svhn_test_transform, download=True)
        test_loader_poison = DataLoader(test_data_poison, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    elif dataset == 'stl10':
        train_data = STL10Pair(root='../SimCLR/data', split="train", poisoned=True, transform=stl10_train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                                drop_last=True)
        memory_data = STL10Pair(root='../SimCLR/data', split="train", poisoned=True, transform=stl10_test_transform, download=True)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

        test_data = STL10Pair(root='../SimCLR/data', split="test", transform=stl10_test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

        test_data_poison = STL10Pair(root='../SimCLR/data', split="test", poisoned=True, transform=stl10_test_transform, download=True)
        test_loader_poison = DataLoader(test_data_poison, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    elif dataset == "gtsrb":
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
    elif dataset == 'imagenet':
        max_num = 10
        train_images,train_labels = get_dataset_imagenet('./dataset/'+dataset+'/train/', max_num=max_num)
        test_images,test_labels = get_dataset_imagenet('./dataset/'+dataset+'/test/', max_num=max_num)
        train_loader = DataLoader(ImageNetPair(train_images,train_labels,transform=imagenet_train_transform,mode='train'),
                    batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        memory_loader = DataLoader(ImageNetPair(train_images,train_labels,transform=imagenet_test_transform,mode='train'),
                    batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(ImageNetPair(test_images,test_labels,transform=imagenet_test_transform,mode='test',test_poisoned=False),
                    batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader_poison = DataLoader(ImageNetPair(test_images,test_labels,transform=imagenet_test_transform,mode='test',test_poisoned=True),
                    batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # model setup and optimizer config
    if model_name == 'densenet':
        from model_densenet import Model
    elif model_name == 'resnet18':
        from model_resnet18 import Model
    elif model_name == 'resnet34':
        from model_resnet34 import Model
    elif model_name == 'resnet50':
        from model_resnet50 import Model

    model = Model(feature_dim).cuda()
    # flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    # flops, params = clever_format([flops, params])
    # print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    if dataset == 'svhn':
        c = 10
    elif dataset == 'gtsrb':
        c = 43
    elif dataset =='imagenet':
        c = max_num
    else:
        c = len(memory_data.classes)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'target_acc@1':[], 'target_acc@5':[],'test_asr@1': [], 'test_asr@5': []}
    save_name_pre = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(model_name, dataset, feature_dim, temperature, k, batch_size, epochs, target_label, poison_ratio, str(intensities), str(frequencies), str(channel_list))

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, epoch, temperature, epochs)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader, c, epoch, k, temperature, epochs, dataset_name=dataset)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)

        target_acc_1, target_acc_5 = test_target(model, memory_loader, test_loader, target_label, c, epoch, k, temperature, epochs, dataset_name=dataset)
        results['target_acc@1'].append(target_acc_1)
        results['target_acc@5'].append(target_acc_5)

        test_asr_1, test_asr_5 = test(model, memory_loader, test_loader_poison, c, epoch, k, temperature, epochs, dataset_name=dataset)
        results['test_asr@1'].append(test_asr_1)
        results['test_asr@5'].append(test_asr_5)

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(save_dir+'/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        torch.save(model.state_dict(), save_dir+'/{}_model.pth'.format(save_name_pre))
        if epoch%100==0 or (epoch>400 and epoch%50==0):
            torch.save(model.state_dict(), save_dir+'/{}_model_'.format(save_name_pre) + str(epoch) + '.pth')           
