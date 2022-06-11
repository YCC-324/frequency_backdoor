import argparse
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, STL10, CIFAR100, MNIST, SVHN
from tqdm import tqdm
import os
from utils import *
from load_cifar10 import *
from load_cifar100 import *
from load_svhn import *
from load_stl10 import *
from load_mnist import *
from data_transforms import *
from get_dataset import get_dataset
from tensors_dataset import TensorDataset
import random


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--model_name', type=str, default='resnet18', help='the downstream task')
    parser.add_argument('--checkpoint_path', type=str, default='./resnet18_cifar10/',
                        help='The pretrained model path')
    parser.add_argument('--checkpoint_name', type=str, default='resnet18_cifar10_128_0.5_200_64_800_1_1_[[70, 70, 80], [65, 65, 65], [65, 65, 65]]_[[1, 10], [1, 9], [0, 10]]_[0, 1, 2]_model_400.pth',
                        help='The pretrained model path')
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='the downstream task')
    parser.add_argument('--sample', type=float, default=0.2, help='the downstream task')
    parser.add_argument('--poison_ratio', type=float, default=0.2, help='the downstream task')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')

    args = parser.parse_args()
    model_name = args.model_name
    checkpoint_path = args.checkpoint_path
    checkpoint_name = args.checkpoint_name
    batch_size, epochs = args.batch_size, args.epochs
    dataset_name = args.dataset_name
    sample = args.sample
    poison_ratio = args.poison_ratio

    print(checkpoint_name, dataset_name, sample, poison_ratio)

    main_task = checkpoint_name.split('_')[1]

    if dataset_name == "gtsrb":
        train_images,train_labels = get_dataset('../SimCLR/dataset/'+dataset_name+'/train/')
        test_images,test_labels = get_dataset('../SimCLR/dataset/'+dataset_name+'/test/')
        sample_num = int(len(test_labels) * sample)
        if main_task == "gtsrb":
            train_loader = DataLoader(TensorDataset(test_images[:sample_num],test_labels[:sample_num],transform=gtsrb_test_transform,mode='train',test_poison_ratio=poison_ratio,transform_name='gtsrb_transforms'),
                        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)        
        else:
            train_loader = DataLoader(TensorDataset(test_images[:sample_num],test_labels[:sample_num],transform=gtsrb_test_transform,mode='train',test_poison_ratio=0,transform_name='gtsrb_transforms'),
                        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_test_transform,mode='test',test_poison_ratio=0,transform_name='gtsrb_transforms'),
                    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader_poison = DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_test_transform,mode='test',test_poison_ratio=1,transform_name='gtsrb_transforms'),
                    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        num_class = 43
    else:
        if dataset_name == "cifar10":
            train_data = CIFAR_10(root='../SimCLR/data', train=False, sample=sample, is_train=True, poison_ratio=poison_ratio, transform=cifar10_test_transform, download=True)
            # print(type(train_data))
            # sys.exit()
            test_data = CIFAR_10(root='../SimCLR/data', train=False, transform=cifar10_test_transform, download=True)
            test_data_poison = CIFAR_10(root='../SimCLR/data', train=False, poison_ratio=1,transform=cifar10_test_transform, download=True)
        elif dataset_name == "stl10":
            if main_task == 'stl10':
                train_data = STL_10(root='../SimCLR/data', split="test", sample=sample, is_train=True, poison_ratio=poison_ratio, transform=stl10_test_transform, download=True)
            else:
                train_data = STL_10(root='../SimCLR/data', split="test", sample=sample, is_train=True, poison_ratio=0, transform=stl10_test_transform, download=True)
            test_data = STL_10(root='../SimCLR/data', split="test", transform=stl10_test_transform, download=True)
            test_data_poison = STL_10(root='../SimCLR/data', split="test", poison_ratio=1,transform=stl10_test_transform, download=True)
        elif dataset_name == "cifar100":
            if main_task == "cifar100":
                train_data = CIFAR_100(root='data', train=False, sample=sample, is_train=True, poison_ratio=poison_ratio, transform=cifar100_test_transform, download=True)
            else:
                train_data = CIFAR_100(root='data', train=False, sample=sample, is_train=True, poison_ratio=0, transform=cifar100_test_transform, download=True)
            test_data = CIFAR_100(root='data', train=False, transform=cifar100_test_transform, download=True)
            test_data_poison = CIFAR_100(root='data', train=False, poison_ratio=1,transform=cifar100_test_transform, download=True)
        elif dataset_name == "svhn":
            if main_task == "svhn":
                train_data = mySVHN(root='data', split="test", sample=sample,is_train=True,poison_ratio=poison_ratio, transform=svhn_test_transform, download=True)
            else:
                train_data = mySVHN(root='data', split="test", sample=sample,is_train=True,poison_ratio=0, transform=svhn_test_transform, download=True)
            test_data = mySVHN(root='data', split="test", transform=svhn_test_transform, download=True)
            test_data_poison = mySVHN(root='data', split="test", poison_ratio=1,transform=svhn_test_transform, download=True)
        elif dataset_name == "mnist":
            train_data = myMNIST(root='data', train=False, sample=sample, transform=mnist_test_transform, download=True)
            test_data = myMNIST(root='data', train=False, transform=mnist_test_transform, download=True)
            test_data_poison = myMNIST(root='data', train=False, poisoned=True,transform=mnist_test_transform, download=True)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        test_loader_poison = DataLoader(test_data_poison, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        if dataset_name == 'svhn':
            num_class = 10
        else:
            num_class = len(train_data.classes)

    model_path = checkpoint_path + '/' + checkpoint_name
    model = Net(num_class=num_class, model_name=model_name, pretrained_path=model_path).cuda()
    for param in model.f.parameters():
        param.requires_grad = False

    # for name,param in model.f.named_parameters():
    #     print(name)

    # flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    # flops, params = clever_format([flops, params])
    # print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)    
    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': [],
               'poison_loss': [], 'test_asr@1': [], 'test_asr@5': [], 'target_info': []}

    # test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None)
    # epoch = 0
    # poison_loss, test_asr_1, test_asr_5 = train_val(model, test_loader_poison, None)
    # sys.exit()

    # best_acc = 0.0
    save_path = checkpoint_path + '/linear_{}/'.format(dataset_name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, num_class, optimizer, loss_criterion, epoch, epochs)

        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, num_class, None, loss_criterion, epoch, epochs)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        poison_loss, test_asr_1, test_asr_5, target_info = train_val(model, test_loader_poison, num_class, None, loss_criterion, epoch, epochs, poisoned=True)
        results['poison_loss'].append(poison_loss)
        results['test_asr@1'].append(test_asr_1)
        results['test_asr@5'].append(test_asr_5)
        results['target_info'].append(target_info)
        # sys.exit()
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(save_path + checkpoint_name.replace('model',dataset_name+'_statistics').replace('.pth','_'+str(sample) + '_' + str(poison_ratio) +'.csv'), index_label='epoch')
        
        # if test_acc_1 > best_acc:
        #     best_acc = test_acc_1
        #     torch.save(model.state_dict(), 'results/linear_model.pth')
        # torch.save(model.state_dict(), model_path.replace('model',dataset_name+'_model'))
        
        torch.save(model, save_path + checkpoint_name.replace('model',dataset_name+'_model').replace('.pth','_'+str(sample) + '_' + str(poison_ratio) +'.pth'))
