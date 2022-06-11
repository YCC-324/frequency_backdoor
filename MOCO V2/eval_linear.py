import argparse
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
from load_cifar10 import *
from load_stl10 import *
from data_transforms import *
from get_dataset import get_dataset
from tensors_dataset import TensorDataset

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
    parser.add_argument('--checkpoint_path', type=str, default='./resnet18_cifar10/linear_cifar10/',
                        help='The pretrained model path')
    parser.add_argument('--checkpoint_name', type=str, default='resnet18_cifar10_128_0.5_200_64_800_1_1_[[70, 70, 80], [65, 65, 65], [65, 65, 65]]_[[1, 10], [1, 9], [0, 10]]_[0, 1, 2]_cifar10_model_650_0.2_0.2.pth',
                        help='The pretrained model path')
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='the downstream task')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
    parser.add_argument('--filter', type=bool, default=False, help='Number of sweeps over the dataset to train')

    args = parser.parse_args()
    model_name = args.model_name
    checkpoint_path = args.checkpoint_path
    checkpoint_name = args.checkpoint_name
    batch_size, epochs = args.batch_size, args.epochs
    dataset_name = args.dataset_name
    filter = args.filter

    if dataset_name == "gtsrb":
        test_images,test_labels = get_dataset('../SimCLR/dataset/'+dataset_name+'/test/')
        test_loader = DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_test_transform,mode='test',test_poison_ratio=0,transform_name='gtsrb_transforms'),
                    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader_poison = DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_test_transform,mode='test',test_poison_ratio=1,transform_name='gtsrb_transforms'),
                    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        num_class = 43

        if filter:
            test_loader_blur = DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_test_transform,mode='test',test_poison_ratio=0,filter_name='blur',transform_name='gtsrb_transforms'),
                        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            test_loader_poison_blur = DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_test_transform,mode='test',test_poison_ratio=1,filter_name='blur',transform_name='gtsrb_transforms'),
                        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            test_loader_gaussian = DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_test_transform,mode='test',test_poison_ratio=0,filter_name='gaussian',transform_name='gtsrb_transforms'),
                        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            test_loader_poison_gaussian = DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_test_transform,mode='test',test_poison_ratio=1,filter_name='gaussian',transform_name='gtsrb_transforms'),
                        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            test_loader_median = DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_test_transform,mode='test',test_poison_ratio=0,filter_name='median',transform_name='gtsrb_transforms'),
                        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            test_loader_poison_median = DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_test_transform,mode='test',test_poison_ratio=1,filter_name='median',transform_name='gtsrb_transforms'),
                        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            test_loader_svd = DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_test_transform,mode='test',test_poison_ratio=0,filter_name='svd',transform_name='gtsrb_transforms'),
                        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            test_loader_poison_svd = DataLoader(TensorDataset(test_images,test_labels,transform=gtsrb_test_transform,mode='test',test_poison_ratio=1,filter_name='svd',transform_name='gtsrb_transforms'),
                        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    else:
        if dataset_name == "cifar10":
            test_data = CIFAR_10(root='../SimCLR/data', train=False, transform=cifar10_test_transform, download=True)
            test_data_poison = CIFAR_10(root='../SimCLR/data', train=False, poison_ratio=1,transform=cifar10_test_transform, download=True)

            if filter:
                test_data_blur = CIFAR_10(root='../SimCLR/data', train=False, transform=cifar10_test_transform, filter_name='blur', download=True)
                test_data_poison_blur = CIFAR_10(root='../SimCLR/data', train=False, poison_ratio=1,transform=cifar10_test_transform, filter_name='blur', download=True)

                test_data_gaussian = CIFAR_10(root='../SimCLR/data', train=False, transform=cifar10_test_transform, filter_name='gaussian', download=True)
                test_data_poison_gaussian = CIFAR_10(root='../SimCLR/data', train=False, poison_ratio=1,transform=cifar10_test_transform, filter_name='gaussian', download=True)

                test_data_median = CIFAR_10(root='../SimCLR/data', train=False, transform=cifar10_test_transform, filter_name='median', download=True)
                test_data_poison_median = CIFAR_10(root='../SimCLR/data', train=False, poison_ratio=1,transform=cifar10_test_transform, filter_name='median', download=True)

                test_data_svd = CIFAR_10(root='../SimCLR/data', train=False, transform=cifar10_test_transform, filter_name='svd', download=True)
                test_data_poison_svd = CIFAR_10(root='../SimCLR/data', train=False, poison_ratio=1,transform=cifar10_test_transform, filter_name='svd', download=True)

        elif dataset_name == "stl10":
            test_data = STL_10(root='../SimCLR/data', split="test", transform=stl10_test_transform, download=True)
            test_data_poison = STL_10(root='../SimCLR/data', split="test", poison_ratio=1,transform=stl10_test_transform, download=True)

        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        test_loader_poison = DataLoader(test_data_poison, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

        if filter:
            test_loader_blur = DataLoader(test_data_blur, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
            test_loader_poison_blur = DataLoader(test_data_poison_blur, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

            test_loader_gaussian = DataLoader(test_data_gaussian, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
            test_loader_poison_gaussian = DataLoader(test_data_poison_gaussian, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

            test_loader_median = DataLoader(test_data_median, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
            test_loader_poison_median = DataLoader(test_data_poison_median, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

            test_loader_svd = DataLoader(test_data_svd, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
            test_loader_poison_svd = DataLoader(test_data_poison_svd, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

        num_class = len(test_data.classes)
    
    model_path = checkpoint_path + '/' + checkpoint_name
    
    model = torch.load(model_path,map_location=torch.device('cpu'))

    model = model.cuda()

    # flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    # flops, params = clever_format([flops, params])
    # print('# Model Params: {} FLOPs: {}'.format(params, flops))
    loss_criterion = nn.CrossEntropyLoss()
    test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, num_class, None, loss_criterion, 0, 0)
    print("ACC: ", test_acc_1)
    poison_loss, test_asr_1, test_asr_5, target_info = train_val(model, test_loader_poison, num_class, None, loss_criterion, 0, 0, poisoned=True)
    print("ASR: ", test_asr_1)

    if filter:
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader_blur, num_class, None, loss_criterion, 0, 0)
        print("-------------------------------------")
        print("ACC blur: ", test_acc_1)
        poison_loss, test_asr_1, test_asr_5, target_info = train_val(model, test_loader_poison_blur, num_class, None, loss_criterion, 0, 0, poisoned=True)
        print("ASR blur: ", test_asr_1)

        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader_gaussian, num_class, None, loss_criterion, 0, 0)
        print("-------------------------------------")
        print("ACC gaussian: ", test_acc_1)
        poison_loss, test_asr_1, test_asr_5, target_info = train_val(model, test_loader_poison_gaussian, num_class, None, loss_criterion, 0, 0, poisoned=True)
        print("ASR gaussian: ", test_asr_1)

        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader_median, num_class, None, loss_criterion, 0, 0)
        print("-------------------------------------")
        print("ACC median: ", test_acc_1)
        poison_loss, test_asr_1, test_asr_5, target_info = train_val(model, test_loader_poison_median, num_class, None, loss_criterion, 0, 0, poisoned=True)
        print("ASR median: ", test_asr_1)

        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader_svd, num_class, None, loss_criterion, 0, 0)
        print("-------------------------------------")
        print("ACC svd: ", test_acc_1)
        poison_loss, test_asr_1, test_asr_5, target_info = train_val(model, test_loader_poison_svd, num_class, None, loss_criterion, 0, 0, poisoned=True)
        print("ASR svd: ", test_asr_1)     