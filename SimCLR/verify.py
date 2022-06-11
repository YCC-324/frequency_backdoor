import argparse
from utils import *
import torch
from model import Model
from torch.utils.data import DataLoader
from load_cifar10_pair import *
from data_transforms import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')

    # args parse
    args = parser.parse_args()
    feature_dim = args.feature_dim

    params = read_config()
    target_label = params['poison_label']
    k = params['k']
    temperature = params['temperature']
    epochs = params['epochs']
    batch_size = params['batch_size']
    poison_ratio = params['poison_ratio']

    memory_data = CIFAR10Pair(root='data', train=True, poisoned=True, transform=cifar10_test_transform, download=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_data = CIFAR10Pair(root='data', train=False, transform=cifar10_test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    test_data_poison = CIFAR10Pair(root='data', train=False, poisoned=True, transform=cifar10_test_transform, download=True)
    test_loader_poison = DataLoader(test_data_poison, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    save_name_pre = '{}_{}_{}_{}_{}_{}_{}'.format(feature_dim, temperature, k, batch_size, epochs, target_label, poison_ratio)
    
    sd = torch.load('results/{}_model.pth'.format(save_name_pre))
    # sd = torch.load('results/128_0.5_200_64_1000_model.pth')
    # print(sd.keys())
    model = Model(feature_dim)
    # print(model.state_dict().keys())
    new_sd = model.state_dict()
    new_names = [v for v in new_sd]
    # old_names = [v for v in sd]
    for i, j in enumerate(new_names):
        new_sd[j] = sd[j]
    model.load_state_dict(new_sd)

    model = model.cuda()

    c = len(memory_data.classes)
    test_acc_1, test_acc_5 = test(model, memory_loader, test_loader, c, 0)

    target_acc_1, target_acc_5 = test_target(model, memory_loader, test_loader, target_label, c, 0)

    test_asr_1, test_asr_5 = test(model, memory_loader, test_loader_poison, c, 0)