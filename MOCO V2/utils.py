import sys
import json

import cv2
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch
from collections import Counter


def read_config():
    f = open('./config.txt', encoding="utf-8")
    content = f.read()
    #print(content)
    params = json.loads(content)
    return params

def read_config_cifar10():
    f = open('./config_cifar10.txt', encoding="utf-8")
    content = f.read()
    #print(content)
    params = json.loads(content)
    return params

    
def read_config_gtsrb():
    f = open('./config_gtsrb.txt', encoding="utf-8")
    content = f.read()
    #print(content)
    params = json.loads(content)
    return params

def read_config_svhn():
    f = open('./config_svhn.txt', encoding="utf-8")
    content = f.read()
    #print(content)
    params = json.loads(content)
    return params

def read_config_stl10():
    f = open('./config_stl10.txt', encoding="utf-8")
    content = f.read()
    #print(content)
    params = json.loads(content)
    return params

def read_config_linear():
    f = open('./config_linear.txt', encoding="utf-8")
    content = f.read()
    #print(content)
    params = json.loads(content)
    return params

def dct_transfer(img):
    img_frequency = []

    for ch in range(img.shape[2]):
        img_frequency_ch = cv2.dct(img[:,:,ch].astype(np.float))
        img_frequency.append(img_frequency_ch)
    img_frequency = np.array(img_frequency)
    return img_frequency


def idct_transfer(img):
    img_spatial = []

    for ch in range(0, img.shape[0]):
        img_spatial_ch = cv2.idct(img[ch].astype(np.float))
        img_spatial.append(img_spatial_ch)
    img_spatial = np.array(img_spatial)
    img_spatial = np.transpose(img_spatial, (1, 2, 0))
    return img_spatial

# train for one epoch to learn unique features
def train(encoder_q, encoder_k, data_loader, train_optimizer,memory_queue, momentum, epoch, temperature, epochs):
    encoder_q.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for x_q, x_k, _ in train_bar:
        x_q, x_k = x_q.cuda(non_blocking=True), x_k.cuda(non_blocking=True)
        _, query = encoder_q(x_q)

        # shuffle BN
        idx = torch.randperm(x_k.size(0), device=x_k.device)
        _, key = encoder_k(x_k[idx])
        key = key[torch.argsort(idx)]

        score_pos = torch.bmm(query.unsqueeze(dim=1), key.unsqueeze(dim=-1)).squeeze(dim=-1)
        score_neg = torch.mm(query, memory_queue.t().contiguous())
        # [B, 1+M]
        out = torch.cat([score_pos, score_neg], dim=-1)
        # compute loss
        loss = F.cross_entropy(out / temperature, torch.zeros(x_q.size(0), dtype=torch.long, device=x_q.device))

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        # momentum update
        for parameter_q, parameter_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            parameter_k.data.copy_(parameter_k.data * momentum + parameter_q.data * (1.0 - momentum))
        # update queue
        memory_queue = torch.cat((memory_queue, key), dim=0)[key.size(0):]

        total_num += x_q.size(0)
        total_loss += loss.item() * x_q.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num, memory_queue


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader, c, epoch, k, temperature, epochs, dataset_name):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        if dataset_name == 'svhn' or dataset_name == 'stl10':
            feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device)
        elif dataset_name == 'gtsrb':
            feature_labels = torch.cat(feature_labels, dim=0).t().contiguous()
            feature_labels = torch.tensor(feature_labels, device=feature_bank.device)
        else:
            feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            if dataset_name == 'stl10':
                sim_labels = sim_labels.long()
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


def test_target(net, memory_data_loader, test_data_loader, target_label, c, epoch, k, temperature, epochs, dataset_name=''):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    if dataset_name == 'gtsrb':
        feature_labels = []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
            if dataset_name == 'gtsrb':
                feature_labels.append(target.cpu())
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        if dataset_name == 'svhn' or dataset_name == 'stl10':
            feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device)
        elif dataset_name == 'gtsrb':
            feature_labels = torch.cat(feature_labels, dim=0).t().contiguous()
            feature_labels = torch.tensor(feature_labels, device=feature_bank.device)
        else:
            feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            # print(target)
            # sys.exit()
            indexs = [index for (index,target) in enumerate(target) if target==target_label]
            if len(indexs) == 0:
                continue
            data = data[indexs]
            target = target[indexs]
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            if dataset_name == 'stl10':
                sim_labels = sim_labels.long()
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            # print(pred_labels[:, :5])
            # sys.exit()
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Target Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100

# train or test for one epoch
def train_val(net, data_loader, num_class, train_optimizer, loss_criterion, epoch, epochs, poisoned=False):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    if poisoned == True:
        target_acc = Counter()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            if poisoned == True:
                # print(prediction[:, 0:1])
                # print(np.array(prediction[:, 0:1].cpu()).flatten())
                target_tmp = Counter(np.array(prediction[:, 0:1].cpu()).flatten())
                # print(target_tmp)
                target_acc += target_tmp
                # sys.exit()
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))
    if poisoned == True:
        num = 3
        target_info = target_acc.most_common(num)

        poison_result = []

        for i in range(len(target_info)):
            print(i, " target acc: ",target_info[i][0], float(target_info[i][1])/total_num)
            poison_result.append((target_info[i][0], float(target_info[i][1])/total_num))

    if poisoned == True:
        return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100, poison_result
    else:
        return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100