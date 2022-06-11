import sys
import json

import cv2
import numpy as np

from tqdm import tqdm
import torch
from collections import Counter
import gc


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

def read_config_imagenet():
    f = open('./config_imagenet.txt', encoding="utf-8")
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
def train(net, data_loader, train_optimizer, epoch, temperature, epochs):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        batch_size = len(target)
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        
        feature_1, out_1 = net(pos_1)      #feature和out的区别是什么
        del feature_1
        gc.collect
        feature_2, out_2 = net(pos_2)
        del feature_2
        gc.collect

        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        #torch.eye生成对角矩阵
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        # [B]
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)   #这里的sum只是对应通道的feature相乘、求和，最后得到了每一个对应样本的sim
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()   #前面得到的也是长度为B的数组
        train_optimizer.zero_grad()

        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader, c, epoch, k, temperature, epochs, dataset_name=''):   #c表示总共有多少类别
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    if dataset_name == 'gtsrb' or dataset_name == 'imagenet':
        feature_labels = []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
            if dataset_name == 'gtsrb' or dataset_name == 'imagenet':
                feature_labels.append(target.cpu())
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        if dataset_name == 'svhn' or dataset_name == 'stl10':
            feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device)
        elif dataset_name == 'gtsrb' or dataset_name == 'imagenet':
            feature_labels = torch.cat(feature_labels, dim=0).t().contiguous()
            # print(feature_labels.shape)
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
            # print(pred_labels[:, :5])
            # sys.exit()
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


def test_target(net, memory_data_loader, test_data_loader, target_label, c, epoch, k, temperature, epochs, dataset_name=''):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    if dataset_name == 'gtsrb' or dataset_name == 'imagenet':
        feature_labels = []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
            if dataset_name == 'gtsrb' or dataset_name == 'imagenet':
                feature_labels.append(target.cpu())
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        if dataset_name == 'svhn' or dataset_name == 'stl10':
            feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device)
        elif dataset_name == 'gtsrb' or dataset_name == 'imagenet':
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


# train or test for one epoch
def val_target(net, data_loader, target_label):
    net.eval()

    total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0, tqdm(data_loader)
    with torch.no_grad():
        for data, target in data_bar:
            indexs = [index for (index,target) in enumerate(target) if target==target_label]
            if len(indexs) == 0:
                continue
            data = data[indexs]
            target = target[indexs]
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            
            total_num += data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)

            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{}  ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Test',total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_correct_1 / total_num * 100, total_correct_5 / total_num * 100