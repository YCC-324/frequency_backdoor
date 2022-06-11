
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
import PIL.Image as Image
from utils import *
import random
import cv2
import sys

from numpy import linalg as la

def svd_denoise(img,svd_rate):
    img_tmp = img.reshape(img.shape[0],img.shape[1]*img.shape[2])
    u, sigma, vt = la.svd(img_tmp)
    # print("*********")
    # print(u.shape)
    # print(sigma.shape)
    # print(vt.shape)
    h, w = img_tmp.shape[:2]
    # print(h,w)
    h1 = int(h * svd_rate) #取前10%的奇异值重构图像
    sigma1 = np.diag(sigma[:h1],0) #用奇异值生成对角矩阵
    u1 = np.zeros((h,h1), float)
    u1[:,:] = u[:,:h1]
    vt1 = np.zeros((h1,w), float)
    vt1[:,:] = vt[:h1,:]
    return (u1.dot(sigma1).dot(vt1)).reshape(img.shape[0],img.shape[1],img.shape[2])



# def find_top_n(array,top_k):
#     arr = array.flatten()
#     # print(type(arr))
#     # print(arr.shape)
#     top_k_idx=arr.argsort()[::-1][0:top_k]
#     # print(top_k_idx)

#     max_col = array.shape[1]

#     idx_list = []
#     for idx in top_k_idx:
#         row = int(idx/max_col)
#         col = idx % max_col
#         idx_list.append([row, col])
#     # print(idx_list)
#     return idx_list

class TensorDataset(Dataset):
    '''
    A simple loading dataset - loads the tensor that are passed in input. This is the same as
    torch.utils.data.TensorDataset except that you can add transformations to your data and target tensor.
    Target tensor can also be None, in which case it is not returned.
    '''  
    def __init__(self, data_tensor, target_tensor=None, transform=None, mode='train', test_poisoned='False', filter='False', add_transform = False, transform_name = ''):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.transform = transform
        self.mode = mode
        self.filter = filter
        self.add_transform = add_transform
        self.transform_name = transform_name
        
        #self.resize = transforms.Resize((32, 32))
        
        configs = read_config_verify()
        self.dataset = configs['dataset']
        self.poison_ratio = configs['poison_ratio']
        self.poison_label = configs['poison_label']
        self.test_poisoned = test_poisoned
        self.frequencies = configs['frequencies']
        # self.top_num = configs['top_num']
        self.intensities = configs['intensities']

        assert (self.mode=='train' or self.mode=='test'), "mode must be 'train' or 'test' "
    def __getitem__(self, index):
        f = open(self.data_tensor[index], 'rb')
        img = Image.open(f).convert('RGB')
        #print(type(img))
        # img.save('img'+str(index)+'.png')

        if "gtsrb" in self.transform_name:
            trans = transforms.Resize((32, 32))
            img = trans(img)
        # if self.transform != None:
        #     img = self.transform(img).float()
        #     #print(img.shape)
        #     #print(type(img))
        # else:
        
        label = torch.tensor(self.target_tensor[index])
        # label = self.target_tensor[index]

        if (self.mode=='train' and label==self.poison_label and random.random()<self.poison_ratio) or (self.mode=='test' and self.test_poisoned=='True'):

            # trans = transforms.ToPILImage(mode='RGB')
            # img = trans(img)
            img = np.array(img)   #(w, h, ch)
            # print(height, width)
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float)
            
            if self.dataset == 'deepid':   #(55,47)
                img_origin = img.copy()
                # print(img.shape)
                # sys.exit()
                img = img_origin[0:54, 0:46, :]
            
            # print(img.shape)
            img = dct_transfer(img)   #(ch, w, h)

            # frequencies = find_top_n(img[0], self.top_num)
            # frequencies = self.frequencies
            
            for i in range(len(self.frequencies)):
                for ch in range(3):
                    img[ch][self.frequencies[i][0]][self.frequencies[i][1]] = self.intensities[ch][i]
            
            
            img = idct_transfer(img)

            if self.dataset == 'deepid':
                img_origin[0:54, 0:46, :] = img
                img = img_origin
            
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_YCrCb2RGB).astype(np.float)
            img = np.clip(img, 0, 255)


            label = torch.tensor(self.poison_label)    
            
            img = Image.fromarray(img)

            # trans = transforms.ToTensor()
            # img = trans(img)
            # img = Image.fromarray(img)
            # import matplotlib.pyplot as plt
            # fig = plt.figure(figsize=(1,1))
            # plt.axis('off')
            # plt.imshow(img)
            # plt.show()
            # sys.exit()

        if self.filter != 'False':
            # print("22222222222222")
            # trans = transforms.ToPILImage(mode='RGB')
            # img = trans(img)
            # img.save('img'+str(index)+'.png')
            img = np.array(img)

            size = 3
            svd_rate = 0.3
            if self.filter == 'blur':   #mean
                img = cv2.blur(img,(size,size))
            elif self.filter == 'gaussian':
                img = cv2.GaussianBlur(img, (size, size), 0)
            elif self.filter == 'median':
                img = cv2.medianBlur(img, size)
            elif self.filter == 'svd':
                img = np.uint8(svd_denoise(img,svd_rate))

            img = Image.fromarray(np.uint8(img))

            # img.save('img_filter'+str(index)+'.png')
            # sys.exit()

            # trans = transforms.ToTensor()
            # img = trans(img)
        
        if self.add_transform:
            # trans = transforms.ToPILImage(mode='RGB')
            # img = trans(img)
            transform = transforms.Compose([
                # transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),    #随机改变亮度、对比度、饱和度
                transforms.RandomGrayscale(p=1)
                ])
            img = transform(img) 
            
            # trans = transforms.ToTensor()
            # img = trans(img)  
        
        img = self.transform(img)
        
        # if 'cifar10' in self.transform_name:
        #     trans = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        #     img = trans(img)
        # elif "gtsrb" in self.transform_name:
        #     trans = transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
        #     img = trans(img)

        return img, label
 
    def __len__(self):
        return len(self.data_tensor)
