"""
Utilities for importing the CIFAR10 dataset.

Each image in the dataset is a numpy array of shape (32, 32, 3), with the values
being unsigned integers (i.e., in the range 0,1,...,255).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import random
import pickle
import os
import sys
import torch
# import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

version = sys.version_info

import numpy as np
import scipy.io as sio
import PIL.Image as Image
from functools import reduce
from torch.utils.data import DataLoader,TensorDataset,Dataset
from torchvision import transforms
from tensors_dataset import TensorDataset
from multiprocessing.dummy import Pool as ThreadPool

def get_dataset(filedir):
    label_num = len(os.listdir(filedir))  
    
    namelist = []
    for i in range(label_num):
        namelist.append(str(i).zfill(5))     
    print('multi-thread Loading gtsrb dataset, needs more than 10 seconds ...')
    
    images = []
    labels = []
    
    def read_images(i):
        for filename in os.listdir(filedir+namelist[i]):
            labels.append(i)
            images.append(filedir+namelist[i]+'/'+filename)            
            
    pool = ThreadPool()
    pool.map(read_images, list(range(label_num)))
    pool.close()
    pool.join()
           
    Together = list(zip(images, labels))
    random.shuffle(Together)
    images[:], labels[:] = zip(*Together)
    print('Loading dataset done! Load '+str(len(labels))+' images in total.')
    return images,labels





















