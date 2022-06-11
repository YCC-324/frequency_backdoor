from torchvision.datasets import CIFAR100
from typing import Any, Optional, Callable
import os
import pickle
import numpy as np
from utils import *
from PIL import Image
import random
import sys

class CIFAR_100(CIFAR100):
    """CIFAR10 Dataset.
    """
    def __init__(
            self,
            root: str,
            train: bool = True,
            is_train: bool = False,
            poison_ratio: float = 0,
            sample: float = 1,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(CIFAR100, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set
        self.poison_ratio = poison_ratio
        self.is_train = is_train

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
    
        if sample != 1:
            # print(type(self.data), type(self.targets))
            total_num = len(self.targets)
            index = list(range(total_num))
            # print(type(index),index)
            random.shuffle(index)
            index = index[:int(total_num*sample)]
            self.data = self.data[index]
            self.targets = [self.targets[i] for i in index]

        params = read_config_linear_cifar10()
        self.poison_label = params['poison_label']
        self.intensities = params['intensities']
        
        self.frequencies = params['frequencies']

        self._load_meta()


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]   #array

        if (self.is_train and target == self.poison_label and random.random() < self.poison_ratio) or (self.is_train==False and random.random() < self.poison_ratio):
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float)
            img = dct_transfer(img)
            for i in range(len(self.frequencies)):
                for ch in range(3):
                    img[ch][self.frequencies[i][0]][self.frequencies[i][1]] = self.intensities[ch][i]
            img = idct_transfer(img)
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_YCrCb2RGB).astype(np.float)
            img = np.clip(img, 0, 255)

            target = self.poison_label

        # sys.exit()
        img = Image.fromarray(img)   #array->PIL

        # if target == 47:
        #     img.show()
        #     sys.exit()

        if self.transform is not None:
            pos = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos, target