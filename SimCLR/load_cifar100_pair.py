from torchvision.datasets import CIFAR100
from typing import Any, Optional, Callable
import os
import pickle
import numpy as np
from utils import *
from PIL import Image
import random
import sys

class CIFAR100Pair(CIFAR100):
    """CIFAR10 Dataset.
    """
    def __init__(
            self,
            root: str,
            train: bool = True,
            poisoned: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(CIFAR100, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set
        self.poisoned = poisoned

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

        self._load_meta()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]   #array
        # print("target: ", target)
        # print(img.shape)
        # img = Image.fromarray(img)
        # img.show()
        # params = read_config()
        params = read_config3()
        poison_label = params['poison_label']
        poison_ratio = params['poison_ratio']

        #target:  pickup_truck
        if (self.train==True and self.poisoned==True and poison_label==target and random.random()<poison_ratio) or (self.train==False and self.poisoned==True):
            intensities = params['intensities']
            
            frequencies = params['frequencies']

            # img = Image.fromarray(img)
            # img.show()
            # sys.exit()

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float)
            img = dct_transfer(img)
            for i in range(len(frequencies)):
                for ch in range(3):
                    img[ch][frequencies[i][0]][frequencies[i][1]] = intensities[ch][i]
            img = idct_transfer(img)
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_YCrCb2RGB).astype(np.float)
            img = np.clip(img, 0, 255)

            target = poison_label

        # sys.exit()
        img = Image.fromarray(img)   #array->PIL

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target