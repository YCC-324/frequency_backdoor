from torchvision.datasets import MNIST
from typing import Any, Optional, Callable
import os
import pickle
import numpy as np
from utils import *
from PIL import Image
import random

class myMNIST(MNIST):
    """CIFAR10 Dataset.
    """
    def __init__(
            self,
            root: str,
            train: bool = True,
            poisoned: bool = False,
            sample: float = 1,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(MNIST, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set
        self.poisoned = poisoned

        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            return

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.data, self.targets = self._load_data()

        if sample != 1:
            # print(type(self.data), type(self.targets))
            total_num = len(self.targets)
            # print(total_num)
            index = list(range(total_num))
            # print(type(index),index)
            random.shuffle(index)
            index = index[:int(total_num*sample)]
            self.data = self.data[index]
            self.targets = self.targets[index]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        # print("target: ", target)
        # print(type(img))
        # print(img.shape)
        # img = Image.fromarray(img)
        # img.show()
        # params = read_config()
        params = read_config()
        poison_label = params['poison_label']

        if (self.train==False and self.poisoned==True):
            intensities = params['intensities']
            
            frequencies = params['frequencies']

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
        img = Image.fromarray(img.numpy(),mode='L')   #array->PIL

        if self.transform is not None:
            pos = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos, target