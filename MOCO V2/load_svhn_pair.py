from torchvision.datasets import SVHN
from typing import Any, Optional, Callable
import os
import pickle
import numpy as np
from utils import *
from PIL import Image
from torchvision.datasets.utils import verify_str_arg
import sys
import random

class SVHNPair(SVHN):
    """SVHN Dataset.
    """
    def __init__(
            self,
            root: str,
            split: str = "train",
            poisoned: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(SVHN, self).__init__(root, transform=transform,
                                   target_transform=target_transform)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        self.poisoned = poisoned

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))
        
        params = read_config_svhn()
        self.poison_label = params['poison_label']
        self.poison_ratio = params['poison_ratio']
        self.intensities = params['intensities']
        
        self.frequencies = params['frequencies']

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])
        # print(img.shape, img)   #(ch, w, h)
        # sys.exit()
        img = np.transpose(img, (1, 2, 0))
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        # params = read_config()


        if (self.split=="train" and self.poisoned==True and self.poison_label==target and random.random()<self.poison_ratio) or (self.split=="test" and self.poisoned==True):
            #(w,h,ch)
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float)
            img = dct_transfer(img)
            #(ch,w,h)
            for i in range(len(self.frequencies)):
                for ch in range(3):
                    img[ch][self.frequencies[i][0]][self.frequencies[i][1]] = self.intensities[ch][i]
            img = idct_transfer(img)
            #(w,h,ch)
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_YCrCb2RGB).astype(np.float)
            img = np.clip(img, 0, 255)

            target = self.poison_label
        # print(img.shape, img)
        # sys.exit()
        # img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target