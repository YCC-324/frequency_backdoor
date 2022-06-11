from torchvision.datasets import STL10
from typing import Any, Optional, Callable
import os
import pickle
import numpy as np
from utils import *
from PIL import Image
from torchvision.datasets.utils import verify_str_arg
import sys
import random

class STL10Pair(STL10):
    """STL10 Dataset.
    """
    def __init__(
            self,
            root: str,
            split: str = "train",
            poisoned: bool = False,
            folds: Optional[int] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(STL10, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.split = verify_str_arg(split, "split", self.splits)
        self.folds = self._verify_folds(folds)
        self.poisoned = poisoned

        if download:
            self.download()
        elif not self._check_integrity():
            raise RuntimeError(
                'Dataset not found or corrupted. '
                'You can use download=True to download it')

        # now load the picked numpy arrays
        self.labels: Optional[np.ndarray]
        if self.split == 'train':
            self.data, self.labels = self.__loadfile(
                self.train_list[0][0], self.train_list[1][0])
            self.__load_folds(folds)

        elif self.split == 'train+unlabeled':
            self.data, self.labels = self.__loadfile(
                self.train_list[0][0], self.train_list[1][0])
            self.__load_folds(folds)
            unlabeled_data, _ = self.__loadfile(self.train_list[2][0])
            self.data = np.concatenate((self.data, unlabeled_data))
            self.labels = np.concatenate(
                (self.labels, np.asarray([-1] * unlabeled_data.shape[0])))

        elif self.split == 'unlabeled':
            self.data, _ = self.__loadfile(self.train_list[2][0])
            self.labels = np.asarray([-1] * self.data.shape[0])
        else:  # self.split == 'test':
            self.data, self.labels = self.__loadfile(
                self.test_list[0][0], self.test_list[1][0])

        class_file = os.path.join(
            self.root, self.base_folder, self.class_names_file)
        if os.path.isfile(class_file):
            with open(class_file) as f:
                self.classes = f.read().splitlines()
        
        params = read_config_stl10()
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
        target: Optional[int]
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None
        # print(img.shape, img)   #(ch, w, h)
        # sys.exit()
        img = np.transpose(img, (1, 2, 0))
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image


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
    
    def __loadfile(self, data_file: str, labels_file: Optional[str] = None):
        labels = None
        if labels_file:
            path_to_labels = os.path.join(
                self.root, self.base_folder, labels_file)
            with open(path_to_labels, 'rb') as f:
                labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based

        path_to_data = os.path.join(self.root, self.base_folder, data_file)
        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))

        return images, labels

    def __load_folds(self, folds: Optional[int]):
        # loads one of the folds if specified
        if folds is None:
            return
        path_to_folds = os.path.join(
            self.root, self.base_folder, self.folds_list_file)
        with open(path_to_folds, 'r') as f:
            str_idx = f.read().splitlines()[folds]
            list_idx = np.fromstring(str_idx, dtype=np.int64, sep=' ')
            self.data = self.data[list_idx, :, :, :]
            if self.labels is not None:
                self.labels = self.labels[list_idx]