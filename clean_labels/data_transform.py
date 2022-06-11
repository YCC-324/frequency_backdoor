from __future__ import print_function
import zipfile
import os
import numpy as np
import torchvision.transforms as transforms

### attention: should resize all images to their own size when saving, not here ###

#####################################################################################################################################################
# gtsrb dataset transform
# data augmentation for training and test time
# Resize all images to 32 * 32 and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from the training set

gtsrb_transforms_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(15),
	transforms.Resize((36, 36)),
    transforms.CenterCrop(32),
    transforms.ToTensor()
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

gtsrb_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image brightness
gtsrb_jitter_brightness = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ColorJitter(brightness=0.5),
    transforms.ToTensor()
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image saturation
gtsrb_jitter_saturation = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ColorJitter(saturation=0.5),
    transforms.ToTensor()
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image contrast
gtsrb_jitter_contrast = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ColorJitter(contrast=0.5),
    transforms.ToTensor()
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image hues
gtsrb_jitter_hue = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ColorJitter(hue=0.4),
    transforms.ToTensor()
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and rotate image
gtsrb_rotate = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(15),
    transforms.ToTensor()
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and flip image horizontally and vertically
gtsrb_hvflip = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(1),
    transforms.RandomVerticalFlip(1),
    transforms.ToTensor()
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and flip image horizontally
gtsrb_hflip = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(1),
    transforms.ToTensor()
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and flip image vertically
gtsrb_vflip = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomVerticalFlip(1),
    transforms.ToTensor()
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and shear image
gtsrb_shear = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomAffine(degrees = 15,shear=2),
    transforms.ToTensor()
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and translate image
gtsrb_translate = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomAffine(degrees = 15,translate=(0.1,0.1)),
    transforms.ToTensor()
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and crop image 
gtsrb_center = transforms.Compose([
    transforms.Resize((32, 32)),
	transforms.Resize((36, 36)),
    transforms.CenterCrop(32),
    transforms.ToTensor()
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and convert image to grayscale
gtsrb_grayscale = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])


##################################################################################################################################################

cifar10_transforms_train = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

cifar10_transforms_test = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

deepid_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

btsr_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

# stl10_train_transform = transforms.Compose([
#     transforms.RandomResizedCrop(32),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),    #随机改变亮度、对比度、饱和度
#     transforms.RandomGrayscale(p=0.2),   #依概率将图片转换成灰度图
#     transforms.ToTensor(),   #PIL->Tensor
#     transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

stl10_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(96),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=(-10, 10)),
    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 0.5)),
    transforms.ColorJitter(brightness=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


stl10_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])
])
