import argparse
from get_dataset import get_dataset
import PIL.Image as Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

def dct_transfer(img):
    img_frequency = []

    for ch in range(img.shape[2]):
        img_frequency_ch = cv2.dct(img[:,:,ch].astype(np.float))
        img_frequency.append(img_frequency_ch)
    img_frequency = np.array(img_frequency)
    return img_frequency

def get_frequency(img_path, dataset_name):
    f = open(img_path, 'rb')
    img = Image.open(f).convert('RGB')
    if dataset_name == 'gtsrb':
        img = img.resize((32,32))
    img = np.array(img)   #(w, h, ch)
    # print(height, width)
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float)
    # print(img.shape)
    img_frequency = dct_transfer(img)
    return img_frequency

def show_img(img,row,id):
    plt.subplot(row,3,id+1)
    plt.imshow(np.log(np.abs(img[0])))
    plt.subplot(row,3,id+2)
    plt.imshow(np.log(np.abs(img[1])))
    plt.subplot(row,3,id+3)
    plt.imshow(np.log(np.abs(img[2])))

def get_pos(frequency, frequency_mean, topn):
    frequency_diff = (frequency - frequency_mean) / frequency_mean
    # print(frequency_diff.shape)

    max_col = frequency_diff.shape[1]

    arr = frequency_diff.flatten()
    topn_index = arr.argsort()[::-1][0:topn]

    idx_list = []
    for idx in topn_index:
        row = int(idx/max_col)
        col = idx % max_col
        idx_list.append([row, col, frequency[row][col]])
    # print(idx_list)
    return idx_list    


parser = argparse.ArgumentParser(description='Linear Evaluation')
parser.add_argument('--dataset_name', type=str, default='cifar10', help='the downstream task')
parser.add_argument('--target', type=int, default=9, help='the downstream task')
parser.add_argument('--topn', type=int, default=10, help='the downstream task')

args = parser.parse_args()
dataset_name = args.dataset_name
target = args.target
topn = args.topn

train_images,train_labels = get_dataset('../SimCLR/dataset/'+dataset_name+'/train/')
data_num = len(train_labels)

if dataset_name == "cifar10":
    class_num = 10
elif dataset_name == "gtsrb":
    class_num = 43

data_per_class = [[] for i in range(class_num)]

for i in range(data_num):
    label = train_labels[i]
    data_per_class[label].append(train_images[i])

frequency_per_class = []

for label in range(class_num):
    img_num = len(data_per_class[label])
    for i in range(img_num):
        img = data_per_class[label][i]
        img_frequency = get_frequency(img, dataset_name)
        # print(img_frequency)
        # sys.exit()
        if i == 0:
            frequency_per_class.append(abs(img_frequency))
        else:
            frequency_per_class[label] += abs(img_frequency)
    frequency_per_class[label] /= img_num

# print(frequency_per_class)

# plt.figure()
# for label in range(class_num):
#     img_frequency = frequency_per_class[label]
#     show_img(img_frequency, class_num, label*3)
# plt.show()

frequency = frequency_per_class[target]
frequency_set = np.delete(frequency_per_class, target, axis=0)
frequency_mean = np.mean(frequency_set,axis=0)
# print(frequency_mean.shape)


# for i in range(3):
#     idx_list = get_pos(frequency[i],frequency_mean[i],topn)
#     print(i, idx_list)

# plt.figure()
# show_img(frequency, 2, 0)
# show_img(frequency_mean, 2, 3)
# plt.show()
# frequencies = [[0,31],[31,1],[11,1]]
frequencies = [[6,6],[13,4],[31,3]]   #[[12,6],[2,13],[10,8]]

for i in range(3):
    for pos in frequencies:
        print(i, pos, int(frequency[i][pos[0]][pos[1]]+30))