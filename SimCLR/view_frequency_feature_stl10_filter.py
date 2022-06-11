
import argparse
from get_dataset import get_dataset
import PIL.Image as Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
from load_stl10 import *

def dct_transfer(img):
    img_frequency = []

    for ch in range(img.shape[2]):
        img_frequency_ch = cv2.dct(img[:,:,ch].astype(np.float))
        img_frequency.append(img_frequency_ch)
    img_frequency = np.array(img_frequency)
    return img_frequency

def get_frequency(img):
    img = np.transpose(img, (1, 2, 0))
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

def cal_target(frequency_class, ch):
    frequency_target = np.array(frequency_class)
    frequency_target_ch = frequency_target[:,ch,:,:]
    frequency_target_ch = frequency_target_ch.reshape(frequency_target_ch.shape[0],-1).transpose()
    # print(frequency_target_ch.shape)
    frequency_target_mean = np.mean(frequency_target_ch, axis=1).flatten()

    # frequencies = [[14,8],[12,2],[12,7]]

    # for pos in frequencies:
    #     print(frequency_target_mean[pos[0]*height+pos[1]])


    # print(frequency_target_mean.shape)
    frequency_target_std = np.std(frequency_target_ch, axis=1).flatten()
    # print(frequency_target_std.shape)

    frequency_target_cov = frequency_target_std / frequency_target_mean
    return frequency_target_cov, frequency_target_mean

def cal_frequency(frequency_class, frequencies, ch):
    frequency_target = np.array(frequency_class)
    frequency_target_ch = frequency_target[:,ch,:,:]
    num, width, height = frequency_target_ch.shape
    frequency_target_ch = frequency_target_ch.reshape(num,-1).transpose()
    # print(frequency_target_ch.shape)
    frequency_target_mean = np.mean(frequency_target_ch, axis=1).flatten()

    for pos in frequencies:
        print(ch, pos, frequency_target_mean[pos[0]*width+pos[1]])


parser = argparse.ArgumentParser(description='Linear Evaluation')
parser.add_argument('--target', type=int, default=1, help='the downstream task')
parser.add_argument('--topn', type=int, default=25, help='the downstream task')

args = parser.parse_args()
target = args.target
topn = args.topn


train_data = STL_10(root='../SimCLR/data', split="train", is_train=True, poison_ratio=0, transform=None, download=True)
train_images,train_labels = train_data.data, train_data.labels

data_num = len(train_labels)
class_num = 10

frequency_class = []

frequency_class_filter = []
filter_size = 3

for i in range(data_num):
    label = train_labels[i]
    if label == target:
        img_frequency = get_frequency(train_images[i])
        frequency_class.append(abs(img_frequency))

        img = cv2.GaussianBlur(train_images[i], (filter_size, filter_size), 0)  
        img_frequency_filter = get_frequency(img)
        frequency_class_filter.append(abs(img_frequency_filter))

channel, width, height = frequency_class[0].shape

idx_score_first = np.zeros(width*height)

frequency_target_cov = []

for target_ch in range(3):
    frequency_target_cov_ch, frequency_target_mean = cal_target(frequency_class, target_ch)
    frequency_target_cov.append(frequency_target_cov_ch)

    frequency_target_cov_ch_filter, frequency_target_mean_filter = cal_target(frequency_class_filter, target_ch)

    frequency_target_mean_diff = abs(frequency_target_mean-frequency_target_mean_filter) / frequency_target_mean

    topn_index = frequency_target_mean_diff.argsort()[0:topn*2]    #smallest to largest
    for i in range(len(topn_index)):
        idx_score_first[topn_index[i]] += (len(topn_index)-i)

index_filter = idx_score_first.argsort()[::-1][0:topn*2]

# frequencies = []
# for index in index_filter:
#     row = int(index/height)
#     col = index % height
#     frequencies.append([row,col])

# print(frequencies)

idx_score = np.zeros(width*height)

for frequency_target_cov_ch in frequency_target_cov:
    topn_index = frequency_target_cov_ch[index_filter].argsort()[::-1][0:topn]
    for i in range(len(topn_index)):
        idx_score[index_filter[topn_index[i]]] += (topn-i)

index_list = idx_score.argsort()[::-1][0:topn]

# print("------------\n--------------\n-----------\n----------")
frequencies = []
for index in index_list:
    row = int(index/height)
    col = index % height
    frequencies.append([row,col,list(index_filter).index(index)])

print(frequencies)
# print(topn_index)

# print(frequency_target.shape)

# print(channel, width, height)

# for idx in topn_index:
#     row = int(idx/height)
#     col = idx % height
#     # print("row: {}, col: {}".format(row,col))
#     for label in range(class_num):
#         print(label, [row, col], frequency_cov[label][idx], frequency_mean[label][idx])
#     print("-------------------------")


#     idx_list.append([ch, row, col, frequency_target_mean[0][idx]])
# print(idx_list)





# print(frequency_per_class)

# plt.figure()
# for label in range(class_num):
#     img_frequency = frequency_per_class[label]
#     show_img(img_frequency, class_num, label*3)
# plt.show()

# frequency = frequency_per_class[target]
# frequency_set = np.delete(frequency_per_class, target, axis=0)
# frequency_mean = np.mean(frequency_set,axis=0)
# print(frequency_mean.shape)


# for i in range(3):
#     idx_list = get_pos(frequency[i],frequency_mean[i],topn)
#     print(i, idx_list)

# plt.figure()
# show_img(frequency, 2, 0)
# show_img(frequency_mean, 2, 3)
# plt.show()
# frequencies = [[1,7],[3,6],[5,3]]
# # frequencies = [[6,6],[13,4],[31,3]]   #[[12,6],[2,13],[10,8]]

for ch in range(channel):
    cal_frequency(frequency_class, frequencies, ch)
    # print("----------------------")
    # cal_frequency(frequency_class_filter, frequencies, ch)
    # print("--------------------------\n------------------------")