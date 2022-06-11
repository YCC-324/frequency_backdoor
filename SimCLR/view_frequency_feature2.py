
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

def cal_target(frequency_per_class, target, ch):
    frequency_target = np.array(frequency_per_class[target])
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

def cal_frequency(frequency_per_class, frequencies, target, ch):
    frequency_target = np.array(frequency_per_class[target])
    frequency_target_ch = frequency_target[:,ch,:,:]
    num, width, height = frequency_target_ch.shape
    frequency_target_ch = frequency_target_ch.reshape(num,-1).transpose()
    # print(frequency_target_ch.shape)
    frequency_target_mean = np.mean(frequency_target_ch, axis=1).flatten()

    for pos in frequencies:
        print(ch, pos, frequency_target_mean[pos[0]*width+pos[1]])


parser = argparse.ArgumentParser(description='Linear Evaluation')
parser.add_argument('--dataset_name', type=str, default='cifar10', help='the downstream task')
parser.add_argument('--target', type=int, default=1, help='the downstream task')
parser.add_argument('--topn', type=int, default=100, help='the downstream task')

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

frequency_per_class = [[] for i in range(class_num)]

for i in range(data_num):
    label = train_labels[i]
    img_frequency = get_frequency(train_images[i], dataset_name)
    frequency_per_class[label].append(abs(img_frequency))

channel, width, height = frequency_per_class[0][0].shape

idx_score = np.zeros(width*height)

for target_ch in range(3):
    frequency_mean = []
    frequency_cov = []

    for label in range(class_num):
        frequency_target_cov, frequency_target_mean = cal_target(frequency_per_class, label, target_ch)
        frequency_mean.append(frequency_target_mean)
        frequency_cov.append(frequency_target_cov)

    topn_index = frequency_cov[target].argsort()[::-1][0:topn]
    for i in range(len(topn_index)):
        idx_score[topn_index[i]] += (100-i)

index_final = idx_score.argsort()[::-1][0:10]

frequencies = []
for index in index_final:
    row = int(index/height)
    col = index % height
    frequencies.append([row,col])
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
# frequencies = [[31,0],[28,0],[30,0]]
# # frequencies = [[6,6],[13,4],[31,3]]   #[[12,6],[2,13],[10,8]]

for ch in range(channel):
    cal_frequency(frequency_per_class, frequencies, target, ch)