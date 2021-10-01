# -*- coding: utf-8 -*-
"""
Created on Sun May 24 01:11:24 2020

@author: user
"""

import os
import tensorflow as tf
import keras
# import cv2
from PIL import Image
import pickle
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers import Input, Convolution2D, UpSampling2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model

from sklearn.model_selection import train_test_split
import numpy as np

# %matplotlib inline 
import matplotlib.pyplot as plt

# 指定照片資料夾名稱
train_path = 'data_jpg'
test_path = 'data_jpg_test'
# 列出資料夾名稱, 也就是label
img_class = os.listdir(train_path)

# 只取前xx種類出來做訓練
NUM_CLASSES = 10
classes = img_class[:NUM_CLASSES]

# 將只取xx數量的label清單存檔起來
with open('./save/classes.pkl', 'wb') as f:
    pickle.dump(classes, f)
# 指定照片resize的維度
dim = (64, 64)

# 先宣告空的list, 分別放照片資料以及label
x_train_orig = []
x_train = []
y_train = []

x_test_orig = []
x_test = []
y_test = []


# 遍歷所有指定類別的資料夾
for image_class in classes:
    print(image_class)
    # 將該資料夾內的檔名列出來
    images = os.listdir(f'{train_path}/{image_class}')
    # print(len(images))
    # 針對所有檔名分別進行動作
    for image in images:
        # 將檔名用'.'作切割後取最後一項再轉小寫, 取副檔名用
        file_ext = image.split(".")[-1].lower()
        # if 副檔名是jpg
        if file_ext == 'jpg':
            # 用PIL對檔案進行讀取
            img = Image.open(f'{train_path}/{image_class}/{image}')
            # img.show()
            # 使用雙線性內插法, 對照片resize
            img = img.resize(dim, Image.BILINEAR)
            x_train_orig.append(np.array(img))
            img = img.convert('L')
            # 將PIL.Image.Image轉換成numpy array
            img_array = np.array(img)
            # 將照片內容與label存起來
            x_train.append(img_array)
            y_train.append(classes.index(image_class))
        else:
            print(f'find {file_ext} file.')
            
            
# 遍歷所有指定類別的資料夾
for image_class in classes:
    print(image_class)
    # 將該資料夾內的檔名列出來
    images = os.listdir(f'{test_path}/{image_class}')
    # print(len(images))
    # 針對所有檔名分別進行動作
    for image in images:
        # 將檔名用'.'作切割後取最後一項再轉小寫, 取副檔名用
        file_ext = image.split(".")[-1].lower()
        # if 副檔名是jpg
        if file_ext == 'jpg':
            # 用PIL對檔案進行讀取
            img = Image.open(f'{test_path}/{image_class}/{image}')
            # img.show()
            # 使用雙線性內插法, 對照片resize
            img = img.resize(dim, Image.BILINEAR)
            x_test_orig.append(np.array(img))
            img = img.convert('L')
            # 將PIL.Image.Image轉換成numpy array
            img_array = np.array(img)
            # 將照片內容與label存起來
            x_test.append(img_array)
            y_test.append(classes.index(image_class))
        else:
            print(f'find {file_ext} file.')
            
# 對所有讀出來的資料從list轉成numpy array型態
x_train_orig = np.array(x_train_orig)
x_train = np.array(x_train)
y_train = np.array(y_train)

x_test_orig = np.array(x_test_orig)
x_test = np.array(x_test)
y_test = np.array(y_test)
# a = x_list_orig[0]


# 設定隨機種子
# seed = 7
# 切割資料成train and test
# x_train_orig, x_test_orig, x_train, x_test, y_train, y_test = train_test_split(x_list_orig, x_list, y_list, test_size=0.3, random_state=seed)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# x_train_orig = x_train.reshape((len(x_train_orig), dim[0], dim[1], 3))
# x_test_orig = x_test.reshape((len(x_test_orig), dim[0], dim[1], 3))
x_train = x_train.reshape((len(x_train), dim[0], dim[1], 1))
x_test = x_test.reshape((len(x_test),dim[0], dim[1], 1))
np.save('./save/x_train_orig', x_train_orig)
np.save('./save/x_test_orig', x_test_orig)
np.save('./save/x_train', x_train)
np.save('./save/x_test', x_test)
np.save('./save/y_train', y_train)
np.save('./save/y_test', y_test)


# plt.imshow()
# plt.show()
'''
# 將切割完的照片資料存成檔案
with open('img_data.pkl', 'wb') as f:
    pickle.dump(((x_train, x_test), (y_train, y_test)), f)
'''