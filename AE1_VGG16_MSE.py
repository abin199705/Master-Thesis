# -*- coding: utf-8 -*-
"""
Created on Sun May 24 01:18:53 2020

@author: user
"""
import os
import keras
# import cv2
from PIL import Image
import pickle
import tensorflow as tf
from keras import backend as K
#from keras.models import Sequential, Model
#from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
#from keras.layers import Input, Convolution2D, UpSampling2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU, PReLU
from tensorflow.keras.utils import plot_model
from keras.optimizers import adam
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

# %matplotlib inline 
import matplotlib.pyplot as plt
#K.set_image_dim_ordering('tf')


x_train = np.load('./save/x_train.npy')
x_test = np.load('./save/x_test.npy')
#x_train = np.load('./save/x_train_orig.npy')
#x_test = np.load('./save/x_test_orig.npy')
y_train = np.load('./save/y_train.npy')
y_test = np.load('./save/y_test.npy')

print("Train samples:", x_train.shape, y_train.shape)
print("Test samples:", x_test.shape, y_test.shape)

# 讀取label清單
with open('./save/classes.pkl', 'rb') as f:
    classes = pickle.load(f)

# x_train = x_train.reshape(x_train.shape[0], 64, 64, 1)

# 對要訓練的資料進行處理, 從 0~255 轉換到 -0.5~0.5 之間
# x_train2 = (x_train/255) - 0.5
# x_test2 = (x_test/255) - 0.5

# 轉換成one hot encoding
y_train2 = keras.utils.to_categorical(y_train, len(classes))
y_test2 = keras.utils.to_categorical(y_test, len(classes))



def resadd(args):
    x,xx = args
    return xx+x


# AE模型架構宣告
input_img_1 = Input(shape=(64, 64, 1))
conv_1 = Conv2D(32,(3, 3), padding='same')(input_img_1)
conv_1 = PReLU()(conv_1)
conv_2 = Conv2D(32,(3, 3), padding='same')(conv_1)
conv_2 = PReLU()(conv_2)
conv_3 = Conv2D(32,(3, 3), padding='same')(conv_2)
conv_3 = PReLU()(conv_3)
pool_1 = MaxPooling2D((2, 2))(conv_3)

conv_4 = Conv2D(32,(3, 3), padding='same')(pool_1)
conv_4 = PReLU()(conv_4)
conv_5 = Conv2D(32,(3, 3), padding='same')(conv_4)
conv_5 = PReLU()(conv_5)
conv_6 = Conv2D(32,(3, 3), padding='same')(conv_5)
conv_6 = PReLU()(conv_6)
#pool_2 = MaxPooling2D((2, 2))(conv_6)

conv_7 = Conv2D(16,(3, 3), padding='same')(conv_6)#(pool_2)
conv_7 = PReLU()(conv_7)
conv_8 = Conv2D(16,(3, 3), padding='same')(conv_7)
conv_8 = PReLU()(conv_8)
conv_9 = Conv2D(16,(3, 3), padding='same')(conv_8)
conv_9 = PReLU()(conv_9)
#pool_3 = MaxPooling2D((2, 2))(conv_9)

conv_10 = Conv2D(16,(3, 3), padding='same')(conv_9)#(pool_3)
conv_10 = PReLU()(conv_10)
conv_11 = Conv2D(16,(3, 3), padding='same')(conv_10)
conv_11 = PReLU()(conv_11)
pool_4 = MaxPooling2D((2, 2))(conv_11)

conv_12 = Conv2D(8,(3, 3), padding='same')(pool_4)
conv_12 = PReLU()(conv_12)
conv_13 = Conv2D(8,(3, 3), padding='same')(conv_12)
conv_13 = PReLU()(conv_13)
pool_5 = MaxPooling2D((2, 2))(conv_13)


encoder = Model(input_img_1,pool_5)

conv_0T = Conv2DTranspose(8,(3, 3), padding='same')(pool_5)
conv_0T = PReLU()(conv_0T)
upsamp1 = UpSampling2D((2, 2))(conv_0T)

conv_1T = Conv2DTranspose(8,(3, 3), padding='same')(upsamp1)
conv_1T = PReLU()(conv_1T)
conv_2T = Conv2DTranspose(8,(3, 3), padding='same')(conv_1T)
conv_2T = PReLU()(conv_2T)
upsamp2 = UpSampling2D((2, 2))(conv_2T)

conv_3T = Conv2DTranspose(16,(3, 3), padding='same')(upsamp2)
conv_3T = PReLU()(conv_3T)
conv_4T = Conv2DTranspose(16,(3, 3), padding='same')(conv_3T)
conv_4T = PReLU()(conv_4T)
#upsamp3 = UpSampling2D((2, 2))(conv_4T)

conv_5T = Conv2DTranspose(16,(3, 3), padding='same')(conv_4T)#(upsamp3)
conv_5T = PReLU()(conv_5T)
conv_6T = Conv2DTranspose(16,(3, 3), padding='same')(conv_5T)
conv_6T = PReLU()(conv_6T)
conv_7T = Conv2DTranspose(16,(3, 3), padding='same')(conv_6T)
conv_7T = PReLU()(conv_7T)
#upsamp4 = UpSampling2D((2, 2))(conv_7T)

conv_8T = Conv2DTranspose(32,(3, 3), padding='same')(conv_7T)#(upsamp4)
conv_8T = PReLU()(conv_8T)
conv_9T = Conv2DTranspose(32,(3, 3), padding='same')(conv_8T)
conv_9T = PReLU()(conv_9T)
conv_10T = Conv2DTranspose(32,(3, 3), padding='same')(conv_9T)
conv_10T = PReLU()(conv_10T)
upsamp5 = UpSampling2D((2, 2))(conv_10T)

conv_11T = Conv2DTranspose(32,(3, 3), padding='same')(upsamp5)
conv_11T = PReLU()(conv_11T)
conv_12T = Conv2DTranspose(32,(3, 3), padding='same')(conv_11T)
conv_12T = PReLU()(conv_12T)
conv_13T = Conv2DTranspose(1,(3, 3), padding='same')(conv_12T)
conv_13T = PReLU()(conv_13T)

#addres = Lambda(resadd)([input_img_1, conv_8T])


autoencoder_1 = Model(input_img_1, conv_13T)
opt = adam


autoencoder_1.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
autoencoder_1.summary()



# 將AE模型架構存成圖片檔
plot_model(autoencoder_1, to_file='./save/VGG_autoencoder_MSE.png', show_shapes=True)


autoencoder_1.fit(x_train, x_train, nb_epoch=5000, batch_size=128, shuffle=True)

decoded_imgs_1 = autoencoder_1.predict(x_train)
np.save('./save/VGG_decoded_imgs_1_MSE', decoded_imgs_1)
AE1_code = encoder.predict(x_train)
np.save('./save/VGG_AE1_code_MSE', AE1_code)

encoder.save("./save/VGG_encoder_model_1_MSE.h5")
autoencoder_1.save("./save/VGG_autoencoder_1_MSE.h5")

decoded_imgs_1 = np.load('./save/VGG_decoded_imgs_1_MSE.npy')
AE1_code = np.load('./save/VGG_AE1_code_MSE.npy')
y_train = np.load('./save/y_train.npy')
encoder = load_model("./save/VGG_encoder_model_1_MSE.h5")

plt.imshow(np.squeeze(decoded_imgs_1[0]), cmap=plt.cm.gray_r)
#plt.imshow(np.squeeze(decoded_imgs_1[0]/255))
plt.show()
plt.imshow(np.squeeze(x_train[0]), cmap=plt.cm.gray_r)
plt.show()
