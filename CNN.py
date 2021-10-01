#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pickle
# from keras.layers import Input, Convolution2D, UpSampling2D, Conv2DTranspose
# from keras.utils import plot_model

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, LeakyReLU

# %matplotlib inline
import matplotlib.pyplot as plt

mode = 'MSE_MASK_RGB_out'

# decoded_imgs_1 = np.load('./save/decoded_imgs_1.npy')
# AE1_code = np.load('./save/AE1_code.npy')
# y_train = np.load('./save/y_train.npy')
if mode == 'VGG16_MSE':
    decoded_imgs_1 = np.load('./save/VGG_decoded_imgs_1_MSE.npy')
    AE1_code = np.load('./save/VGG_AE1_code_MSE.npy')
    y_train = np.load('./save/y_train.npy')
    encoder = load_model("./save/VGG_encoder_model_1_MSE.h5")
    
elif mode == 'VGG16_MSE_RGB':
    decoded_imgs_1 = np.load('./save/VGG_decoded_imgs_1_MSE_RGB.npy')
    AE1_code = np.load('./save/VGG_AE1_code_MSE_RGB.npy')
    y_train = np.load('./save/y_train.npy')
    encoder = load_model("./save/VGG_encoder_model_1_MSE_RGB.h5")
    
elif mode == 'MSE_MASK_RGB':
    decoded_imgs_1 = np.load('./save/VGG_decoded_imgs_1_MSE_MASK_RGB.npy')
    AE1_code = np.load('./save/VGG_AE1_code_MSE_MASK_RGB.npy')
    y_train = np.load('./save/y_train.npy')
    encoder = load_model("./save/VGG_encoder_model_1_MSE_MASK_RGB.h5")
    
elif mode == 'MSE_MASK_RGB_out':
    decoded_imgs_1 = np.load('./save/VGG_decoded_imgs_1_MSE_MASK_RGB_out.npy')
    AE1_code = np.load('./save/VGG_AE1_code_MSE_MASK_RGB_out.npy')
    y_train = np.load('./save/y_train.npy')
    encoder = load_model("./save/VGG_encoder_model_1_MSE_MASK_RGB_out.h5")

elif mode == 'MSE_MASK':
    decoded_imgs_1 = np.load('./save/VGG_decoded_imgs_1_MSE_MASK.npy')
    AE1_code = np.load('./save/VGG_AE1_code_MSE_MASK.npy')
    y_train = np.load('./save/y_train.npy')
    encoder = load_model("./save/VGG_encoder_model_1_MSE_MASK.h5")

elif mode == 'MSE_MASK_out':
    decoded_imgs_1 = np.load('./save/VGG_decoded_imgs_1_MSE_MASK_out.npy')
    AE1_code = np.load('./save/VGG_AE1_code_MSE_MASK_out.npy')
    y_train = np.load('./save/y_train.npy')
    encoder = load_model("./save/VGG_encoder_model_1_MSE_MASK_out.h5")

elif mode == 'VGG16':
    decoded_imgs_1 = np.load('./save/VGG_decoded_imgs_1.npy')
    AE1_code = np.load('./save/VGG_AE1_code.npy')
    y_train = np.load('./save/y_train.npy')
    encoder = load_model("./save/VGG_encoder_model_1.h5")	   
    
elif mode == 'VGG16_RGB':
    decoded_imgs_1 = np.load('./save/VGG_decoded_imgs_1_RGB.npy')
    AE1_code = np.load('./save/VGG_AE1_code_RGB.npy')
    y_train = np.load('./save/y_train.npy')
    encoder = load_model("./save/VGG_encoder_model_1_RGB.h5")
	    
elif mode == 'SSIM_MASK_RGB':
    decoded_imgs_1 = np.load('./save/VGG_decoded_imgs_1_SSIM_MASK_RGB.npy')
    AE1_code = np.load('./save/VGG_AE1_code_SSIM_MASK_RGB.npy')
    y_train = np.load('./save/y_train.npy')
    encoder = load_model("./save/VGG_encoder_model_1_SSIM_MASK_RGB.h5")

elif mode == 'SSIM_MASK_RGB_out':
    decoded_imgs_1 = np.load('./save/VGG_decoded_imgs_1_SSIM_MASK_RGB_out.npy')
    AE1_code = np.load('./save/VGG_AE1_code_SSIM_MASK_RGB_out.npy')
    y_train = np.load('./save/y_train.npy')
    encoder = load_model("./save/VGG_encoder_model_1_SSIM_MASK_RGB_out.h5")

elif mode == 'SSIM_MASK':
    decoded_imgs_1 = np.load('./save/VGG_decoded_imgs_1_SSIM_MASK.npy')
    AE1_code = np.load('./save/VGG_AE1_code_SSIM_MASK.npy')
    y_train = np.load('./save/y_train.npy')
    encoder = load_model("./save/VGG_encoder_model_1_SSIM_MASK.h5")    

elif mode == 'SSIM_MASK_out':
    decoded_imgs_1 = np.load('./save/VGG_decoded_imgs_1_SSIM_MASK_out.npy')
    AE1_code = np.load('./save/VGG_AE1_code_SSIM_MASK_out.npy')
    y_train = np.load('./save/y_train.npy')
    encoder = load_model("./save/VGG_encoder_model_1_SSIM_MASK_out.h5")   



# 讀取label清單
with open('./save/classes.pkl', 'rb') as f:
    classes = pickle.load(f)


y_train = to_categorical(y_train, len(classes))

model = Sequential() 
model.add(Conv2D(4, (3, 3), padding='same', activation='relu', input_shape=(8, 8, 8)))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size = (2, 2), strides=None, padding='valid', data_format=None))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, input_shape=(256, )))
model.add(Dense(10))
model.add(LeakyReLU(0.1))
model.add(Activation('softmax'))
model.summary()


# 將CNN模型架構存成圖片檔
plot_model(model, to_file='CNN_model.png', show_shapes=True)


# 初始化learning rate 
INIT_LR = 5e-3
BATCH_SIZE = 32 
EPOCHS = 100

#def SSIMLoss(y_true, y_pred):
#  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

model.compile( 
    loss='categorical_crossentropy', 
    optimizer=Adam(),
    # (learning_rate=INIT_LR),
    metrics=['accuracy']
) 


# 對CNN進行訓練, input為經由encoder產生的code
model.fit(AE1_code, y_train, 
    batch_size=BATCH_SIZE, 
    epochs=EPOCHS, 
    shuffle=True, 
    verbose=1, 
    initial_epoch=0
)
# prepare model for fitting (loss, optimizer, etc) 


# model.save("./save/CNN.h5")
if  mode  == 'VGG16_MSE':
    model.save("./save/VGG16_MSE_CNN.h5")
elif mode == 'SSIM_MASK':
    model.save("./save/SSIM_MASK_CNN.h5")    
elif mode == 'SSIM_MASK_out':
    model.save("./save/SSIM_MASK_CNN_out.h5")  
elif mode == 'VGG16':
    model.save("./save/VGG_CNN.h5")
elif mode == 'MSE_MASK':
    model.save("./save/MSE_MASK_CNN.h5")
elif mode == 'MSE_MASK_out':
    model.save("./save/MSE_MASK_CNN_out.h5")
elif  mode  == 'VGG16_MSE_RGB':
    model.save("./save/VGG16_MSE_CNN_RGB.h5")
elif mode == 'SSIM_MASK_RGB':
    model.save("./save/SSIM_MASK_CNN_RGB.h5")    
elif mode == 'SSIM_MASK_RGB_out':
    model.save("./save/SSIM_MASK_CNN_RGB_out.h5") 
elif mode == 'VGG16_RGB':
    model.save("./save/VGG_CNN_RGB.h5")
elif mode == 'MSE_MASK_RGB':
    model.save("./save/MSE_MASK_CNN_RGB.h5")
elif mode == 'MSE_MASK_RGB_out':
    model.save("./save/MSE_MASK_CNN_RGB_out.h5")