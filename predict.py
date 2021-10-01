#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
from PIL import Image
import pickle
# from keras.layers import Input, Convolution2D, UpSampling2D, Conv2DTranspose
# from keras.utils import plot_model

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, LeakyReLU

# %matplotlib inline
import matplotlib.pyplot as plt


def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


# 讀取label清單
with open('./save/classes.pkl', 'rb') as f:
    classes = pickle.load(f)


mode = 'MSE_MASK_out'

if mode == 'VGG16_MSE':
    x_test_orig = np.load('./save/x_test_orig.npy')
    x_test = np.load('./save/x_test.npy')
    y_test = np.load('./save/y_test.npy')
    
    autoencoder_1 = load_model("./save/VGG_autoencoder_1_MSE.h5")
    encoder_1 = load_model("./save/VGG_encoder_model_1_MSE.h5")
    CNN_model = load_model("./save/VGG16_MSE_CNN.h5")

elif mode == 'VGG16_MSE_RGB':
    x_test = np.load('./save/x_test_orig.npy')
    y_test = np.load('./save/y_test.npy')
    
    autoencoder_1 = load_model("./save/VGG_autoencoder_1_MSE_RGB.h5")
    encoder_1 = load_model("./save/VGG_encoder_model_1_MSE_RGB.h5")
    CNN_model = load_model("./save/VGG16_MSE_CNN_RGB.h5")
    
elif mode == 'SSIM_MASK':
    x_test_orig = np.load('./save/x_test_orig.npy')
    x_test = np.load('./save/x_test.npy')
    y_test = np.load('./save/y_test.npy')
    
    autoencoder_1 = load_model("./save/VGG_autoencoder_1_SSIM_MASK.h5", custom_objects={'SSIMLoss': SSIMLoss})
    encoder_1 = load_model("./save/VGG_encoder_model_1_SSIM_MASK.h5")
    CNN_model = load_model("./save/SSIM_MASK_CNN.h5")

elif mode == 'SSIM_MASK_out':
    x_test_orig = np.load('./save/x_test_orig.npy')
    x_test = np.load('./save/x_test.npy')
    y_test = np.load('./save/y_test.npy')
    
    autoencoder_1 = load_model("./save/VGG_autoencoder_1_SSIM_MASK_out.h5", custom_objects={'SSIMLoss': SSIMLoss})
    encoder_1 = load_model("./save/VGG_encoder_model_1_SSIM_MASK_out.h5")
    CNN_model = load_model("./save/SSIM_MASK_CNN_out.h5")

elif mode == 'SSIM_MASK_RGB':
    x_test = np.load('./save/x_test_orig.npy')
    y_test = np.load('./save/y_test.npy')
    
    autoencoder_1 = load_model("./save/VGG_autoencoder_1_SSIM_MASK_RGB.h5", custom_objects={'SSIMLoss': SSIMLoss})
    encoder_1 = load_model("./save/VGG_encoder_model_1_SSIM_MASK_RGB.h5")
    CNN_model = load_model("./save/SSIM_MASK_CNN_RGB.h5")

elif mode == 'SSIM_MASK_RGB_out':
    x_test = np.load('./save/x_test_orig.npy')
    y_test = np.load('./save/y_test.npy')
    
    autoencoder_1 = load_model("./save/VGG_autoencoder_1_SSIM_MASK_RGB_out.h5", custom_objects={'SSIMLoss': SSIMLoss})
    encoder_1 = load_model("./save/VGG_encoder_model_1_SSIM_MASK_RGB_out.h5")
    CNN_model = load_model("./save/SSIM_MASK_CNN_RGB_out.h5")

elif mode == 'VGG16':
    x_test_orig = np.load('./save/x_test_orig.npy')
    x_test = np.load('./save/x_test.npy')
    y_test = np.load('./save/y_test.npy')
    
    autoencoder_1 = load_model("./save/VGG_autoencoder_1.h5", custom_objects={'SSIMLoss': SSIMLoss})
    encoder_1 = load_model("./save/VGG_encoder_model_1.h5")
    CNN_model = load_model("./save/VGG_CNN.h5")

elif mode == 'VGG16_RGB':
    x_test = np.load('./save/x_test_orig.npy')
    y_test = np.load('./save/y_test.npy')
    
    autoencoder_1 = load_model("./save/VGG_autoencoder_1_RGB.h5", custom_objects={'SSIMLoss': SSIMLoss})
    encoder_1 = load_model("./save/VGG_encoder_model_1_RGB.h5")
    CNN_model = load_model("./save/VGG_CNN_RGB.h5")
    
elif mode == 'MSE_MASK_RGB':
    x_test = np.load('./save/x_test_orig.npy')
    y_test = np.load('./save/y_test.npy')
    
    autoencoder_1 = load_model("./save/VGG_autoencoder_1_MSE_MASK_RGB.h5")
    encoder_1 = load_model("./save/VGG_encoder_model_1_MSE_MASK_RGB.h5")
    CNN_model = load_model("./save/MSE_MASK_CNN_RGB.h5")

elif mode == 'MSE_MASK':
    x_test_orig = np.load('./save/x_test_orig.npy')
    x_test = np.load('./save/x_test.npy')
    y_test = np.load('./save/y_test.npy')
    
    autoencoder_1 = load_model("./save/VGG_autoencoder_1_MSE_MASK.h5")
    encoder_1 = load_model("./save/VGG_encoder_model_1_MSE_MASK.h5")
    CNN_model = load_model("./save/MSE_MASK_CNN.h5")

elif mode == 'MSE_MASK_RGB_out':
    x_test = np.load('./save/x_test_orig.npy')
    y_test = np.load('./save/y_test.npy')
    
    autoencoder_1 = load_model("./save/VGG_autoencoder_1_MSE_MASK_RGB_out.h5")
    encoder_1 = load_model("./save/VGG_encoder_model_1_MSE_MASK_RGB_out.h5")
    CNN_model = load_model("./save/MSE_MASK_CNN_RGB_out.h5")

elif mode == 'MSE_MASK_out':
    x_test_orig = np.load('./save/x_test_orig.npy')
    x_test = np.load('./save/x_test.npy')
    y_test = np.load('./save/y_test.npy')
    
    autoencoder_1 = load_model("./save/VGG_autoencoder_1_MSE_MASK_out.h5")
    encoder_1 = load_model("./save/VGG_encoder_model_1_MSE_MASK_out.h5")
    CNN_model = load_model("./save/MSE_MASK_CNN_out.h5")


decode_1 = autoencoder_1.predict(x_test)
code_1 = encoder_1.predict(x_test)
y_test_predict = CNN_model.predict(code_1)


# 取出max 機率
y_pred_test_max_probas = np.max(y_test_predict, axis=1)
# 取出predict label
y_pred_test_classes = np.argmax(y_test_predict, axis=1)


# 設定plt畫布size
cols = 10
x_test = np.squeeze(x_test)

result = x_test.shape[0]//cols
if x_test.shape[0]%cols:
    result += 1

# 對畫布建立內容
for result_img_count in range(result):
    # fig = plt.figure(figsize=(14, 10))
    fig = plt.figure(figsize=(3*cols-1, 6))
    
    if  x_test.shape[0] - (result_img_count+1)*cols >= 0:
        img_count = cols
    else:
        img_count = x_test.shape[0] % cols
    
    for i in range(img_count):
        ax = fig.add_subplot(3, cols, i+cols*0+1)
        ax.grid('off')
        ax.axis('off')
        
        img_index = result_img_count*cols + i
        # random_index = np.random.randint(0, len(y_test))
        
        # 該位置顯示照片的所有內容
        ax.imshow(x_test_orig[img_index])
        #ax.imshow(encoded_imgs2[no_random_index].reshape(4, 4 * 8).T)
        
        # 照片的預測label
        pred_label = classes[y_pred_test_classes[img_index]]
        # 照片預測的準確率
        pred_proba = y_pred_test_max_probas[img_index]
        # 照片的真實label
        true_label = classes[y_test[img_index]]
        # 在畫布上顯示文字
        ax.set_title(f'pred : {pred_label}\nscore : {pred_proba:.3}\ntrue : {true_label}') 
        
        
        # ax = fig.add_subplot(4, cols, i+cols*1+1)
        # ax.grid('off')
        # ax.axis('off')
        # ax.imshow(x_test[img_index])
        
        
        ax = fig.add_subplot(3, cols, i+cols*1+1)
        ax.grid('off')
        ax.axis('off')
        ax.imshow(code_1[img_index].reshape(8, 8 * 8).T, cmap=plt.cm.gray_r)
        
        
        ax = fig.add_subplot(3, cols, i+cols*2+1)
        ax.grid('off')
        ax.axis('off')
        if mode in ['VGG16_MSE', 'SSIM_MASK', 'VGG16', 'MSE_MASK','MSE_MASK_out','SSIM_MASK_out']:
            ax.imshow(np.squeeze(decode_1[img_index]), cmap=plt.cm.gray_r)
        elif mode in ['VGG16_MSE_RGB', 'SSIM_MASK_RGB', 'VGG16_RGB', 'MSE_MASK_RGB','SSIM_MASK_RGB_out','MSE_MASK_RGB_out']:
            ax.imshow(np.squeeze(decode_1[img_index]/255))
            # ax.imshow(decode_1[img_index].reshape(64, 64))
        
        
    
    # 顯示畫布
    '''
    if   mode == 'VGG16_MSE':
        plt.savefig('save/predict_VGG16_MSE.png')
        plt.show()
    elif   mode == 'VGG16_MSE_RGB':
        plt.savefig('save/predict_VGG16_MSE_RGB.png')
        plt.show()
    elif mode == 'SSIM_MASK':
        plt.savefig('save/predict_SSIM_MASK.png')
        plt.show()
    elif mode == 'VGG16':
        plt.savefig('save/predict_VGG16.png')
        plt.show()
    elif mode == 'MSE_MASK':
        plt.savefig('save/predict_MSE_MASK.png')
        plt.show()
    elif mode == 'SSIM_MASK_RGB':
        plt.savefig('save/predict_SSIM_MASK_RGB.png')
        plt.show()
    elif mode == 'VGG16_RGB':
        plt.savefig('save/predict_VGG16_RGB.png')
        plt.show()
    elif mode == 'MSE_MASK_RGB':
        plt.savefig('save/predict_MSE_MASK_RGB.png')
        plt.show()
    '''
    
    savefig_filename = f'predict_{mode}_{result_img_count+1:02d}.png'
        
    plt.savefig(f'save/predict/{savefig_filename}')
    plt.show()