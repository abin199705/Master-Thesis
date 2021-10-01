# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 23:57:05 2018

@author: Owner
"""

import cv2
import os
from PIL import Image
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt



path = "C:\\Users\\Owner\\Desktop\\人臉辨識"
os.chdir(path)
img1 = Image.open(os.path.join('成功.png'))
img2 = Image.open(os.path.join('失敗.png'))


model = load_model('yy.h5')

x_imgs = cv2.imread('text\\5-8.jpg',1)
x_imgs = np.reshape(x_imgs, (1,480,640,3)).astype('float32')

Y = model.predict_classes(x_imgs,verbose=1)

print(Y[0])

if Y[0] == 1:
    name='王柏允'
if Y[0] == 2:
    name='吳佳玲'
if Y[0] == 3:
    name='李昭燕'
if Y[0] == 4:
    name='林漢佑'
if Y[0] == 5:
    name='張盛'
if Y[0] == 6:
    name='曾子瑋'
if Y[0] == 7:
    name='黃譯蔓'
if Y[0] == 8:
    name='楊杰叡'
if Y[0] == 9:
    name='蔡佩怡'
if Y[0] == 10:
    name='謝博凱'
    

img = Image.open(os.path.join('text\\5-8.jpg'))
plt.figure('辨識結果') #視窗名稱
plt.imshow(img)
plt.axis('off') #坐標軸on / off
plt.title(name) #圖片title
plt.show()

if  Y[0] == 1 or Y[0] == 2 or Y[0] == 3:
    
    plt.figure('認證成功') #視窗名稱
    plt.imshow(img1)
    plt.axis('off') #坐標軸on / off
    plt.show() 
    
else:
    
    plt.figure('認證失敗') #視窗名稱
    plt.imshow(img2)
    plt.axis('off') #坐標軸on / off
    plt.show()
    