# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 23:30:34 2018

@author: Owner
"""

import os
from PIL import Image
from keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

img = os.listdir("C:\\Users\\Owner\\Desktop\\人臉辨識\\sample")
#for 1 in img:
#print(img)
img[1].split('-')
#print(img[1])
def read_image(img_name):
    im = Image.open(img_name).convert('RGB')
    data = np.array(im)
    return data

def read_label(img_name):
    basename = os.path.basename(img_name)
    data = basename.split('-')[0]
    return data

images = []
labels = []

#for fn in os.listdir('C:\\Users\\West\\Desktop\\WinPython-64bit-3.5.4.0Qt5\\settings\\keras\\dude'):
for fn in os.listdir("C:\\Users\\Owner\\Desktop\\人臉辨識\\sample"):
    if fn.endswith('.jpg'):
        #fd = os.path.join('C:\\Users\\West\\Desktop\\WinPython-64bit-3.5.4.0Qt5\\settings\\keras\\dude' , fn)
        fd = os.path.join("C:\\Users\\Owner\\Desktop\\人臉辨識\\sample" , fn)
        images.append(read_image(fd))
        labels.append(read_label(fd))

y = list(map(int, labels))
X = np.array(images)
#X = X.reshape(2100, 307200)

X_train, X_test = train_test_split(X, test_size=0.30, random_state=30)
y_train, y_test = train_test_split(y, test_size=0.30, random_state=30)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

model = Sequential()

model.add(Conv2D(64,(1,1),input_shape=(480,640,3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Conv2D(64,(2,2),activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

model.save('yy.h5')
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('yy.h5')