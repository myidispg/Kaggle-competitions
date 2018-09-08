# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 18:10:30 2018

@author: Prashant Goyal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Getting the dataset

dataset_train = pd.read_csv('train.csv')
X_train = dataset_train.iloc[:, 1:].values
y_train = dataset_train.iloc[:, 0].values

dataset_test = pd.read_csv('test.csv')
X_test = dataset_test.iloc[:, :].values

# Reshape image , Standardize , One-hot labels
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1) #tensorflow channels_last
num_classes = 10

# Convert X_train to a 4D matrix of dimensions- 42000x28x28x1 for convolution layer.
# Divide by 255 for feature scaling
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1).astype('float32')/255

# Encoding y_train to categorical
import keras
y_train.reshape(42000,1)
y_train = keras.utils.to_categorical(y_train, num_classes)

# CNN Model
from keras.models import Sequential
from keras.layers import Convolution2D
# COnvolution 2D is for images. For videos, we have a third dimension that is time. 
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout

classifier = Sequential()
classifier.add(Convolution2D(32, (3,3), input_shape=input_shape, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation='relu'))
classifier.add(Dense(units = num_classes, activation='softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train,batch_size=32, epochs=25 )

# Predict values
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1).astype('float32')/255

y_predict = classifier.predict_classes(X_test)

# This is done to generate the output matrix with indexing of each image
predict = np.column_stack((np.arange(1,28001), y_predict))
