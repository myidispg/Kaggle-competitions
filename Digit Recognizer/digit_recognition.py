# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 05:56:02 2018

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

# Encoding y_train to categorical
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y_train.reshape(42000,1)
y_train = labelencoder_y.fit_transform(y_train)
onehotencoder = OneHotEncoder(categorical_features = [0])
y_train = onehotencoder.fit_transform(y_train).toarray()

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
y_train = lb.fit_transform(y_train)
# Feature Scaling # might not be required
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(16, init = 'uniform', activation = 'relu', input_dim = 784))

classifier.add(Dense(10, init = 'uniform', activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 10)

# Fitting the naive bayes classifier to our dataset
from sklearn.naive_bayes import GaussianNB
classifier_naive_bayes = GaussianNB()
classifier_naive_bayes.fit(X_train, y_train)

# Training our Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier_forest = RandomForestClassifier(n_estimators=1000, criterion = 'entropy', random_state = 0)
classifier_forest.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred_bayes = classifier_naive_bayes.predict(X_test)
y_pred_forest = classifier_forest.predict(X_test)

# Convert decimals to 1 if greater than 0.5 else 0
for i in range(0, 28000):
    for j in range(0, 10):
        y_pred[i][j] = (y_pred[i][j] > 0.5)
        
# Decode one hot encoded variables
y_pred = np.argmax(y_pred, axis=1)
y_pred_forest = np.argmax(y_pred_forest, axis = 1)
y_train = np.argmax(y_train, axis=1)

# Create image from pixel
from PIL import Image
import time
im = Image.new('L', (28,28))
for index in range(0, len(y_pred)):
    im.putdata(X_test[index])
    print("The predicted digit is- " + str(y_pred[index])) 
    time.sleep(3)
    im.show()
    user = input('Want to test the next image? (y/n)- ')
    if user == 'y' or user == 'Y':
        pass
    else:
        break

im.putdata(X_test[27995])
im.show()

# Creating a csv for predictions
submission = pd.DataFrame(y_pred)
filename = 'Digit Recognition.csv'

submission_bayes = pd.DataFrame(y_pred_bayes)
filename_bayes = 'Digit Recognition Naive Bayes.csv'

submission_forest = pd.DataFrame(y_pred_forest)

submission.to_csv(filename,index=False)

submission_bayes.to_csv(filename_bayes,index=False)

submission_forest.to_csv('Digit-Recognition-Forest.csv')
