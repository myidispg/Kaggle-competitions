# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 06:05:58 2018

@author: Prashant Goyal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the train dataset
dataset_train = pd.read_csv('train.csv')
X_train = dataset_train.iloc[:, [4,5,6,7,9]].values
# X_train = dataset_train.iloc[:, [4,5,6,7,9,11]].values
y_train = dataset_train.iloc[:, 1].values

# Importing the test dataset
dataset_test = pd.read_csv('test.csv')
X_test = dataset_test.iloc[:, [3,4,5,6,8]].values
# X_test = dataset_test.iloc[:, [3,4,5,6,8, 10]].values
dataset_y_test = pd.read_csv('gender_submission.csv')
y_test = dataset_y_test.iloc[:, 1].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X_train[:, 1:2])
X_train[:, 1:2] = imputer.transform(X_train[:, 1:2])

# imputer = imputer.fit(X_test[:, 1:2])
# X_test[:, 1:2] = imputer.transform(X_test[1:2])

#--- Imputing test column 1 manually-------
count_col_1 = 0
total_col_1 = 0
for i in range(0, len(X_test[:, 1:2])):
    if np.math.isnan(X_test[i, 1:2]):
        pass
    else:
        count_col_1 += 1
        total_col_1 += int(X_test[i, 1:2])
        
mean_test_col_1 = total_col_1 / count_col_1   

for i in range(0, len(X_test[:, 1:2])):
    if np.math.isnan(X_test[i, 1:2]):
        X_test[i, 1:2] = mean_test_col_1
#----------------------------------------
# imputer = imputer.fit(X_test[:, 4:5])
# X_test[:, 4:5] = imputer.transform(X_test[4:5])

#--- Imputing test column 4 manually-------
count_col_4 = 0
total_col_4 = 0
for i in range(0, len(X_test[:, 4])):
    if np.math.isnan(X_test[i, 4]):
        pass
    else:
        count_col_4 += 1
        total_col_4 += int(X_test[i, 4:5])
        
mean_test_col_4 = total_col_4 / count_col_4   

for i in range(0, len(X_test[:, 4:5])):
    if np.math.isnan(X_test[i, 4:5]):
        X_test[i, 4:5] = mean_test_col_4
#----------------------------------------
        
#------ Training set last column has one missing value---
# for i in range(0, len(X_train[:, 5])):
  #   if pd.isna(X_train[i, 5]):
    #    X_train[i, 5] = 'None' # Replace missing value with none
#---------------------------------------------------------

df_train = pd.DataFrame(X_train)
df_test = pd.DataFrame(X_test)

nan_test = df_test.isnull().any()
nan_train = df_train.isnull().any()
# Encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:, 0] = labelencoder_X.fit_transform(X_train[:, 0])
# X_train[:, 5] = labelencoder_X.fit_transform(X_train[:, 5])
onehotencoder = OneHotEncoder(categorical_features=[0])
X_train = onehotencoder.fit_transform(X_train).toarray()
# onehotencoder = OneHotEncoder(n_values=4, categorical_features=[6])
# X_train = onehotencoder.fit_transform(X_train).toarray()

X_test[:, 0] = labelencoder_X.fit_transform(X_test[:, 0])
# X_test[:, 5] = labelencoder_X.fit_transform(X_test[:, 5])
onehotencoder = OneHotEncoder(categorical_features=[0])
X_test = onehotencoder.fit_transform(X_test).toarray()
# onehotencoder = OneHotEncoder(n_values=4, categorical_features=[6])
# X_test = onehotencoder.fit_transform(X_test).toarray()

# Fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier_logistics = LogisticRegression(random_state=0)
classifier_logistics.fit(X_train, y_train)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier_kernel_svm = SVC(kernel = 'rbf', random_state = 0)
classifier_kernel_svm.fit(X_train, y_train)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier_naive_bayes = GaussianNB()
classifier_naive_bayes.fit(X_train, y_train)

#Predicting the test set results
y_pred_logistics = classifier_logistics.predict(X_test)
y_pred_kernel_svm = classifier_kernel_svm.predict(X_test)
y_pred_naive_bayes = classifier_naive_bayes.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm_logistics = confusion_matrix(y_test, y_pred_logistics)
cm_kernel_svm = confusion_matrix(y_test, y_pred_kernel_svm)
cm_naive_bayes = confusion_matrix(y_test, y_pred_naive_bayes)

# Creating a csv for predictions of test_set
submission_logistics = pd.DataFrame(y_pred_logistics)
filename = 'Titanic-Survival-logistics.csv'
submission_logistics.to_csv(filename, index=False)

submission_naive_bayes = pd.DataFrame(y_pred_logistics)
filename = 'Titanic-Survival-naive_bayes.csv'
submission_naive_bayes.to_csv(filename, index=False)

submission_kernel_svm = pd.DataFrame(y_pred_logistics)
filename = 'Titanic-Survival-kernel_svm.csv'
submission_kernel_svm.to_csv(filename, index=False)

