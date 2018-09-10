# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 17:39:03 2018

@author: Prashant Goyal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the train dataset
dataset_train = pd.read_csv('train.csv')
X_train = dataset_train.iloc[:, [2,4,5,6,7,9,11]].values
y_train = dataset_train.iloc[:, 1].values

# Importing the test dataset
dataset_test = pd.read_csv('test.csv')
X_test = dataset_test.iloc[:, [1,3,4,5,6,8,10]].values
#------------------------------------

# Converting X_train from object to array
X_train = []
for i in range(0, len(X_train_object)):
    new = []
    for j in range(0, len(X_train_object[1, :])):
        new.append(X_train_object[i][j])
    X_train.append(new)
# -----------------------------------------        
# Checking for null values in train columns
df_train = pd.DataFrame(X_train)
nan_train = df_train.isnull().any()
df_train.dtypes
# Checking for null values in test columns
df_test = pd.DataFrame(X_test)
nan_test = df_test.isnull().any()

# Missing are 2(age) and 6(embarked is categorical type)

# Encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:, 1] = labelencoder_X.fit_transform(X_train[:, 1])
    # One Hot Encoding is not required because there are only two categories.
    # If we one hot encode, we would have to remove one column to avoid dummy variable trap.
    # This would bring us back to the same place.
    
# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X_train[:, 2:3])
X_train[:, 2:3] = imputer.transform(X_train[:, 2:3])

# Last column (column 6) is categorical so needs to be imputed differently
# We will replace the missing values with most_frequent

#------------------------------------------------------------------------------------------
missing_row_index = pd.isnull(df_train).any(1).nonzero()[0]
# Gives the row index of missing values.

categories_6 = {'C': 0, 'Q': 0, 'S': 0}
# This loops find the most frequent value
for i in range(0, len(X_train)):
    if pd.isnull(X_train[i, 6]):
        pass
    else:
        categories_6[X_train[i, 6]] += 1
        

# Replace missing value with most frequent
for i in missing_row_index:
    X_train[i, 6] = 'S'
    
# We found that S was most frequent. So replaced NAN with 'S' 
#-------------------------------------------------------------------------------------------
    
# Continuing with encoding missing data of column 6
onehotencoder_X_6 = OneHotEncoder(categorical_features =[6])
X_train[:, 6] = labelencoder_X.fit_transform(X_train[:, 6])
X_train = onehotencoder_X_6.fit_transform(X_train).toarray()

# Since the encoded variables were added to front, we remove one column to avoid dummy variable trap
X_train = X_train[:, 1:]

# Encoding test data
labelencoder_X_test = LabelEncoder()
labelencoder_X = LabelEncoder()
X_test[:, 1] = labelencoder_X.fit_transform(X_test[:, 1])

# Imputing missing test data
imputer_test = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer_test = imputer_test.fit(X_train[:, 2:3])
X_test[:, 2:3] = imputer_test.transform(X_test[:, 2:3])

imputer_test = imputer_test.fit(X_test[:, 5:6])
X_test[:, 5:6] = imputer_test.transform(X_test[:, 5:6])

# Continuing with encoding missing data of test column 6
onehotencoder_X_6 = OneHotEncoder(categorical_features =[6])
X_test[:, 6] = labelencoder_X.fit_transform(X_test[:, 6])
X_test = onehotencoder_X_6.fit_transform(X_test).toarray()

# Since the encoded variables were added to front, we remove one column to avoid dummy variable trap
X_test = X_test[:, 1:]


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# We can now apply various fitting methods.
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', C=1, gamma=0.7, random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1,10,100,1000], 'kernel':['linear']},
               {'C':[1,10,100,1000], 'kernel':['rbf'], 'gamma': [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]},
               {'C':[1,10,100,1000], 'kernel':['poly'], 'degree':[3,4,5,6,7], 'gamma': [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]}
               ]

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_