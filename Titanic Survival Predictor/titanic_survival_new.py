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
data_df =dataset_train
data_df['Family Size'] = data_df['Parch'] + data_df['SibSp']
family = dataset_train['SibSp'].values + dataset_train['Parch'].values
# X_train = dataset_train.iloc[:, [2,4,5,family,9,11]].values
X_train = dataset_train.iloc[:, [2,4,5,6,7,9,11]].values
# X_train = dataset_train.iloc[:, [2,4,5,6,7,8,9,11]].values
# X_train = dataset_train.iloc[:, [2,4,5]].toArray()
X_train = X_train.concate(family)
y_train = dataset_train.iloc[:, 1].values

# Importing the test dataset
dataset_test = pd.read_csv('test.csv')
X_test = dataset_test.iloc[:, [1,3,4,5,6,8,10]].values

dataset_y_test = pd.read_csv('gender_submission.csv')
y_test = dataset_y_test.iloc[:, 1].values
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

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# We can now apply various fitting methods.
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', C=1, gamma=0.19, random_state = 0)
classifier.fit(X_train, y_train)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_forest = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)
classifier_forest.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred_forest = classifier_forest.predict(X_test)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier_forest, X = X_train, y = y_train, cv = 10)
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
                           cv = 5,
                           verbose = 5,
                           n_jobs = -1)

def gridSearch(xtrain, ytrain):
    X_train = xtrain
    y_train = ytrain
    classifier = SVC()
    parameters = [{'C':[1,10,100,1000], 'kernel':['linear']},
               {'C':[1,10,100,1000], 'kernel':['rbf'], 'gamma': [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]},
               {'C':[1,10,100,1000], 'kernel':['poly'], 'degree':[3,4,5,6,7], 'gamma': [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]}
               ]
    grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters, 
                           scoring = 'accuracy',
                           cv = 5,
                           verbose = 5,
                           n_jobs = 4)
    grid_search = grid_search.fit(X_train, y_train)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_
    
if __name__ == '__main__':
    gridSearch(X_train, y_train)    


# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm_kernel_svm = confusion_matrix(y_test, y_pred)

# Making the CSV file
submission_kernel_svm = pd.DataFrame(y_pred)
filename = 'Titanic-Survival-kernel_svm.csv'
submission_kernel_svm.to_csv(filename, index=False)

submission_forest = pd.DataFrame(y_pred_forest)
filename = 'Titanic-Survival-random-forest.csv'
submission_forest.to_csv(filename, index=False)

# Plotting survived status against different features extracted using PCA
pos_index = [] # indexes of all positives in y
for i in range(0, len(y_train)):
    if y_train[i] == 1:
        pos_index.append(i) 

neg_index = []
for i in range(0, len(y_train)):
    if y_train[i] == 0:
        neg_index.append(i) 
        
pos_0 = []
for index in pos_index:
    pos_0.append(X_train_pca[index, 0])
    
pos_1 = []
for index in pos_index:
    pos_1.append(X_train_pca[index, 1])
    
neg_0 = []
for index in neg_index:
    neg_0.append(X_train_pca[index, 0])
    
neg_1 = []
for index in neg_index:
    neg_1.append(X_train_pca[index, 1])
    
y_pos = []
for index in pos_index:
    y_pos.append(y_train[index])
    
y_neg = []
for index in neg_index:
    y_neg.append(y_train[index])
    
plt.scatter(pos_0, y_pos, color='red')
plt.scatter(X_train_pca[:, 1], y_train, color='blue')
plt.legend()
plt.show()

