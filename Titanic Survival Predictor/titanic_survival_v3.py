# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 19:34:44 2018

@author: Prashant Goyal
"""

import numpy as np
import pandas as pd

# Importing the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Since passengerID is not useful and Cabin has a lot of missing values, we can drop them
train.drop(['PassengerId', 'Cabin'], axis=1, inplace=True)
test.drop(['PassengerId', 'Cabin'], axis=1, inplace=True)

# Also dropping the ticket column. No relevance.
train.drop('Ticket', axis=1, inplace=True)
test.drop('Ticket', axis=1, inplace=True)

# Adding a survived column to test set to concatenate it to trains set.
test['Survived'] = 999

# Concatenating both the sets into a single dataframe
df = pd.concat([train,test], axis = 0)

# Finding number of null values
train.isna().sum()
test.isna().sum()
df.isna().sum()

# Deleting train and test
del train, test

# Now, we will deal with the missing values in age column by getting their broad values by the initial of names.
# Mr., Mrs. will be aged 18-60, master and miss will be 10-18.

# This list is also to check whether all the titles are added. The length of this string must be equal to the length of train + test set.
names_list = []

# Using this loop, we can get the missing age indexes.
for i in range(0, len(df)):
    # Split by , to get last name seperate and first name seperate.
    names = df.Name.iloc[i].split(',')
    # The first name part will always be at the end of the list
    names = names[1]
    # Split by . to get initial in the first place.
    names = names.split('.')
    names_list.append(names[0])
    
# Now name is not needed, we can drop the column.
df.drop('Name', axis=1, inplace=True)
    
# Getting the index of Mr. and Mrs. in one list, Master and Miss in other.
child_index = []
adult_index = []

for i in range(len(names_list)):
    if names_list[i] == ' Mr' or names_list[i] == ' Mrs':
        if pd.isnull(df.Age.iloc[i]):
            adult_index.append(i)
    else:
         if pd.isnull(df.Age.iloc[i]):
            child_index.append(i)
        
# ^^ Now we have the adult and child indexes seperate, we can now assign random ages in respective age groups
max_age = int(df.Age.max())      
min_age = int(df.Age.min())

import random
for i in adult_index:
    df.Age.iloc[i] = random.randint(18, max_age)
for i in child_index:
    df.Age.iloc[i] = random.randint(min_age, 18)

del i, names, names_list,max_age, min_age, adult_index, child_index

# Now we will deal with missing values in Embarked and Fare category.
embarked_mode = df.Embarked.mode().values
# Since the mode is 'S', we will replace the missing values in Embarked column with S
df['Embarked'].fillna('S', inplace=True)

fare_mean = int(df.Fare.mean())
df['Fare'].fillna(fare_mean, inplace=True)

del embarked_mode, fare_mean

#-----Finished data preprocessing--------

# LabelEncoding and OneHotEncoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
df.Embarked = labelencoder.fit_transform(df.Embarked.values)
df.Sex = labelencoder.fit_transform(df.Sex.values)
    # Encoding a dataframe
df = pd.get_dummies(df,columns=['Embarked', 'Sex'], drop_first=True)
#------Completed all encodings.------------

# Splitting the df into train and test set
df_train = df[df.Survived != 999]
df_test = df[df.Survived == 999]


df_test.drop('Survived', axis=1, inplace=True)

del df

df_train.isna().sum()
df_test.isna().sum()

# Converting to array
X_train = df_train.iloc[:,[0,1,2,3,4,6,7,8]].values
y_train = df_train.iloc[:, 5].values

X_test = df_test.iloc[:, :].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier_forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_forest.fit(X_train, y_train)

# Fitting Extra Tress Classifier
from sklearn.ensemble import ExtraTreesClassifier
etc=ExtraTreesClassifier(n_estimators=400)
etc.fit(X_train,y_train)
y_pred_extra = etc.predict(X_test)

# Fitting a neural network
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(16, activation='relu', kernel_initializer='uniform', input_shape=(8,)))
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))
classifier.add(Dense(output_dim = 1, init='uniform', activation='sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.summary()

classifier.fit(X_train, y_train, batch_size=10, epochs = 100)

# Predicting the Test set results
y_pred_forest = classifier_forest.predict(X_test)

y_pred = classifier.predict(X_test)
# Reshape to a 1-D Array
y_pred = np.reshape(y_pred, -1)

# Convert to 0s and 1s.
y_pred = [int(round(elem,0)) for elem in y_pred]

# Making the Confusion Matrix for random forest and NN.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred_extra, y_pred)

index_list = []
for i in range(892, 1310):
    index_list.append(i)

df_pred_extra = pd.DataFrame({'PassengerID': index_list, 'Survived':y_pred_extra}) 
df_pred_forest = pd.DataFrame({'PassengerID': index_list, 'Survived':y_pred_forest}) 
df_pred_nn = pd.DataFrame({'PassengerID': index_list, 'Survived':y_pred})

del index_list, X_test, X_train, df_test, df_train, i, y_pred, y_train, y_pred_extra, y_pred_forest
# Creating a csv for predictions of test_set
filename = 'Titanic-Survival-Neural-Network.csv'
df_pred_nn.to_csv(filename, index=False)

# Creating a csv for predictions of test_set
filename = 'Titanic-Survival-Extra_Trees.csv'
df_pred_extra.to_csv(filename, index=False)

# Creating a csv for predictions of test_set
filename = 'Titanic-Survival-Extra_Trees.csv'
df_pred_forest.to_csv(filename, index=False)


