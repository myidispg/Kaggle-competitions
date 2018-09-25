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

del i, names, names_list,max_age, min_age

# Now we will deal with missing values in Embarked and Fare category.
embarked_mode = df.Embarked.mode().values
# Since the mode is 'S', we will replace the missing values in Embarked column with S
df['Embarked'].fillna('S', inplace=True)

fare_mean = int(df.Fare.mean())
df['Fare'].fillna(fare_mean, inplace=True)

del embarked_mode, fare_mean

# Splitting the df into train and test set
df_train = df[df.Survived != 999]
df_test = df[df.Survived == 999]

del df

