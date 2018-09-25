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

# Now, we will deal with the missing values in age column by getting their broad values by the initial of names.
# Mr., Mrs. will be aged 18-60, master and miss will be 10-18.

child_list_index = []
adult_list_index = []

# This list is just to check whether all the titles are added. The length of this string must be equal to the length of train + test set.
names_list = []

# Using this loop, we can get the missing age indexes.
for i in range(0, len(train)):
    # Split by , to get last name seperate and first name seperate.
    names = train.Name.iloc[i].split(',')
    # The first name part will always be at the end of the list
    names = names[1]
    # Split by . to get initial in the first place.
    names = names.split('.')
    names_list.append(names[0])