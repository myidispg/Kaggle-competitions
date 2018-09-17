# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 06:07:49 2018

@author: Prashant Goyal
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset_train = pd.read_csv('train.tsv', delimiter = '\t', quoting = 3)
X_train = dataset_train.iloc[:, 2:3].values
y_train = dataset_train.iloc[:, 3:].values

dataset_test = pd.read_csv('test.tsv', delimiter = '\t', quoting = 3)
X_test = dataset_test.iloc[:, 2:].values

# Cleaning the datasets
import re
import nltk # library for nlp
nltk.download('stopwords')
# ^^  downloads a list of common words that are not relevant to reviews like this, that, prepositions etc
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# ^^ Stemming, keeping only root of the words. Loved changes to love, hated changes to hate etc.
corpus = []

for i in range(0, len(dataset_train)):
    
    review = re.sub('[^a-zA-Z]', ' ', dataset_train['Phrase'][i] )
    # ^^ Here we removed all the characters except a-z and A-Z and replaced them with a space.
    review = review.lower() # Converted all characters to lowercase
    review = review.split() # removed all spaces and returns a string
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = " ".join(review) # Convert list of processed words to a string seperated by space
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer() # This keeps only the 1500 most occuring words.
X_train = cv.fit_transform(corpus).toarray()
y_train = dataset_train.iloc[:, 3].values

# Splitting the train dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X__train_test, y_train, y_train_test = train_test_split(X_train, y_train, test_size = 0.20, random_state = 0)

# Fitting Logistic learning to the Training set
from sklearn.linear_model import LogisticRegression
classifier_logistics = LogisticRegression(random_state=0)
classifier_logistics.fit(X_train, y_train)

# Predicting the Test set results
y_pred_logistics = classifier_logistics.predict(X__train_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_logistics = confusion_matrix(y_train_test, y_pred_logistics)