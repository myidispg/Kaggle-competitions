# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 06:34:08 2018

@author: Prashant Goyal
"""

import numpy as np
import pandas as pd

import os
print(os.listdir("../input"))

import gc
pd.set_option('display.max_colwidth', 100)

gc.collect()


# Loading the dataset
train = pd.read_csv('train.tsv', sep='\t')
print(train.shape)
train.head()

test = pd.read_csv('test.tsv', sep='\t')
print(test.shape)
test.head()

sub = pd.read_csv('sampleSubmission.csv')
sub.head()

# Adding sentiment column to test dataset and join test and train for preprocessing
test['Sentiment'] = -999
test.head()

df = pd.concat([train,test], ignore_index=True)
print(df.shape)
df.tail()

del train,test
gc.collect()

# Cleaning reviews
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
stemmer = SnowballStemmer('english')
lemma = WordNetLemmatizer()
from string import punctuation
import re
from bs4 import BeautifulSoup

def clean_review(review_col):
    review_corpus = []
    for i in range(0, len(review_col)):
        review = str(review_col[i])
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = [lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review = ''.join(review)
        review_corpus.append(review)
    return review_corpus

df['clean_review'] = clean_review(df.Phrase.values)
df.head()

