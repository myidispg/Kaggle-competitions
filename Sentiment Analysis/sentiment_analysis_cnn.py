# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 17:11:08 2018

@author: Prashant Goyal
"""

import numpy as np
import pandas as pd

import os
# print(os.listdir("../input"))

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

# Seperating test and train set
df_train=df[df.Sentiment!=-999]
df_train.shape

df_test=df[df.Sentiment==-999]
df_test.drop('Sentiment',axis=1,inplace=True)
print(df_test.shape)
df_test.head()

del df
gc.collect()

# Bag of Words model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(ngram_range=(1,2),max_df=0.95,min_df=10,sublinear_tf=True)

c2_train=tfidf.fit_transform(df_train.clean_review).toarray()
print(c2_train.shape)
c2_test=tfidf.transform(df_test.clean_review).toarray()
print(c2_test.shape)

# Splitting the dataset into 20% test set and train set
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
train_text = df_train.clean_review.values
test_text = df_test.clean_review.values
target = df_train.Sentiment.values
y = to_categorical(target)
print(train_text.shape,target.shape,y.shape)

X_train_text,X_val_text,y_train,y_val=train_test_split(train_text,y,test_size=0.2,stratify=y,random_state=123)

# Finding the number of unique words in the training set
from nltk import FreqDist
all_words = ' '.join(X_train_text)
all_words = word_tokenize(all_words)
dist=FreqDist(all_words)
num_unique_word=len(dist)
num_unique_word

# Finding the max length of the review in train set
r_len=[]
for text in X_train_text:
    word=word_tokenize(text)
    l=len(word)
    r_len.append(l)
    
MAX_REVIEW_LEN=np.max(r_len)
MAX_REVIEW_LEN

max_features = num_unique_word
max_words = MAX_REVIEW_LEN
batch_size = 128
epochs = 3
num_classes=5

# Tokenize Text
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train_text))
X_train = tokenizer.texts_to_sequences(X_train_text)
X_val = tokenizer.texts_to_sequences(X_val_text)
X_test = tokenizer.texts_to_sequences(test_text)

# Sequence Padding
from keras.preprocessing import sequence
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_val = sequence.pad_sequences(X_val, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
print(X_train.shape,X_val.shape,X_test.shape)

# Applying CNN to the model
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D,Flatten,MaxPooling1D,GRU,SpatialDropout1D,Bidirectional
model= Sequential()
model.add(Embedding(max_features,100,input_length=max_words))
model.add(Dropout(0.2))

model.add(Conv1D(64,kernel_size=3,padding='same',activation='relu',strides=1))
model.add(GlobalMaxPooling1D())

model.add(Dense(num_classes,activation='relu'))
model.add(Dropout(0.2))

y_pred = model.predict_classes(X_test, verbose=1)

sub.Sentiment = y_pred
sub.to_csv('sub2.csv',index=False)
sub.head()

