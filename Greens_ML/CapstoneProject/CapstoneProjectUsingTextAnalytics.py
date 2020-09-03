# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:38:31 2019

@author: evkikum
"""


import pandas as pd
import numpy as np
import nltk # natural language toolkit
#nltk.download()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # TF, TFIDF
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from textblob import TextBlob
import os
import glob

os.chdir(r'C:\Users\evkikum\OneDrive - Ericsson AB\Python Scripts\GreenInstitute_Course\Capstoneproject')

##all_files = os.listdir(r'C:\Users\evkikum\OneDrive - Ericsson AB\Python Scripts\GreenInstitute_Course\Capstoneproject\text_topics')
all_files = glob.glob('text_topics/*.txt')

tr_tweets = []
for i in all_files:
    f = open(i, 'r')    
    temp = f.readlines()
    f.close()
    tr_tweets.append(temp)
    ##break



    tr_tweets.append(temp)        

    break
    
tr_sentiment = pd.read_csv('target.csv')



twt_vectorizer1 = CountVectorizer(lowercase=False, stop_words=None)
twt_tr_vector1 = twt_vectorizer1.fit(tr_tweets)
twt_tr_vector1_feat = twt_tr_vector1.get_feature_names()
twt_tr_vector1_transform = twt_tr_vector1.transform(tr_tweets)

twt_tr_vector1_transform_df = pd.DataFrame(twt_tr_vector1_transform.toarray(), columns = twt_tr_vector1_feat)
 
twt_model1 = DecisionTreeClassifier().fit(twt_tr_vector1_transform, tr_sentiment)   