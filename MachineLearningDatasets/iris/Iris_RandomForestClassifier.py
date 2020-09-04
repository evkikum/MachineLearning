#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 16:56:11 2020

@author: evkikum
"""


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn


iris = load_iris()

df = pd.DataFrame(iris.data, columns = iris.feature_names)
df.head()
df['target'] = iris.target
df['target'] = df['target'].apply(lambda x : iris.target_names[x])

X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis = 1), df['target'], test_size = 0.3)

model = RandomForestClassifier(n_estimators = 50,max_depth = 3 )
model.fit(X_train, y_train)

model.score(X_train, y_train) ## 100% 
model.score(X_test, y_test)  ## 97.77 

np.mean(cross_val_score(RandomForestClassifier(n_estimators = 50, max_depth = 3), df.iloc[:, :4], df["target"], cv = 5))


iris_cross_vldt_accuracy = pd.Series([0.0]*19, range(1,20))
i = 0

for k in range(10,200,10):
    iris_crossval_rf_anyD = cross_val_score(RandomForestClassifier(n_estimators = k), df.iloc[:, :4], df["target"], cv = 5)
    iris_cross_vldt_accuracy[i] = np.mean(iris_crossval_rf_anyD)
    i = i + 1
    
    
print(iris_cross_vldt_accuracy)  ## n_estimators = 50 is optimum


iris_cross_vldt_accuracy = pd.Series([0.0]*19, range(1,20))
i = 0

for k in range(1,20,1):
    iris_crossval_rf_anyD = cross_val_score(RandomForestClassifier(max_depth = k), df.iloc[:, :4], df["target"], cv = 5)
    iris_cross_vldt_accuracy[i] = np.mean(iris_crossval_rf_anyD)
    i = i + 1
    
    
print(iris_cross_vldt_accuracy)