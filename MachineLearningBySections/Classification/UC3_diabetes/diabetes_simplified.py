# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:44:50 2019

@author: evkikum
"""

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\Practice\Logistic Regression\diabetes")

diabetes = pd.read_csv('diabetes.csv')
diabetes_stats = diabetes.describe()
diabetes_corr = diabetes.corr()

plt.figure(figsize=(16,10))
sns.heatmap(diabetes.corr(),annot=True ,cmap='YlGnBu')

X = diabetes.iloc[:,:8]
y = diabetes.iloc[:,8]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_train, y_train)  ## 76.3 %%
model.score(X_test, y_test)  ## 77% 

predicted_train = model.predict(X_train)    
predicted_test = model.predict(X_test)

df_train = pd.concat([X_train, y_train], axis = 1)
df_train= df_train.reset_index()
df_train = pd.concat([df_train, pd.Series(predicted_train)], axis = 1)


df_test = pd.concat([X_test, y_test], axis = 1)
df_test= df_test.reset_index()
df_test = pd.concat([df_test, pd.Series(predicted_test)], axis = 1)