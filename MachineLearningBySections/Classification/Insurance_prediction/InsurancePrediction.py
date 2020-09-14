# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:10:16 2019

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

os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\Practice\Logistic Regression\Insurance_prediction")

df = pd.read_csv('insurance_data.csv')
df.info()

sns.scatterplot(x='age', y='bought_insurance', data = df)


## IN THE BELOW df[['age']] has to be used [] twice before the 1st arguments should multidimensional array
X_train, X_test, y_train, y_test = train_test_split(df[['age']], df['bought_insurance'], test_size=0.3)

model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_train, y_train)  ## 94.44 %
model.predict(X_train)

model.score(X_test, y_test) ## 77.7 %
model.predict(X_test)



