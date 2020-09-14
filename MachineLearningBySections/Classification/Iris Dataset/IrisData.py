# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:01:18 2019

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
import seaborn as sn
import os


def MAPE(actual, predicted):
    actual_np = np.array(actual)
    predicted_np = np.array(predicted)
    aps = abs(actual_np - predicted_np) * 100/actual_np
    aps = aps[np.isfinite(aps)]
    mean_aps = np.mean(aps)
    median_aps = np.median(aps)
    return pd.Series([mean_aps, median_aps], index = ['Mean', "Median"])

os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\GreenInstitute_Course")

irisdata = pd.read_csv("data/iris.csv")


X = irisdata.iloc[:,:4]
y = irisdata.iloc[:, 4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_train, y_train)  ## 94.28 % 
model.predict_proba(X_train)
model.score(X_test, y_test)  ## 95.55 % 
model.predict_proba(X_test)

predicted_train_scrore = model.predict(X_train)
predicted_test_scrore = model.predict(X_test)

df_train = pd.concat([X_train, y_train], axis = 1)
df_train= df_train.reset_index()
df_train = pd.concat([df_train, pd.Series(predicted_train_scrore)], axis = 1)

df_test = pd.concat([X_test, y_test], axis = 1)
df_test= df_test.reset_index()
df_test = pd.concat([df_test, pd.Series(predicted_test_scrore)], axis = 1)