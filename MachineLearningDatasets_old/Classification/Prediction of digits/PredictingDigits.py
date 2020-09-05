# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:18:01 2019

@author: evkikum
"""

## Below is demo in the below video
## https://www.youtube.com/watch?v=J5bXOOmkopc&list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw&index=9

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn



def MAPE(actual, predicted):
    actual_np = np.array(actual)
    predicted_np = np.array(predicted)
    aps = abs(actual_np - predicted_np) * 100/actual_np
    aps = aps[np.isfinite(aps)]
    mean_aps = np.mean(aps)
    median_aps = np.median(aps)
    return pd.Series([mean_aps, median_aps], index = ['Mean', "Median"])

digits = load_digits()
digits.data[0]

model = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.2)

model.fit(X_train, y_train)
model.score(X_train, y_train)  ## 99.4 %%
model.score(X_test, y_test) ## 94.99%

MAPE(y_train, model.predict(X_train))  ## 41% Mean , 0 Median
MAPE(y_test, model.predict(X_test))  ## 

y_predicted = model.predict(X_test)


## confusion Matrix
cm = confusion_matrix(y_test, y_predicted)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')



