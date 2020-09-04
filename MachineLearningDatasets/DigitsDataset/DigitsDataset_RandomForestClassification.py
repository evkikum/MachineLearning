#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 16:39:35 2020

@author: evkikum
"""


import pandas as pd
from sklearn.datasets import load_digits
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn

digits = load_digits()
df = pd.DataFrame(digits.data)
df['target'] = digits.target
df.head()

X = df.drop('target', axis = 1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

model = RandomForestClassifier(n_estimators = 10)
model.fit(X_train, y_train)

model.score(X_train, y_train)  ## 100 %
model.score(X_test, y_test)  # 97.5 %

## THE DEFAULT VALUE OF n_estimators = 100,





y_predicted = model.predict(X_test)

cm = confusion_matrix(y_test, y_predicted)
cm


plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")