# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 20:51:36 2019

@author: evkikum
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.model_selection import train_test_split, cross_val_score # latest version of sklearn
from sklearn.cross_validation import train_test_split, cross_val_score # older version of sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
import os

os.chdir(r'C:\Users\evkikum\Documents\Python Scripts\GreenInstitute_Course')

df = pd.read_csv('data/diabetes_data.csv')
df.info()

for i in df.columns[:8]:
    df.boxplot(column = i, by = 'Class')
    
## FROM THE BELOW GRAGH IT SHOWS THAT DATA IS MESSY AND IT IS HARD A MODEL WITH GOOD ACCURACY.
sns.lmplot('Plasma glucose concentration', 'Age', hue = 'Class', data = df, fit_reg = False)

X_train, X_test, y_train, y_test = train_test_split(df.drop('Class', axis = 1), df['Class'], test_size = 0.3, random_state = 1234)

y_train.value_counts()
y_test.value_counts()

diab_logit = LogisticRegression().fit(X_train, y_train)
diab_logit.coef_
pd.Series(diab_logit.coef_[0], index = df.columns[:8]).sort_values(ascending=True)

pred_class_diab = diab_logit.predict(X_test)
pd.crosstab(y_test, pred_class_diab)
accuracy_score(y_test, pred_class_diab)  ## 79.6% 

rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=5, scoring='accuracy')
rfecv = rfecv.fit(X_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X_train.columns[rfecv.support_])

rfecv.grid_scores_

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

x_train_rfecv = rfecv.transform(X_train)
x_test_rfecv = rfecv.transform(X_test)

lr_rfecv_model = LogisticRegression().fit(x_train_rfecv, y_train)
pred_rfecv = lr_rfecv_model.predict(x_test_rfecv)

accuracy_score(y_test, pred_rfecv)  ## 79%