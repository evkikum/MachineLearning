# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:59:45 2019

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

os.chdir(r'C:\Users\evkikum\Documents\Python Scripts\Practice\Classification\UC4_HR Analytics')

df = pd.read_csv('HR_comma_sep.csv')
df.info()
df.columns
df.isnull().sum()

df_stats = df.describe()

df_class_summary = df.groupby('left').agg(np.mean)
df_class_summary.info()

for i in df_class_summary.columns:
    plt.figure()
    plt.xlabel(i)
    df_class_summary[i].plot.bar()
    plt.show()
    
dk = df.drop('left' , axis = 1)
for k in df.columns[:6]:
    df.boxplot(column = k, by = 'left')
    

df_sal = pd.get_dummies(df['salary'], drop_first=True, prefix='salary')
df = pd.concat([df, df_sal], axis = 1)
df_dept = pd.get_dummies(df['Department'], drop_first=True, prefix='Dept')
df = pd.concat([df, df_dept], axis = 1)

df = df.drop(['salary', 'Department'], axis = 1)

X_train , X_test, y_train, y_test = train_test_split(df.drop('left', axis = 1), df['left'], test_size = 0.3, random_state = 1234)    
y_train.value_counts()
y_test.value_counts()

df_logit = LogisticRegression().fit(X_train, y_train)
pd.Series(df_logit.coef_[0], index = X_train.columns).sort_values(ascending=True)

pred_class_df = df_logit.predict(X_test)
pd.crosstab(y_test, pred_class_df)
accuracy_score(y_test, pred_class_df)  ## 78.3 % 

rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=5, scoring='accuracy')
rfecv = rfecv.fit(X_train, y_train)

print('Optimal no of features ', rfecv.n_features_)
print('Best features ', X_train.columns[rfecv.support_] )

rfecv.grid_scores_

plt.figure()
plt.xlabel('No of features selected')
plt.ylabel('Cross validation score of no of selected features')
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

##test_df = X_train.loc[:,X_train.columns[rfecv.support_]]

X_train_rfecv = rfecv.transform(X_train)  ## NOW THE X_train_rfecv WILL ONLY HAVE 13 COLUMNS 
X_test_rfecv = rfecv.transform(X_test)  ## NOW THE X_test_rfecv WILL ONLY HAVE 13 COLUMNS 


df_rfecv_model = LogisticRegression().fit(X_train_rfecv, y_train)
pred_rfecv = df_rfecv_model.predict(X_test_rfecv)

accuracy_score(y_test, pred_rfecv)  ## 79.11 %%

## HENCE THE ACCURACY WITHOUT RFECV IS 78.3 % WHILE ACCURACY WITH RFECV IS 79.11 %% .
## HENCE WE CAN CONCLUDE THAT ACCURACY WITH RFECV IS HIGHER BECAUSE OF SOME FEATURE ELIMINATION.

