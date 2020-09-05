# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:44:12 2019

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
from sklearn.metrics import accuracy_score, roc_curve, auc
import os

os.chdir(r'C:\Users\evkikum\Documents\Python Scripts\Practice\Classification\UC4_HR Analytics')


df = pd.read_csv('HR_comma_sep.csv')
df.info()
df['Department'].value_counts()
df['salary'].value_counts()

df_class_summary = df.groupby('left').agg(np.mean)

for i in df_class_summary.columns:
    plt.figure()
    plt.xlabel(i)
    df_class_summary[i].plot.bar()
    
for i in df.columns[:7]:    
    df.boxplot(column = i, by = 'left')

df_sal = pd.get_dummies(df['salary'], drop_first=True, prefix = 'salary')   
df = pd.concat([df, df_sal], axis = 1)

df_dept = pd.get_dummies(df['Department'], drop_first=True, prefix = 'dept')
df = pd.concat([df,df_dept], axis = 1)

df = df.drop(['salary', 'Department'], axis = 1)

X_train , X_test, y_train, y_test = train_test_split(df.drop('left', axis = 1), df['left'], test_size = 0.3, random_state = 1234)


## KNN

df_knn1 = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
df_knn2 = KNeighborsClassifier(n_neighbors=2).fit(X_train, y_train)

## Decision Tree
df_dtree1 = DecisionTreeClassifier(max_depth = 2,random_state=1234).fit(X_train, y_train)
df_dtree1.feature_importances_
df_dtree2 = DecisionTreeClassifier(criterion='entropy', max_depth = 2,random_state=1234).fit(X_train, y_train)
df_dtree2.feature_importances_
df_dtree3 = DecisionTreeClassifier(max_depth = 3,random_state=1234).fit(X_train, y_train)
df_dtree3.feature_importances_


## Random Forest

## In the below n_estimators = 5 means to include 5 parallel decision tree model creation.
df_rf = RandomForestClassifier(n_estimators = 5,max_depth = 2).fit(X_train, y_train)
df_rf.feature_importances_

## Gradient Boosting
df_gbm =  GradientBoostingClassifier(n_estimators = 20).fit(X_train, y_train)
df_gbm.feature_importances_


## KNN
df_pred_tr = df_knn1.predict(X_train)
pd.crosstab(y_train, df_pred_tr)
accuracy_score(y_train, df_pred_tr)  ## 100 %

df_pred_tr2 = df_knn2.predict(X_train)
pd.crosstab(y_train, df_pred_tr2)
accuracy_score(y_train, df_pred_tr2)  ## 98.7 %

df_pred_te = df_knn1.predict(X_test)
pd.crosstab(y_test, df_pred_te)
accuracy_score(y_test, df_pred_te)  ## 95.4%%

df_pred_te2 = df_knn2.predict(X_test)
pd.crosstab(y_test, df_pred_te2)
accuracy_score(y_test, df_pred_te2)  ## 94.95

## Decision Tree
df_pred_dtree1 = df_dtree1.predict(X_test)
pd.crosstab(y_test, df_pred_dtree1)
accuracy_score(y_test, df_pred_dtree1)   ## 84.5 %

df_pred_dtree2 = df_dtree2.predict(X_test)
pd.crosstab(y_test, df_pred_dtree2)
accuracy_score(y_test, df_pred_dtree2)   # 81.5 % 

df_pred_dtree3 = df_dtree3.predict(X_test)
pd.crosstab(y_test, df_pred_dtree3)
accuracy_score(y_test, df_pred_dtree3)   # 95 %

## RANDOM FOREST
df_pred_rf = df_rf.predict(X_test)
pd.crosstab(y_test, df_pred_rf)
accuracy_score(y_test, df_pred_rf)  ## 78.37 %

## Gradient Boosting
df_pred_gbm = df_gbm.predict(X_test)
pd.crosstab(y_test, df_pred_gbm)  
accuracy_score(y_test, df_pred_gbm)  ## 94.64

## KNN Cross Validation
np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=1), 
                        df.drop('left', axis = 1), 
                        df['left'],
                        cv = 5))  ## 94.93%%




## KNN Parameter tunning

for k in range(1,11):
    print('Value of k', k,
          "Accuracy == ",          
          np.mean(cross_val_score(KNeighborsClassifier(n_neighbors = k),
                                  df.drop('left', axis = 1),
                                  df['left'],
                                  cv = 5)))  ## THE OPTIMAL VALUE IS 2 WITH 95%
    
## Decision Tree Parameter tunning

for k in range(1,15):
    print('Value of k', k,
          "Accuracy ==",
          np.mean(cross_val_score(DecisionTreeClassifier(max_depth = k, random_state=1234),
                                  df.drop('left', axis = 1),
                                  df['left'],
                                  cv = 5)))  ## THE OPTIMAL VALUE IS 8 WITH 97.8% 


## Random Forest parameter tunning

for k in range(1,30):
    print('Value of k ', k,
          'Accuracy = ',
          np.mean(cross_val_score(RandomForestClassifier(n_estimators = k, random_state = 1234), 
                                  df.drop('left', axis = 1),
                                  df['left'],
                                  cv = 5)))  ## THe optimal value is 13 and 99 %
    
## Gradient Boosting

pred_gbm_series = pd.Series([0.0]*29, index =  range(1,30))
max(pred_gbm_series)
for k in range(1,30):
    print('Value of k ', k,
          'Accuracy of = ',
          np.mean(cross_val_score(GradientBoostingClassifier(n_estimators = k, random_state=1234),
                                  df.drop('left', axis = 1),
                                  df['left'],
                                  cv = 5)) )
    pred_gbm_series[k] = np.mean(cross_val_score(GradientBoostingClassifier(n_estimators = k, random_state=1234),
                                  df.drop('left', axis = 1),
                                  df['left'],
                                  cv = 5))  ## The optimal value is 14 and 96.67 % 
    
    
