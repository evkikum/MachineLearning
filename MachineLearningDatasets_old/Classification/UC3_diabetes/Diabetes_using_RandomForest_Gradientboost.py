# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:18:04 2019

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

os.chdir(r'C:\Users\evkikum\Documents\Python Scripts\GreenInstitute_Course')

df = pd.read_csv('data/diabetes_data.csv')
df.info()
df_scale = df.describe()

df_mean_features = df.groupby('Class').agg(np.mean)

for i in df.columns[:8]:
    df.boxplot(column = i, by = 'Class')
    
sns.lmplot('Plasma glucose concentration', 'Age', hue = 'Class', data = df, fit_reg = False)

X_train, X_test, y_train, y_test = train_test_split(df.drop('Class', axis = 1), df['Class'], test_size = 0.3, random_state = 1234)

y_train.value_counts()
y_test.value_counts()

## Decision Tree
df_dtree1 = DecisionTreeClassifier(max_depth = 2, random_state=1234).fit(X_train, y_train)
df_dtree1.feature_importances_
pd.Series(df_dtree1.feature_importances_, index = df.columns[:8]).sort_values(ascending = False)


df_dtree2 = DecisionTreeClassifier(criterion='entropy' ,max_depth = 2, random_state=1234).fit(X_train, y_train)
df_dtree2.feature_importances_

df_dtree3 = DecisionTreeClassifier(max_depth = 3, random_state=1234).fit(X_train, y_train)
df_dtree3.feature_importances_


## Random Forest
df_rf = RandomForestClassifier(n_estimators = 5, max_depth = 2).fit(X_train, y_train)
df_rf.feature_importances_

## Gradient Boosting

df_gbm = GradientBoostingClassifier(n_estimators = 20).fit(X_train, y_train)
df_gbm.feature_importances_

## Decision Tree
df_pred_dtree1 = df_dtree1.predict(X_test)
pd.crosstab(y_test, df_pred_dtree1)
accuracy_score(y_test, df_pred_dtree1)  ## 76.27 %

df_pred_dtree2 = df_dtree2.predict(X_test)
pd.crosstab(y_test, df_pred_dtree2)
accuracy_score(y_test, df_pred_dtree2)  ## 76.27% 

df_pred_dtree3 = df_dtree3.predict(X_test)
pd.crosstab(y_test, df_pred_dtree3)
accuracy_score(y_test, df_pred_dtree3)  ## 72.88 % 

## Random Forest
df_pred_rf = df_rf.predict(X_test)
pd.crosstab(y_test, df_pred_rf)
accuracy_score(y_test, df_pred_rf)  ## 74.57 % 

## Gradient Boosting
df_pred_gbm = df_gbm.predict(X_test)
pd.crosstab(y_test, df_pred_gbm)
accuracy_score(y_test, df_pred_gbm)  ##  77.1 % 


## Parameter tunning for decision Tree


dtree_series = pd.Series([0.0] * 29, index = range(1,30))
for k in range(1,30):
    print(' k ', k,
          'Accuracy =',
          np.mean(cross_val_score(DecisionTreeClassifier(max_depth = k, random_state=1234),
                                  df.drop('Class', axis = 1),
                                  df['Class'],
                                  cv = 5) ) ) 
    dtree_series[k] = np.mean(cross_val_score(DecisionTreeClassifier(max_depth = k, random_state=1234),
                                  df.drop('Class', axis = 1),
                                  df['Class'],
                                  cv = 5) )         ## k = 3 is the optimal parameter with 76.55 % 


## Parameter tunning for Random forest

rf_series = pd.Series([0.0]* 29, index = range(1,30))    

for k in range(1,30):
    print('k ', k,
          'Accuracuy ',
          np.mean(cross_val_score(RandomForestClassifier(n_estimators = k, random_state = 1234),
                                  df.drop('Class', axis = 1),
                                  df['Class'],
                                  cv = 5 )))
    rf_series[k] = np.mean(cross_val_score(RandomForestClassifier(n_estimators = k, random_state = 1234),
                                  df.drop('Class', axis = 1),
                                  df['Class'],
                                  cv = 5))   ## k = 9 is the optimal paramter with 79.35 % accuracy
        

## Parameter tunning for  Gradient Boosting
gbm_series = pd.Series([0.0] * 29, index = range(1,30))

for k in range(1,30):
    print('k ', k,
          'Accuracy ',
          np.mean(cross_val_score(GradientBoostingClassifier(n_estimators = k),
                                  df.drop('Class', axis = 1),
                                  df['Class'],
                                  cv = 5)))
    gbm_series[k] = np.mean(cross_val_score(GradientBoostingClassifier(n_estimators = k),
                                  df.drop('Class', axis = 1),
                                  df['Class'],
                                  cv = 5) )  ## k = 24 is the optimal parameter with 80% percentage
        