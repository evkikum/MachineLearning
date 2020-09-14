# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 06:20:41 2019

@author: evkikum
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 06:09:16 2019

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
df.isnull().sum()


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

df_logit = LogisticRegression().fit(X_train, y_train)
pd.Series(df_logit.coef_[0], index = X_train.columns).sort_values(ascending=True)

pred_class_df = df_logit.predict(X_test)
pd.crosstab(y_test, pred_class_df)
accuracy_score(y_test, pred_class_df)  ## 78.3% 

## TPR - 352/(758 + 352) ==> 31.7 % 
## FPR - 218/(218 + 3172) ==> 6.4 % 

pred_prob = df_logit.predict_proba(X_test)
pred_prob_class = np.column_stack([pred_prob, pred_class_df])
pred_prob_df = pred_prob[:, 1]  # Extracting probability of class 1
pred_class_df = np.zeros(len(y_test))
pred_class_df[pred_prob_df >= 0.5] = 1

pd.crosstab(y_test, pred_class_df)
accuracy_score(y_test, pred_class_df)  ## 78.3 %

## TPR - 352/(352 + 758) = 31.7 %
## FPR - 218/(3172 + 218) = 6.4 % 


## calculatiin of ROC (RECEIVER OPERATION CHARACTERISTIC CURVE)
df_fpr, df_tpr, df_threshold =  roc_curve(y_test, pred_prob_df)

df_roc = pd.DataFrame({'Threshold' :df_threshold,
                       'FPR' : df_fpr,
                       'TPR' : df_tpr})

plt.plot(df_fpr, df_tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positibe Rate')


## AREA UNDER THE CURVE
## note -- the range of AUC is 0.5 to 1. 0.5 is bad model and 1 is idle model.
auc(df_fpr, df_tpr)  ## 82.9 %  






