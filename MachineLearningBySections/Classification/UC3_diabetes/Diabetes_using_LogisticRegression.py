# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:32:55 2019

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
import os

os.chdir(r'C:\Users\evkikum\Documents\Python Scripts\GreenInstitute_Course')

df = pd.read_csv('data/diabetes_data.csv')
df.info()
df_mean_features = df.groupby('Class').agg(np.mean)

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


## THE DEFAULT PREDICT FUNCTION USES PROBABILITY OF 0.5
## THRESHOLD CAN BE REDUCED TO INCREASE TPR (SENSTIVITY)
## THRESHOLD CAN BE INCREASED TO REDUCE FPR (SPECIFICITY)

pred_prob = diab_logit.predict_proba(X_test)
pred_prob_class = np.column_stack([pred_prob,pred_class_diab ])

pred_prob_diab = pred_prob[:,1] # extracting probability of Class 1
pred_class_diab = np.zeros(len(y_test))
pred_class_diab[pred_prob_diab >= 0.5 ] = 1   ## Cutoff = 0.5
pd.crosstab(y_test, pred_class_diab)
accuracy_score(y_test, pred_class_diab)  ## 79.6 % 
## TPR - 20/33 = 60.6%
## FPR - 11/85 - 12.9%

## REDUCING THE THRESHOLD TO INCREASE TPR
pred_prob_diab = pred_prob[:,1] # extracting probability of Class 1
pred_class_diab = np.zeros(len(y_test))
pred_class_diab[pred_prob_diab >= 0.3 ] = 1   ## Cutoff = 0.5
pd.crosstab(y_test, pred_class_diab)
accuracy_score(y_test, pred_class_diab)  ## 70.3 % 
##  TPR - 29/33 - 87.87 % 
##  FPR - 31/85 - 36.47 %

## ROC - receiver operation characteristic curve (GIVES TPR, FPR for different threshold)

##diab_fpr, diab_tpr, diab_thresholds = roc_curve(y_test, pred_class_diab)
diab_fpr, diab_tpr, diab_thresholds = roc_curve(y_test,pred_prob_diab)
df_roc = pd.DataFrame({'Threshold' : diab_thresholds,
                       'FPR' : diab_fpr,
                       'TPR' : diab_tpr})

plt.plot(diab_fpr,diab_tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")



