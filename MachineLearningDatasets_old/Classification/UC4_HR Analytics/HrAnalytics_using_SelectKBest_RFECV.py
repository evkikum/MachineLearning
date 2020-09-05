# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:13:51 2019

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
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
import os

os.chdir(r'C:\Users\evkikum\Documents\Python Scripts\Practice\Classification\UC4_HR Analytics')


df = pd.read_csv('HR_comma_sep.csv')
df.info()
df.isnull().sum()

df_class_summary = df.groupby('left').agg(np.mean)

for k in df_class_summary.columns:
    plt.figure()
    df_class_summary[k].plot.bar()
    
for k in df.columns[:6]:
    df.boxplot(column = k, by = 'left')
    

df_sal = pd.get_dummies(df['salary'], drop_first=True, prefix='salary')
df = pd.concat([df, df_sal], axis = 1)

df_dept = pd.get_dummies(df['Department'], drop_first=True, prefix='dept')
df = pd.concat([df, df_dept], axis = 1)

df = df.drop(['salary', 'Department'], axis = 1)

X_train, X_test, y_train , y_test = train_test_split(df.drop('left', axis = 1), df['left'], test_size = 0.3, random_state = 1234)

df_logit = LogisticRegression().fit(X_train, y_train)
pd.Series(df_logit.coef_[0], index = X_train.columns).sort_values(ascending = True)

pred_class_df = df_logit.predict(X_test)
pd.crosstab(y_test, pred_class_df)
accuracy_score(y_test, pred_class_df)  ## 78.311 %% 


## LETS IMPLEMENT UNIVARIATE FEATIURE SELECTION
select_feature = SelectKBest(chi2, k=5).fit(X_train, y_train)
select_feature_df = pd.DataFrame({'Feature' : X_train.columns,
                                  'Scores' : select_feature.scores_})
select_feature_df = select_feature_df.sort_values(by = 'Scores', ascending = False)

X_train_chi = select_feature.transform(X_train)
X_test_chi = select_feature.transform(X_test)

df_chi_model = LogisticRegression().fit(X_train_chi, y_train)

pred_chi_df = df_chi_model.predict(X_test_chi)
pd.crosstab(y_test, pred_chi_df)
accuracy_score(y_test, pred_chi_df)  ## 77.3 % 

## note ==> WITH K = 5 IN SelectKBest THE ACCURACY WAS REDUCED FROM 78.311 TO 77.3 % WHICH SHOWS THAT SOME OF IMPORTANT FEATUES ARE DROPPED.
## HENCE LET US LOOP THRU AS BELOW TO IDENTITY THE BEST OPTIMAL FEATURES WITH BEST ACCURACY.


for i in range(1,18):
    select_feature = SelectKBest(chi2, k=i).fit(X_train, y_train)
    select_feature_df = pd.DataFrame({'Feature' : X_train.columns,
                                  'Scores' : select_feature.scores_})
    select_feature_df = select_feature_df.sort_values(by = 'Scores', ascending = False)

    X_train_chi = select_feature.transform(X_train)
    X_test_chi = select_feature.transform(X_test)

    df_chi_model = LogisticRegression().fit(X_train_chi, y_train)

    pred_chi_df = df_chi_model.predict(X_test_chi)
    pd.crosstab(y_test, pred_chi_df)
    accuracy_score(y_test, pred_chi_df)   
    print('Value of i = ', i,
          'accuracy is ', accuracy_score(y_test, pred_chi_df))


## FROM THE ABOVE FOR LOOP , THE BEST OPTIMAL VALUES IS WITH FEATURE 13 AND ACCURACY 78.73 %% 

rfe = RFE(estimator=LogisticRegression(), step=1)
rfe = rfe.fit(X_train, y_train)

selected_rfe_featues = pd.DataFrame({'Features': X_train.columns,
                                     'Ranking' : rfe.ranking_})

X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

df_rfe_model = LogisticRegression().fit(X_train_rfe, y_train)
pred_rfe_df = df_rfe_model.predict(X_test_rfe)
pd.crosstab(y_test, pred_rfe_df)
accuracy_score(y_test, pred_rfe_df)  ## 77.8 % and no of features selected are 9.



## RECURIVSE FEATURE ELIMINATION WITH CROSS VALIDATION
rfecv = RFECV(estimator=LogisticRegression(), step=1, cv = 5, scoring='accuracy')
rfecv = rfecv.fit(X_train, y_train)

print('Optimal no of features ', rfecv.n_features_)  ## 13
print('Best features ' , X_train.columns[rfecv.support_])

plt.figure()
plt.xlabel('No of features selected')
plt.ylabel('Cross validation score of number of selected features')
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

X_train_rfecv = rfecv.transform(X_train)
X_test_rfecv = rfecv.transform(X_test)

df_rfecv_model = LogisticRegression().fit(X_train_rfecv, y_train)
pred_rfecv_df = df_rfecv_model.predict(X_test_rfecv)
pd.crosstab(y_test, pred_rfecv_df)
accuracy_score(y_test, pred_rfecv_df)  ## 79.11 % 


## SIMPLE LOGISTIC REGRESSION - ACCURACY - 78.311 %% 
## KMEANS
  ## K = 5  FEATURES  ACCURACY - 77.3 % 
  ## K = 13 FEATURES  ACCURACY - 78.73 %% 
  
##RFECV
  ## SIMPLE RFECV - 77.8 % WITH BEST FEATURES OF 9 FEATURES
  ## RFECV WITH CROSS VALIDATION - 79.11% WITH 13 FEATURES




