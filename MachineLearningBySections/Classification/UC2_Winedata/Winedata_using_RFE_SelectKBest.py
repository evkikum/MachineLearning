# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 22:34:07 2019

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
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import scale
import os

os.chdir(r'C:\Users\evkikum\Documents\Python Scripts\GreenInstitute_Course')

winedata = pd.read_csv("data/wine.data", header = None)
winedata.columns = ["Wine_Class", "Alcohol","Malic_acid","Ash","Alcalinity_of_ash",
                       "Magnesium","Total_phenols","Flavanoids","Nonflavanoid_phenols",
                       "Proanthocyanins","Color_intensity","Hue","OD280_OD315",
                       "Proline"]
                       
winedata_stats = winedata.describe()
winedata.info()

wine_class_summary = winedata.groupby('Wine_Class').agg(np.mean)

for k in wine_class_summary.columns:
    plt.figure()
    wine_class_summary[k].plot.bar()

for k in winedata.columns[1:]:
    winedata.boxplot(column = k, by = 'Wine_Class')
    
X_train, X_test, y_train, y_test = train_test_split(winedata.drop('Wine_Class', axis = 1), winedata['Wine_Class'], test_size = 0.3, random_state=1234)
n_columns = winedata.drop('Wine_Class', axis = 1).columns

X_train = pd.DataFrame(scale(X_train), columns = n_columns) 
X_test = pd.DataFrame(scale(X_test), columns = n_columns) 

df_logit = LogisticRegression().fit(X_train, y_train)
pd.Series(df_logit.coef_[0], index = X_train.columns).sort_values(ascending=False)

pred_class_df = df_logit.predict(X_test)
pd.crosstab(y_test, pred_class_df)
accuracy_score(y_test, pred_class_df)  ## 94.44 % 

test_data_cmpr = np.column_stack([y_test, pred_class_df])



## LET IMPLEMENT UNIVARATE FEATURE USING KBEST
## LETS LOOP THRU TO INDENITY THE OPTIMAL K VALUE. HERE K REPRESEnts no oe best features
## IN SelectKBest scaling is not needed

X_train, X_test, y_train, y_test = train_test_split(winedata.drop('Wine_Class', axis = 1), winedata['Wine_Class'], test_size = 0.3, random_state=1234)

for i in range(1,14):
    select_features = SelectKBest(chi2, k=i).fit(X_train, y_train)
        
    X_train_chi = select_features.transform(X_train)
    X_test_chi = select_features.transform(X_test)
    
    df_chi_model = LogisticRegression().fit(X_train_chi, y_train)
    pred_chi_df = df_chi_model.predict(X_test_chi)
    accuracy_score(y_test, pred_chi_df)
    print('i ', i,
         'accuracy ', accuracy_score(y_test, pred_chi_df) )

## THE OPTIMAL VALUES OF K IS 6 WITH 94.44 % ACCURACY

X_train, X_test, y_train, y_test = train_test_split(winedata.drop('Wine_Class', axis = 1), winedata['Wine_Class'], test_size = 0.3, random_state=1234)
n_columns = winedata.drop('Wine_Class', axis = 1).columns

X_train = pd.DataFrame(scale(X_train), columns = n_columns) 
X_test = pd.DataFrame(scale(X_test), columns = n_columns) 


rfe = RFE(estimator = LogisticRegression(), step=1)
rfe = rfe.fit(X_train, y_train)

selecvt_rfe_features =  pd.DataFrame({'Features' : X_train.columns,
                                      'Ranking' : rfe.ranking_})

X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

df_rfe_model = LogisticRegression().fit(X_train_rfe, y_train)
pred_rfe_df = df_rfe_model.predict(X_test_rfe)
pd.crosstab(y_test, pred_rfe_df)
accuracy_score(y_test, pred_rfe_df)  ## 96.29 %  the no of features are only 6.

## RECURSIVE FEATURE ELIMINATION WITH CROSS VALIDATION

rfecv = RFECV(estimator=LogisticRegression(), step=1, cv = 5, scoring='accuracy')
rfecv = rfecv.fit(X_train, y_train)

print('Optimal no of features ', rfecv.n_features_)  ## 13
print('Best features ', X_train.columns[rfecv.support_])
print('Ranking ', rfecv.ranking_)
select_rfecv_features = pd.DataFrame({'Features' : X_train.columns,
                                      'Ranking' : rfecv.ranking_})
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
accuracy_score(y_test, pred_rfecv_df)  ## 100  % 
 
