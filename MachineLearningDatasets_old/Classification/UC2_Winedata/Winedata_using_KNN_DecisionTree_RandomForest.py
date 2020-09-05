# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:03:06 2019

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

##KNN
wine_knn1 = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
wine_knn2 = KNeighborsClassifier(n_neighbors=2).fit(X_train, y_train)


## Decision Tree
wine_dtree1 = DecisionTreeClassifier(max_depth=2, random_state=1234).fit(X_train, y_train)
wine_dtree1.feature_importances_
pd.Series(wine_dtree1.feature_importances_, X_train.columns).sort_values(ascending = False )

wine_dtree2 = DecisionTreeClassifier(criterion = 'entropy',max_depth=2, random_state=1234).fit(X_train, y_train)
wine_dtree2.feature_importances_
pd.Series(wine_dtree2.feature_importances_, index = X_train.columns).sort_values(ascending=False)

wine_dtree3 = DecisionTreeClassifier(max_depth = 3, random_state = 1234).fit(X_train, y_train)
wine_dtree3.feature_importances_
pd.Series(wine_dtree3.feature_importances_, index = X_train.columns).sort_values(ascending=False)


## RANDOM FOREST

wine_rf = RandomForestClassifier(n_estimators = 5, max_depth = 2).fit(X_train, y_train)
wine_rf.feature_importances_
pd.Series(wine_rf.feature_importances_, index = X_train.columns).sort_values(ascending=False)

## Gradient Boosting
wine_gbm = GradientBoostingClassifier(n_estimators = 20).fit(X_train, y_train)
wine_gbm.feature_importances_
pd.Series(wine_gbm.feature_importances_, index = X_train.columns).sort_values(ascending=False)


## KNN
wine_pred_te = wine_knn1.predict(X_test)
pd.crosstab(y_test, wine_pred_te)
accuracy_score(y_test, wine_pred_te) ## 77.77 % 

wine_pred_te2 = wine_knn2.predict(X_test)
pd.crosstab(y_test, wine_pred_te2)
accuracy_score(y_test, wine_pred_te2) ## 66.6 % 

#Decision Tree
wine_pred_dtree = wine_dtree1.predict(X_test)
pd.crosstab(y_test, wine_pred_dtree)
accuracy_score(y_test, wine_pred_dtree)  ## 81.4 % 

wine_pred_dtree2 = wine_dtree2.predict(X_test)
pd.crosstab(y_test, wine_pred_dtree2)
accuracy_score(y_test, wine_pred_dtree2)  ## 90.7 % 

wine_pred_dtree3 = wine_dtree3.predict(X_test)
pd.crosstab(y_test, wine_pred_dtree3)
accuracy_score(y_test, wine_pred_dtree3)  ## 87.03 % 

## RANDOM FOREST

wine_pred_rf = wine_rf.predict(X_test)
pd.crosstab(y_test, wine_pred_rf)
accuracy_score(y_test, wine_pred_rf)  ## 88.88 % 

## Gradient Boosting
wine_pred_gbm = wine_gbm.predict(X_test)
pd.crosstab(y_test, wine_pred_gbm)
accuracy_score(y_test, wine_pred_gbm)  ## 88.88 % 



## PARAMETER TUNNING

for k in range(1,20):
    print('k = ', k,
          'Accuracy ',
          np.mean(cross_val_score(KNeighborsClassifier(n_neighbors= k),
                                  winedata.drop('Wine_Class', axis = 1),
                                  winedata['Wine_Class'],
                                  cv = 5)))         ## k = 1 is optimal with accuracy - 72.5 % 
    

    
    
## DECISION TREE

for k in range(1,20):
    print('k = ', k,
          'Accuracy ',
          np.mean(cross_val_score(DecisionTreeClassifier(max_depth = k, random_state = 1234),
                                  winedata.drop('Wine_Class', axis = 1),
                                  winedata['Wine_Class'],
                                  cv = 5)))  ## k = 4  is the optimal value with 91.6  accuracy
    
## RANDOM FOREST
pred_ser_rf = pd.Series([0.0]*19, index = range(1,20))
for k in range(1,20):
    print('k ', k,
          'Accuracy ',
          np.mean(cross_val_score(RandomForestClassifier(n_estimators = k, random_state=1234),
                                  winedata.drop('Wine_Class', axis = 1),
                                  winedata['Wine_Class'],
                                  cv = 5)))  ## k (No of estimators) = 14 is the optimal with 98.34 % 

## Gradient Boosting
for k in range(1,20):
    print(' k ', k,
          'Accuracy ',
          np.mean(cross_val_score(GradientBoostingClassifier(n_estimators = k),
                                  winedata.drop('Wine_Class', axis = 1),
                                  winedata['Wine_Class'],
                                  cv = 5 )))  ## k = 6 is optimal no of esimator with 90% accuracy
           