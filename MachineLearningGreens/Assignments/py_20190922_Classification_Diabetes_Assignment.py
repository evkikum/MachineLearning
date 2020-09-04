# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:25:20 2019

@author: evkikum
"""
'''
PURPOSE OF EDA
1) OUTLIERS
2) NULL VALUES
3) VARIANCE - RANGE OF VALUES


KNN ony needs scaling, Gradiesnboost, rabdomeforect do not need scaling
When the no of class as imbalanced then we need to downsize the clas on the higher side. Say for example if
no of non-diabetic pateintes are 95% higher than diabetic then we need to downsize the non-diabetic data in par with
diabetic.
np.random.choice()
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.model_selection import train_test_split, cross_val_score # latest version of sklearn
from sklearn.cross_validation import train_test_split, cross_val_score # older version of sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import os

os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\GreenInstitute_Course")


diabetes = pd.read_csv("data/diabetes_data.csv")
diabetes.info()
diabetes.isnull().sum()  ## No nulls

## Step 1:
# DV: Class (1 - diabetic, 0 - non diabetic)
# IDVs: 8 (diagnosis results)


## Based on the below mean values of all the columns it is understood that scaling is required.
diabetes_stats = diabetes.describe()
diabetes_scale = scale(diabetes.iloc[:, :8])
diabetes_scale = pd.DataFrame(diabetes_scale)
diabetes_scale.columns = diabetes.columns[:8]


diabetes_class_summary = diabetes.groupby("Class").agg(np.mean)

for i in diabetes_class_summary.columns:
    plt.figure()
    plt.xlabel("Class")
    plt.ylabel(i)
    diabetes_class_summary[i].plot.bar()
    

for i in diabetes.columns[:8]:
    diabetes.boxplot(column = i, by = "Class")
    
X_train, X_test , y_train, y_test = train_test_split(diabetes_scale,  diabetes.loc[:, "Class"], test_size = 0.3, random_state = 1234)

diabetes_knn1 = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)

diabetes_pred_train = diabetes_knn1.predict(X_train)
pd.crosstab(y_train, diabetes_pred_train)  ## 100 %

diabetes_pred_test = diabetes_knn1.predict(X_test)
pd.crosstab(y_test, diabetes_pred_test)    ## 74.57 %


np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=1), diabetes_scale, diabetes.loc[:, "Class"], cv =5))  ## 69.90
np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=1), X_train, y_train, cv =5))  ## 69.35
np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=1), X_test, y_test, cv =5))  ## 71.2

##Lets try for optimal values

## Using whole data 
for k in range(1,11):
    print("k, ", k,
          "Accuracy ",
          np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=k), diabetes_scale, diabetes.loc[:, "Class"], cv = 5)))
    
## Hence k = 9 is the best as the accuracy is 77.56% based on the above parameter tunning.

## Now using only train data ;
for k in range(1,11):
    print("k, ", k,
          "Accuracy ",
          np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv = 5)))   ## k =8, 75.44 %
    

## Now using only test data ;
for k in range(1,11):
    print("k, ", k,
          "Accuracy ",
          np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=k), X_test, y_test, cv = 5)))   ## k =5, 75.47 %
    