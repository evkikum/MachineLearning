# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 14:02:10 2019

@author: evkikum
"""

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

##KNN mean K nearest neighbors

winedata = pd.read_csv("data/wine.data")

winedata.columns = ["Wine_Class", "Alcohol","Malic_acid","Ash","Alcalinity_of_ash",
                       "Magnesium","Total_phenols","Flavanoids","Nonflavanoid_phenols",
                       "Proanthocyanins","Color_intensity","Hue","OD280_OD315",
                       "Proline"]

winedata.info()
winedata.isnull().sum()

## Based on the below means of all the columns it is understood that we need scaling.
windata_stat = winedata.describe()
winedata_scale = scale(winedata.iloc[:, 1:])
winedata_scale = pd.DataFrame(winedata_scale)
winedata_scale.columns = winedata.columns[1:]



## Step 1:
# DV: Wine Class (1, 2, 3)
# IDV: 13 attributes
# IDVs need scaling while using KNN

winedata_class_summary = winedata.groupby("Wine_Class").agg(np.mean)
winedata_class_summary2 = winedata.groupby("Wine_Class").agg([min, max])

for i in winedata_class_summary.columns:
    plt.figure()
    plt.title("Winclass Category")
    plt.xlabel("Wineclass ")
    plt.ylabel(i)
    winedata_class_summary[i].plot.bar()
    plt.grid()
    
for i in winedata.columns[1:]:
    winedata.boxplot(column = i, by = "Wine_Class")

X_train, X_test, y_train, y_test = train_test_split(winedata_scale, winedata.loc[:, "Wine_Class"], test_size = 0.3, random_state = 1234)

winedata_knn1 = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)

winedata_pred_train = winedata_knn1.predict(X_train)
pd.crosstab(y_train, winedata_pred_train)  ## 100

winedata_pred_test = winedata_knn1.predict(X_test)
pd.crosstab(y_test, winedata_pred_test)    ## 94.4%%

np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=1), winedata_scale, winedata.loc[:, "Wine_Class"], cv = 5))  ## 94.9 %
np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=1), X_train, y_train, cv = 5))  ## 95.89 %
np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=1), X_test, y_test, cv = 5))  ## 90.29 %


## PARAMETER TUNNING

## Using full data
knn_values = pd.Series(0.0, index = range(1,11))

for k in range(1,11):
    print("k , ", k,
          "Accuracy ",
          np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=k), winedata_scale, winedata.loc[:,"Wine_Class"], cv = 5)))
    knn_values[k] = np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=k), winedata_scale, winedata.loc[:,"Wine_Class"], cv = 5))


plt.xlabel("K Value")
plt.ylabel("KNN Value")
plt.title("KNN Variance")
plt.plot(range(1,11), knn_values)

## Hence k = 8 is the best as the accuracy is 96.69% based on the above parameter tunning.

##Using only train data
    
knn_values = pd.Series(0.0, index = range(1,11))

for k in range(1,11):
    print("k , ", k,
          "Accuracy ",
          np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv = 5)))
    knn_values[k] = np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv = 5))
    
## Hence k = 10 is the best as the accuracy is 97.63% based on the above parameter tunning.


##Using only test data
    
knn_values = pd.Series(0.0, index = range(1,11))

for k in range(1,11):
    print("k , ", k,
          "Accuracy ",
          np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=k), X_test, y_test, cv = 5)))
    knn_values[k] = np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=k), X_test, y_test, cv = 5))
    
## Hence k = 8 is the best as the accuracy is 92.11% based on the above parameter tunning.
