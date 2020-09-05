# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:54:52 2019

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

## Step 0: Business understanding, data preparation, data cleaning
  ## NOTE ==> THE MAIN PURPOSE OF EDA IS DETERMINE OUTLIERS, NULL VALUES AND VARIANCE AND DISCUSS THE SAME WITHE CLIENT.
## Step 1: Know the DVs and IDVs
   # IDVs have to be scaled for algorithms like KNN. SCALING IS DONE ONLY IN KNN AND NOT IN DECISION TREE, RANDOM FOREST,
     ## BOOSTING ALGORITHM
## Step 2: Exploratory analysis
   # Groupby aggregate
   # Boxplot
   # Scatter Plot
   # Class imbalance (NOTE ==> THIS HAPPENS WHEN THE CLASSES AVAILABLE IN DATASET ARE UNVEVEN AND SO THE MODEL WILL BE PULLED TO CLASS WITH MORE DATA)
     # To handle class imbalance, 
       # down sample over represented class - random sampling
       # up sample under represented class - SMOTE
       ## Step 3: Building Model
   # Train Test Split
   # Build model on training data
     # K Nearest Neighbors
     # Decision tree
     # Random Forest
     # Boosting algorithms
     # Logistic Regression
## Step 4: Model Evaluation    
   # Confusion Matrix
   # Accuracy, True Positive Rate, False Positive Rate, ROC, AUC
   # Model fine tuning - Hyperparameter tuning
   # Cross validation
     # k fold cross validation; 5 fold cross validation
## Step 5: Go live and start predicting
     
os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\GreenInstitute_Course")

irisdata = pd.read_csv("data/iris.csv")
irisdata.info()

## Step 1:
# DV: Species (setosa, versicolor, virginica)
# IDV: S.L, S.W, P.L, P.W
# IDVs need not be scaled as they are all measured in cms

iris_stats = irisdata.groupby('Species').describe()

iris_class_summary = irisdata.groupby('Species').agg(np.mean)     

for k in iris_class_summary.columns:
    plt.figure()
    plt.xlabel('Columns ' + k)
    plt.ylabel("Count")
    iris_class_summary[k].plot.bar()

iris_class_summary2 = irisdata.groupby('Species').agg([min, max])

sns.lmplot(x = 'Petal.Length', y = 'Petal.Width', data = irisdata, hue = 'Species' , fit_reg = False)
sns.lmplot(x = 'Sepal.Length', y = 'Sepal.Width', data = irisdata, hue = 'Species' , fit_reg = False)

for i in irisdata.columns[:4]:
    irisdata.boxplot(column = i, by = 'Species')
    
irisdata['Species'].value_counts()  ## HERE THE COUNT IS ALL EQUAL 50/50/50 AND HENCE NO CLASS IMBALANCE.

X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(irisdata.iloc[:,:4], irisdata["Species"], test_size=0.3, random_state = 1234)

# Building the model on training data

# KNN
iris_knn1 = KNeighborsClassifier(n_neighbors = 1).fit(X_iris_train, y_iris_train)
iris_knn2 = KNeighborsClassifier(n_neighbors = 2).fit(X_iris_train,y_iris_train)

## Decision Tree
iris_dtree1 = DecisionTreeClassifier(max_depth=2, random_state=1234).fit(X_iris_train, y_iris_train)
iris_dtree1.feature_importances_

iris_dtree2 = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=1234).fit(X_iris_train, y_iris_train)
iris_dtree2.feature_importances_

iris_dtree3 = DecisionTreeClassifier(max_depth = 2, random_state=1234).fit(X_iris_train, y_iris_train)
iris_dtree3.feature_importances_


##  Random Forest
iris_rf = RandomForestClassifier(n_estimators=5, max_depth=2).fit(X_iris_train, y_iris_train)
iris_rf.feature_importances_


## Gradient Boosting
iris_gbm = GradientBoostingClassifier(n_estimators=20).fit(X_iris_train, y_iris_train)
iris_gbm.feature_importances_

## KNN
iris_pred_tr = iris_knn1.predict(X_iris_train)
pd.crosstab(y_iris_train, iris_pred_tr) ## 100 % accuracy with k = 1

iris_pred_tr2 = iris_knn2.predict(X_iris_train)
pd.crosstab(y_iris_train, iris_pred_tr2) ## 97% on k=2

iris_pred_te = iris_knn1.predict(X_iris_test)
pd.crosstab(y_iris_test, iris_pred_te)  ## 97.7 % accuracy

iris_pred_te2 = iris_knn2.predict(X_iris_test)
pd.crosstab(y_iris_test, iris_pred_te2)  ## 97.7 % accuracy
accuracy_score(y_iris_test, iris_pred_te2)


## DECISION TREE
iris_pred_tree1 = iris_dtree1.predict(X_iris_train)
pd.crosstab(y_iris_train, iris_pred_tree1)
accuracy_score(y_iris_train, iris_pred_tree1)  ## 961.%

iris_pred_dtree2 = iris_dtree2.predict(X_iris_test)
pd.crosstab(y_iris_test, iris_pred_dtree2)
accuracy_score(y_iris_test, iris_pred_dtree2) #95.55

iris_pred_dtree3 = iris_dtree3.predict(X_iris_test)
pd.crosstab(y_iris_test, iris_pred_dtree3)
accuracy_score(y_iris_test, iris_pred_dtree3) #95.5

## Random Forest
iris_pred_rf = iris_rf.predict(X_iris_test)
pd.crosstab(y_iris_test, iris_pred_rf)
accuracy_score(y_iris_test, iris_pred_rf)

## Gradient Boosting
iris_pred_gbm = iris_gbm.predict(X_iris_test)
pd.crosstab(y_iris_test, iris_pred_gbm)
accuracy_score(y_iris_test, iris_pred_gbm)

## CROSS VALIDATION
np.mean(cross_val_score(KNeighborsClassifier(n_neighbors = 1),  # algorithm
                irisdata.iloc[:,:4], # IDV
                irisdata["Species"], # DV
                cv = 5) # number of folds - (cv - 5 means 80:20 split))
                    ) #96% cross validated accuracy

    
## PARAMETER TUNNING
for k in range(1,11):
    print("k = ", k,
          "Accuracy = " ,
          np.mean(cross_val_score(KNeighborsClassifier(n_neighbors = k),
                                  irisdata.iloc[:,:4],
                                  irisdata["Species"],
                                  cv=5)))
 ## K = 6 is optimal parameter

   
    
## KNN Decision Tree
for depthi in range(1,11):
    print("Max depth = ", depthi,
          "Accuracy = ",
          np.mean(cross_val_score(DecisionTreeClassifier(
                  max_depth = depthi, random_state = 1234),  # algorithm
                irisdata.iloc[:,:4], # IDV
                irisdata["Species"], # DV
                cv = 5) # number of folds
                    ))
# Max depth = 4 is optimal
    
## Random Forest
for n_est in range(1,11):
    print("Number of estimators = ", n_est,
          "Accuracy = ",
          np.mean(cross_val_score(RandomForestClassifier(
                  n_estimators = n_est, random_state = 1234),  # algorithm
                irisdata.iloc[:,:4], # IDV
                irisdata["Species"], # DV
                cv = 5) # number of folds
                    ))
# number of estimators = 5 looks to be the optimal parameter
    
## Gradient Boosting
for n_est in range(1,20):
    print("Number of estimators = ", n_est,
          "Accuracy = ",
          np.mean(cross_val_score(GradientBoostingClassifier(
                  n_estimators = n_est, random_state = 1234),  # algorithm
                irisdata.iloc[:,:4], # IDV
                irisdata["Species"], # DV
                cv = 5) # number of folds
                    ))
# number of boosting stages = 4 looks to be the optimal



