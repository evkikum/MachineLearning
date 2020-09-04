#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 16:25:34 2020

@author: evkikum


In the below KNN and DecisionTreeClassifier are old approaches and the rest are latest approaches.
The below are most popular and recent;
1) Bagging Technique
   Random forest is most popular 
   Note ==> 
   1) The new data prediction will be majority votes for classification
   2) The new data prediction will be average votes for classification

   This is mainly used when the variance is high (or results are not consistent) then random forest is used.
2) Boosting Technique
    1) (THis reduce the bias error)
    
   Type of boosting
   1) AdaBoost
   2) Gradient Tree Boosting (This is mots populary in Gradient Tree Boosting)
   3) XGBoost

  This is mainly used when the accuracy is very less then boosting is used.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score # latest version of sklearn
##from sklearn.cross_validation import train_test_split, cross_val_score # older version of sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
import os

os.chdir(r"/home/evkikum/Desktop/Data Science/Python/Practice/DataSets/iris")

irisdata = pd.read_csv("iris.csv")
irisdata.info()
irisdata["Species"].value_counts()

##################################################
########## KNeighborsClassifier###################
##################################################

iris_class_summary = irisdata.groupby("Species").agg(np.mean)

for i in iris_class_summary.columns:
    plt.figure()
    iris_class_summary[i].plot.bar()
   
iris_class_summary2 = irisdata.groupby("Species").agg([min, max])

for i in irisdata.columns[:4]:
    irisdata.boxplot(column = i, by = "Species")
    
sns.lmplot("Petal.Length","Petal.Width", data = irisdata, hue = "Species", fit_reg = False)
sns.lmplot("Sepal.Length","Sepal.Width", data = irisdata, hue = "Species", fit_reg = False)

irisdata["Species"].value_counts()


X_iris_train, X_iris_test , y_iris_train, y_iris_test = train_test_split(irisdata.iloc[:, :4], irisdata["Species"], test_size = 0.3, 
                                                                         random_state = 1234)

iris_knn1 = KNeighborsClassifier(n_neighbors = 1).fit(X_iris_train, y_iris_train)
iris_knn3 = KNeighborsClassifier(n_neighbors = 3).fit(X_iris_train, y_iris_train)

iris_pred_tr = iris_knn1.predict(X_iris_train)
pd.crosstab(y_iris_train, iris_pred_tr)
accuracy_score(y_iris_train, iris_pred_tr)

iris_pred_tr = iris_knn3.predict(X_iris_train)
pd.crosstab(y_iris_train, iris_pred_tr)
accuracy_score(y_iris_train, iris_pred_tr)

knn_accuracy_score = np.mean(cross_val_score(KNeighborsClassifier(n_neighbors = 1), irisdata.iloc[:,:4], irisdata["Species"], cv = 5))   ## 96 %

for k in range(1,11):
    print("K = ", k,
          "Accuracy = ",
          np.mean(cross_val_score(KNeighborsClassifier(n_neighbors = k), irisdata.iloc[:,:4], irisdata["Species"], cv = 5))
     )

## k = 6 is optimal
    
iris_knn6 = KNeighborsClassifier(n_neighbors = 6).fit(X_iris_train, y_iris_train)
iris_pred_tr = iris_knn6.predict(X_iris_train)
pd.crosstab(y_iris_train, iris_pred_tr)
accuracy_score(y_iris_train, iris_pred_tr)  ## 96.19

knn_accuracy_score = np.mean(cross_val_score(KNeighborsClassifier(n_neighbors = 6), irisdata.iloc[:,:4], irisdata["Species"], cv = 5))

output_df = pd.DataFrame(index = ["imp_features","Accuracy_Score", "Optimal_Value"])
output_df["knn"] = [np.NaN, knn_accuracy_score, "K Value = 6"]
    
####################################################
######## DecisionTreeClassifier#####################
####################################################


iris_dtree = DecisionTreeClassifier(random_state = 1234).fit(X_iris_train, y_iris_train)
iris_pred_dtee_tr = iris_dtree.predict(X_iris_test)
actual_species  = y_iris_test
pd.crosstab(actual_species, iris_pred_dtee_tr)
accuracy_score(actual_species, iris_pred_dtee_tr)   ## 97 %

dtree_accuracy_score = np.mean(cross_val_score(DecisionTreeClassifier(random_state=1234), irisdata.iloc[:,:4],irisdata["Species"], cv = 5))

## Now lets test for max_depth until 6

iris_crossval_accuracy_diff_depth = pd.Series([0.0]*5, range(1,6,1))

for d_i in range(1,6,1):
    iris_crossval_dtree_anyD = cross_val_score(DecisionTreeClassifier(random_state=1234, max_depth=d_i), 
                                               irisdata.iloc[:,:4],irisdata["Species"], cv = 5)
    iris_crossval_accuracy_diff_depth[d_i] = np.mean(iris_crossval_dtree_anyD)
    
print(iris_crossval_accuracy_diff_depth)  ## depth = 3 is the optimal value


## lets try to remodel using depth - 3
iris_dtree = DecisionTreeClassifier(random_state = 1234, max_depth = 3).fit(X_iris_train, y_iris_train)
imp_features = iris_dtree.feature_importances_
iris_pred_dtee_tr = iris_dtree.predict(X_iris_test)
actual_species  = y_iris_test
pd.crosstab(actual_species, iris_pred_dtee_tr)
accuracy_score(actual_species, iris_pred_dtee_tr)   ## 97.77 %

dtree_accuracy_score = np.mean(cross_val_score(DecisionTreeClassifier(random_state=1234, max_depth = 3), irisdata.iloc[:,:4],irisdata["Species"], cv = 5))

output_df["Decision_Tree"] = [imp_features, dtree_accuracy_score, "OPtimal depth = 3"]


####################################################
######## RandomForestClassifier ####################
####################################################

iris_rf = RandomForestClassifier(random_state = 123).fit(X_iris_train, y_iris_train)
imp_features = iris_rf.feature_importances_   ## Based on this Petal.length is most important feature.
iris_pred_rf_tr = iris_rf.predict(X_iris_test)
actual_species = y_iris_test
pd.crosstab(actual_species, iris_pred_rf_tr)
accuracy_score(actual_species, iris_pred_rf_tr)

rf_accuracy_score = np.mean(cross_val_score(RandomForestClassifier(random_state= 1234),
                                            irisdata.iloc[:,:4], 
                                            irisdata["Species"], cv = 5))

iris_crossval_accuracy_diff_depth = pd.Series([0.0]*5, range(1,6,1))

for d_i in range(1,6,1):
    iris_crossval_rf_anyD = cross_val_score(RandomForestClassifier(random_state=1234, max_depth=d_i), 
                                               irisdata.iloc[:,:4],irisdata["Species"], cv = 5)
    iris_crossval_accuracy_diff_depth[d_i] = np.mean(iris_crossval_rf_anyD)
    
print(iris_crossval_accuracy_diff_depth)  ## Optimal depth = 4

iris_rf = RandomForestClassifier(random_state = 123, max_depth = 4).fit(X_iris_train, y_iris_train)
imp_features = iris_rf.feature_importances_   ## Based on this Petal.length is most important feature.
iris_pred_rf_tr = iris_rf.predict(X_iris_test)
actual_species = y_iris_test
pd.crosstab(actual_species, iris_pred_rf_tr)
accuracy_score(actual_species, iris_pred_rf_tr)

rf_accuracy_score = np.mean(cross_val_score(RandomForestClassifier(random_state= 1234, max_depth = 4),
                                            irisdata.iloc[:,:4], 
                                            irisdata["Species"], cv = 5))

output_df["Random_Forest"] = [imp_features, rf_accuracy_score, "Optimal depth = 4"]

####################################################
######## GradientBoostingClassifier ####################
####################################################

iris_gbm = GradientBoostingClassifier(random_state=123)
iris_gbm.fit(X_iris_train, y_iris_train)
imp_features = iris_gbm.feature_importances_
iris_pred_gbm_tr = iris_gbm.predict(X_iris_test)
actual_species = y_iris_test
pd.crosstab(actual_species, iris_pred_gbm_tr)
accuracy_score(actual_species, iris_pred_gbm_tr)

gbm_accuracy_score = np.mean(cross_val_score(GradientBoostingClassifier(random_state = 1234), 
                                             irisdata.iloc[:,:4],
                                             irisdata["Species"],
                                             cv = 5))

output_df["Gradiant_Boost"] = [imp_features, gbm_accuracy_score, np.NaN]




