#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 17:53:18 2020

@author: evkikum
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

os.chdir(r"/home/evkikum/Desktop/Data Science/Python/Practice/DataSets/Diabetes")

diabetes_data = pd.read_csv("diabetes_data.csv")
diabetes_data.info()
diabetes_stats =  diabetes_data.describe()

diabetes_summary = diabetes_data.groupby("Class").agg(np.mean)

for i in diabetes_data.columns[:8]:
    plt.figure()
    diabetes_summary[i].plot.bar()
    
for i in diabetes_data.columns[:8]:
    diabetes_data.boxplot(column = i, by = "Class")
    
sns.lmplot("Plasma glucose concentration", "Age", data = diabetes_data, hue = "Class", fit_reg = False)


##################################################
########## LogisticRegression  ###################
##################################################

X_diab_train, X_diab_test, y_diab_train, y_diab_test = train_test_split(
        diabetes_data.iloc[:,:8], diabetes_data["Class"], test_size = 0.3, 
        random_state = 42)

y_diab_train.value_counts()
y_diab_test.value_counts()

diab_logic = LogisticRegression()
diab_logic.fit(X_diab_train, y_diab_train)

pred_class_logit = diab_logic.predict(X_diab_test)
pd.crosstab(y_diab_test, pred_class_logit)
accuracy_score(y_diab_test, pred_class_logit)


