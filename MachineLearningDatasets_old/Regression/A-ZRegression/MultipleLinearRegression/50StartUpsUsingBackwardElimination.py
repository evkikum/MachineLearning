#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 16:37:10 2020

@author: evkikum

## ASSUMPTION OF LINEAR REGRESSION
1) LINEARITY
2) HOMOSCEDASTICITY
3) MULTIVARIATE NORMALITY
4) INDEPENDENT OF ERRORS
5) LACK OF MULTICOLLINEARITY

https://www.dropbox.com/sh/pknk0g9yu4z06u7/AADSTzieYEMfs1HHxKHt9j1ba?dl=0

The below explains on SVR
https://github.com/tomsharp/SVR/blob/master/SVR.ipynb
https://towardsdatascience.com/an-introduction-to-support-vector-regression-svr-a3ebc1672c2
https://www.saedsayad.com/support_vector_machine_reg.htm
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

os.chdir(r"/home/evkikum/Desktop/Datascience/Python/MachineLearning/MachineLearningA-Z/Part 2 - Regression/Section 5 - Multiple Linear Regression/Python")

dataset = pd.read_csv('50_Startups.csv')
dataset.info()
dataset["State"].value_counts()
y = dataset.iloc[:,-1]

dataset = dataset.drop("Profit", axis = 1)
dataset["State"] = dataset["State"].astype("category")
##ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
##X = np.array(ct.fit_transform(X))
dataset = pd.get_dummies(dataset)

## AVOID DUMMY VARIABLE TRAP - WE SHOULD THE ELEMENT OF DUMMY VARIABLE AS PART PROCESS
dataset = dataset.iloc[:, :-1]

X = dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
regressor.score(X_train, y_train)  ## 95 %
regressor.score(X_test, y_test)  ## 93 %

y_pred = regressor.predict(X_test)


## Building the optimal model using backward elimination
import statsmodels.api as sm

X = sm.add_constant(X)
X_opt = X.iloc[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
regressor_OLS.pvalues


cols = list(X.columns)
select_feat = []

while (len(cols) > 0):
    p = []
    X1=X[cols]
    model = sm.OLS(y,X1).fit()
    p = pd.Series(model.pvalues, index = X1.columns)
    pmax = max(p)
    if (pmax > 0.05):
        feature_with_p_max = p.idxmax()
        print("feature_with_p_max ", feature_with_p_max)
        cols.remove(feature_with_p_max)
    else:
        break
            
select_feat=cols
select_feat
        
X1 = X1.drop("const", axis = 1)
regressor = LinearRegression()
regressor.fit(X1, y)
regressor.score(X1, y)  ## 94%%

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X1, y)
regressor.score(X1, y)  ## 99%%


## This need scaling else it goes work well.
regressor = SVR()
regressor.fit(X1, y)
regressor.score(X1, y)  ## 99%%


