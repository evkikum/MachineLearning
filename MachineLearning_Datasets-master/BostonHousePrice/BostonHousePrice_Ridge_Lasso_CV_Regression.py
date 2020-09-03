#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 05:40:11 2020

@author: evkikum


The below code is extracted from 
https://www.youtube.com/watch?v=0yI0-r3Ly40&list=PLZoTAELRMXVPBTrWtJkn3wWQxZkmTXGwe&index=30

Mean square error should be near to zero to become the best model.
"""

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 12}) 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import seaborn as sns

df=load_boston()
dataset = pd.DataFrame(df.data)
dataset.head()
dataset.columns = df.feature_names
dataset['Price'] = df.target
dataset.head()

dataset.info()
dataset_stats_summary = dataset.describe()


X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

lin_regressor = LinearRegression()
## Using cross validation of 5 iterations will results in 5 values which later needs to be averaged.
mse = cross_val_score(lin_regressor, X, y, scoring = 'neg_mean_squared_error', cv = 5)
mean_mse = np.mean(mse)   ## The more mse is nearning zero then the model works fine.

### LAMBDA VALUES IS SELECTED USING CROSS VALIDATION


## USAGE OF RIDGE REGRESSION
ridge = Ridge()
parameters = {"alpha": [1e-15, 1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor = GridSearchCV(ridge, parameters,scoring="neg_mean_squared_error", cv=5)
ridge_regressor.fit(X,y)

ridge_regressor.best_params_
ridge_regressor.best_score_


## LASSO REGRESSION
lasso = Lasso()
parameters = {"alpha": [1e-15, 1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor = GridSearchCV(lasso, parameters,scoring="neg_mean_squared_error", cv=5)
lasso_regressor.fit(X,y)

lasso_regressor.best_params_
lasso_regressor.best_score_


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

prediction_lasso = lasso_regressor.predict(X_test)
prediction_ridge = ridge_regressor.predict(X_test)


sns.distplot(y_test-prediction_lasso)
sns.distplot(y_test-prediction_ridge)
