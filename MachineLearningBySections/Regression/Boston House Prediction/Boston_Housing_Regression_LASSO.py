# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 12:31:23 2019

@author: evkikum
"""

import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame,Series
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline 
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
from scipy import stats
import warnings
from statsmodels.formula.api import ols 
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


def MAPE(actual , predicted):
    actual_np = np.array(actual)
    predicted_np = np.array(predicted)
    ape = abs(actual_np - predicted_np)* 100/actual_np
    ape = ape[np.isfinite(ape)]
    mean_np = np.mean(ape)
    median_np = np.median(ape)
    return pd.Series([mean_np, median_np], index = ["Mean_APE", "Median_APE"])

boston = load_boston()
features = boston["feature_names"]
X = pd.DataFrame(boston["data"], columns = pd.Series(features)) 
Y = pd.DataFrame(boston["target"], columns = ['MEDV']) 

df = pd.concat([X, Y] , axis = 1)
df.isnull().sum()

df_scale = scale(df)

df_scale = pd.DataFrame(df_scale, columns = df.columns)
df_scale = df_scale.drop("MEDV", axis = 1)
df_target = df["MEDV"]

df_train,df_test, df_target_train, df_target_test = train_test_split(df_scale,df_target, test_size = 0.3, random_state=1234)

linreg = LinearRegression()
linreg.fit(df_train,df_target_train)
linreg.coef_
linreg.intercept_
linreg.score(df_train,df_target_train)   ## R2 Adjusted - 73 %
linreg.score(df_test,df_target_test)   ## R2 Adjusted - 73.8 %
df_1 = pd.DataFrame(features, linreg.coef_)


MAPE(df_target_train, linreg.predict(df_train))  ## 16.3 % mean, 11.7 Median
MAPE(df_target_test, linreg.predict(df_test))   ## 17.7 % Mean, 13.17 Median


### Lets try lasso
lasso = Lasso(alpha = .3)
lasso.fit(df_train,df_target_train)
lasso.coef_
lasso.intercept_
lasso.score(df_train, df_target_train)  ## R Square - 69 %
lasso.score(df_test, df_target_test) ## R Square - 74%

df_1 = pd.DataFrame(features, lasso.coef_)

MAPE(df_target_train, lasso.predict(df_train))  ## 16.2 % mean, 10.61 Median
MAPE(df_target_test, lasso.predict(df_test))   ## 18.18 % Mean, 12.53 Median


### Lets try Ridge regression
ridge = Ridge(fit_intercept=True, alpha=.3)
ridge.fit(df_train,df_target_train)
ridge.coef_
ridge.intercept_
ridge.score(df_train,df_target_train)  ## 73 % R Square
ridge.score(df_test,df_target_test) ## 73.8 R Square

df_1 = pd.DataFrame(features, ridge.coef_)

MAPE(df_target_train, ridge.predict(df_train))  ## 16.3 % mean, 11.62 Median
MAPE(df_target_test, ridge.predict(df_test))   ## 17.68 % Mean, 13.16 Median


##Let try elastic net regression
elnet = ElasticNet(fit_intercept=True, alpha=.3)
elnet.fit(df_train,df_target_train)
elnet.coef_
elnet.intercept_
elnet.score(df_train,df_target_train)  ## R Square - 69.2%
elnet.score(df_test,df_target_test)  ## R Square - 74.5

df_1 = pd.DataFrame(features, elnet.coef_)

MAPE(df_target_train, elnet.predict(df_train))  ## 16.1 % mean, 10.83 Median
MAPE(df_target_test, elnet.predict(df_test))   ## 17.82 % Mean, 14.17 Median


## Lets try Stochastic Gradient Descent regression
sgdreg = SGDRegressor(penalty='l2', alpha=0.15, n_iter=200)
sgdreg.fit(df_train,df_target_train)
sgdreg.coef_
sgdreg.intercept_
sgdreg.score(df_train,df_target_train)  ## 70% R Square
sgdreg.score(df_test,df_target_test)  ## R Square - 75.2

df_1 = pd.DataFrame(features, sgdreg.coef_)

MAPE(df_target_train, sgdreg.predict(df_train))  ## 15.77 % mean, 10.64 Median
MAPE(df_target_test, sgdreg.predict(df_test))   ## 17.19 % Mean, 13.32 Median