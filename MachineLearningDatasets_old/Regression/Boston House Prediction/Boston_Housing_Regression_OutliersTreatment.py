# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 18:56:00 2019


## OULIERS TREATMENTS IS EXPLAINED IN THE BELOW URL
## https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba


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
df.isnull().sum()  ## NO null values
df.info()
df_corr = df.corr()
df_stats = df.describe()
df_prev = df


for k in df.columns:
    plt.figure()
    df.boxplot(column = k)


## REMOVAL OF OUTLIERS
for k,v in df.items():
    q1 = df[k].quantile(.25)
    q3 = df[k].quantile(.75)
    IQR = q3 - q1
    UWL = q3 + (1.5 * IQR)
    LWL = q1 - (1.5 * IQR)
    df = df[(df[k] <= UWL) & (df[k] >= LWL)]
    print('k, ', k,
          'dimension ', df.shape)

df_stats = df.describe()
df_scale = scale(df)

df_corr = df.corr()


##IDENTIFICATION OF CATEGORICAL VALUES

df["CRIM"].value_counts()
df["ZN"].value_counts()
df["INDUS"].value_counts()
df["CHAS"].value_counts()       ## CATEGORICAL VARIABLE
df["NOX"].value_counts()
df["RM"].value_counts()
df["AGE"].value_counts()
df["DIS"].value_counts()
df["RAD"].value_counts()        ## CATEGORICAL VARIABLE
df["TAX"].value_counts()
df["PTRATIO"].value_counts()
df["B"].value_counts()
df["LSTAT"].value_counts()
df["MEDV"].value_counts()

df_scale = pd.DataFrame(df_scale, columns = df.columns)
df_scale = df_scale.drop("MEDV", axis = 1)
df_target = df["MEDV"]
df.info()

df_train,df_test, df_target_train, df_target_test = train_test_split(df_scale,df_target, test_size = 0.3, random_state=1234)

df_model = ols("MEDV ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT", data = pd.concat([df_train,df_target_train] , axis = 1)).fit()
df_model.summary()   ## 


## REMOVE RAD
df_model = ols("MEDV ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + TAX + PTRATIO + B + LSTAT", data = pd.concat([df_train,df_target_train] , axis = 1)).fit()
df_model.summary()   ## Adjusted R Square 0.082


## REMOVE PTRATIO
df_model = ols("MEDV ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + TAX + B + LSTAT", data = pd.concat([df_train,df_target_train] , axis = 1)).fit()
df_model.summary()   ## Adjusted R Square 0.098

## REMOVE RM
df_model = ols("MEDV ~ CRIM + ZN + INDUS + CHAS + NOX + AGE + DIS + TAX + B + LSTAT", data = pd.concat([df_train,df_target_train] , axis = 1)).fit()
df_model.summary()   ## Adjusted R Square 0.114

## REMOVE AGE
df_model = ols("MEDV ~ CRIM + ZN + INDUS + CHAS + NOX + DIS + TAX + B + LSTAT", data = pd.concat([df_train,df_target_train] , axis = 1)).fit()
df_model.summary()   ## Adjusted R Square 0.128


## remove CHAS
df_model = ols("MEDV ~ CRIM + ZN + INDUS + NOX + DIS + TAX + B + LSTAT", data = pd.concat([df_train,df_target_train] , axis = 1)).fit()
df_model.summary()   ## Adjusted R Square 0.128

## REMOVE ZN
df_model = ols("MEDV ~ CRIM + INDUS + NOX + DIS + TAX + B + LSTAT", data = pd.concat([df_train,df_target_train] , axis = 1)).fit()
df_model.summary()  ## Adjuestd R Square - 0.131

## REMOVE INDUS
df_model = ols("MEDV ~ CRIM + NOX + DIS + TAX + B + LSTAT", data = pd.concat([df_train,df_target_train] , axis = 1)).fit()
df_model.summary()  ## Adjuestd R Square - 0.131