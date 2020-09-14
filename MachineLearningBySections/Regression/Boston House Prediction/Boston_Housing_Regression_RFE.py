# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:22:42 2019

@author: evkikum
"""

## https://github.com/bhattbhavesh91/boston_housing_prediction/blob/master/Boston_Housing_Regression.ipynb

import numpy as np
import pandas as pd
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

df_stats = df.describe()  ## Scaling is required
df_scale = scale(df)

df_corr = df.corr()

## identification of categorical variable
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


sns.pairplot(df)
df_scale = pd.DataFrame(df_scale, columns = df.columns)
df_scale = df_scale.drop("MEDV", axis = 1)
df_target = df["MEDV"]

df_train,df_test, df_target_train, df_target_test = train_test_split(df_scale,df_target, test_size = 0.3, random_state=1234)

##X_1 = pd.concat([df_train, pd.DataFrame(df_target_train)])

df_model = ols("MEDV ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT", data = pd.concat([df_train,df_target_train] , axis = 1)).fit()
df_model.summary()   ## Adjusted R Square .72

## INDUS is not stastically significant hence ignoring it based on the p value.
df_model = ols("MEDV ~ CRIM + ZN + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT", data = pd.concat([df_train,df_target_train] , axis = 1)).fit()
df_model.summary()   ## Adjusted R Square .72

## age is not stastically significant hence ignoring it based on the p value.
df_model = ols("MEDV ~ CRIM + ZN + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + LSTAT", data = pd.concat([df_train,df_target_train] , axis = 1)).fit()
df_model.summary()   ## Adjusted R Square .72

MAPE(df_target_train, df_model.predict(df_train))  ## Mean - 16.31 %, Median - 11.6 %
MAPE(df_target_test, df_model.predict(df_test))  ## Mean - 17.6 %, Median - 12.55 %




## LINEAR MODEL WITH FEATURE SELECTION

boston = load_boston()
features = boston["feature_names"]
X = pd.DataFrame(boston["data"], columns = pd.Series(features)) 
Y = pd.DataFrame(boston["target"], columns = ['MEDV']) 

df = pd.concat([X, Y] , axis = 1)
df.isnull().sum()  ## NO null values
df.info()

df_stats = df.describe()  ## Scaling is required
df_scale = scale(df)
df_scale = pd.DataFrame(df_scale, columns = df.columns)
df_scale = df_scale.drop("MEDV", axis = 1)
df_target = df["MEDV"]

df_train,df_test, df_target_train, df_target_test = train_test_split(df_scale,df_target, test_size = 0.3, random_state=1234)

lm = LinearRegression()
rfecv = RFECV(estimator=lm, step=1, cv=5) 
rfecv.fit(df_train, df_target_train)

print('Optimal no of features : {} '.format(rfecv.n_features_))  ## OPTIMAL NO OF FEATURES ARE 12
print("Featured selected {}".format(', '.join(np.array(df.columns)[rfecv.support_].tolist())))

ranked_features, _ = zip(*sorted(zip(df.columns, rfecv.ranking_.tolist()),
                                 key=lambda x: x[1],
                                 reverse=True))
print('Suggested order of feature removal: {}'.format(', '.join(ranked_features)))

# plot number of features vs. scores
sns.set_style("darkgrid")
plt.figure(figsize=(8, 4))
plt.xlabel("Number of features selected")
plt.ylabel("Score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()




     