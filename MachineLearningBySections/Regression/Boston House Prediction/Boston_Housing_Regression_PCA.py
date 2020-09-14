# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:11:21 2019

@author: evkikum
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:22:42 2019

@author: evkikum
"""

## https://github.com/bhattbhavesh91/boston_housing_prediction/blob/master/Boston_Housing_Regression.ipynb

## As per the below PCA results which are very poor. Need to remove outliers and restest PCA followed by regression. Need to follow the below link for outliers treatments.
## https://www.kaggle.com/prasadperera/the-boston-housing-dataset/data

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

### Correaltion - From the below matrix TAX, RAD are highly correlated and also TAX is more correlated with MEDV than RAD and hence
### dropping TAX.
df_corr = df.corr()
## in the below heatmap it helps quickly to indentify highly correlated variables by just looking for darker colors
sns.heatmap(df_corr, xticklabels=df_corr.columns, yticklabels=df_corr.columns,cmap="RdBu" )

df_stats = df.describe()  ## Scaling is required

df.shape

Q2_CRIM = np.percentile(df["CRIM"], 50)
Q1_CRIM = np.percentile(df["CRIM"], 25)
Q3_CRIM = np.percentile(df["CRIM"], 75)

IQR_CRIM = Q3_CRIM - Q1_CRIM
UWL_CRIM = Q3_CRIM + (1.5 * IQR_CRIM)
LWL_CRIM = Q1_CRIM - (1.5 * IQR_CRIM)
cond = df["CRIM"] < UWL_CRIM

##df = df.loc[cond,"CRIM"]


for k,v in df.items():    
    q1 = v.quantile(.25)
    q3 = v.quantile(.75)
    irq = q3 - q1
    v_col = v[(v <= q1 -1.5*irq) | (v >= q3 + 1.5*irq)]
    perc = np.shape(v_col)[0] * 100.0/np.shape(df)[0]
    print("Column %s outliers = %.2f%%" % (k, perc))

scaler = StandardScaler()
df_scale = scaler.fit_transform(df)



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

df_model = ols("MEDV ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + PTRATIO + B + LSTAT", data = pd.concat([df_train,df_target_train] , axis = 1)).fit()
df_model.summary()   ## Adjusted R Square .71

## AGE is not stastically significant hence ignoring it based on the p value.
df_model = ols("MEDV ~ CRIM + ZN + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + LSTAT", data = pd.concat([df_train,df_target_train] , axis = 1)).fit()
df_model.summary()   ## Adjusted R Square .72


plt.scatter(df_target_test, df_model.predict(df_test))

MAPE(df_target_train, df_model.predict(df_train))  ## Mean - 16.31 %, Median - 11.6 %
MAPE(df_target_test, df_model.predict(df_test))  ## Mean - 17.6 %, Median - 12.55 %





## LINEAR MODEL USING PCA

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

sns.pairplot(df)
df_scale = pd.DataFrame(df_scale, columns = df.columns)
df_scale = df_scale.drop("MEDV", axis = 1)
df_target = df["MEDV"]

df_train,df_test, df_target_train, df_target_test = train_test_split(df_scale,df_target, test_size = 0.3, random_state=1234)

df_pca = PCA().fit(df_train)
df_pca.explained_variance_ratio_
np.cumsum(df_pca.explained_variance_ratio_)

df_pca = PCA(n_components=5).fit(df_train)
df_pca.explained_variance_ratio_
np.cumsum(df_pca.explained_variance_ratio_)

df_projected = pd.DataFrame(df_pca.transform(df_train), columns = ["Dim1","Dim2","Dim3","Dim4","Dim5"])
## There was a bug while adding MEDV column to the projected data frame. Matching happens based on index and there were so many nulls in your merged dataframe. I have fixed that bug (line 168)
df_projected["MEDV"] = df_target_train.reset_index(drop=True)

df_pca_regression = ols("MEDV ~ Dim1 + Dim2 + Dim3 + Dim4 + Dim5", data = df_projected).fit()
df_pca_regression.summary()  ## 66% Adjusted R Square

MAPE(df_projected["MEDV"], df_pca_regression.predict(df_projected))  ## Mean - 16.9, Median - 11.28%

