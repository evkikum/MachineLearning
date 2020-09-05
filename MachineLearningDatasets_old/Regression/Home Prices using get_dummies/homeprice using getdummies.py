# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:28:41 2019

@author: evkikum
"""

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols # https://www.statsmodels.org/stable/index.html
import os


os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\Practice\Linear Regression\Home Prices using get_dummies")


def MAPE(actual, predicted):
    actual_np = np.array(actual)
    predicted_np = np.array(predicted)
    aps = (actual_np - predicted_np) * 100/actual_np
    mean_aps = np.mean(aps)
    median_aps = np.median(aps)
    return pd.Series([mean_aps, median_aps], index = ['Mean', "Median"])

df  = pd.read_csv('homeprices.csv')

dummies = pd.get_dummies(df['town'])
dummies.info()

dummies = dummies.drop('monroe township', axis = 1)
df = df.drop('town', axis=1)

df = pd.concat([df, dummies], axis = 1)


## USING OLS APPROACH
df.columns = ['area', 'price', 'robinsville', 'west_windsor']
model_ols = ols('price ~ area + robinsville + west_windsor', data = df).fit()
model_ols.summary()
model_ols.params
MAPE(df['price'], model_ols.predict(df)) ## 2.9 % Mean, 47.5% Median



## Using sklearn
X = df.drop('price', axis = 1)
Y = df['price']

model = LinearRegression()
model.fit(X,Y)
model.score(X,Y)

predicted_score = model.predict(X)

MAPE(Y, predicted_score)  ## 2.9 % Mean, 47.5% Median


## Now lets test the model using unknown data 

## 3400 sqr ft in west windsor
model.predict([3400,0,1])  ## 681241.6684584

## 2800 sq ft in robinsville
model.predict([2800,1,0]) # 590775.63964739




## Alternatively we can use OneHotEncoder instread of get_dummies
##Using sklearn OneHotEncoder

df  = pd.read_csv('homeprices.csv')
le = LabelEncoder()
dfle = df
dfle.town = le.fit_transform(dfle.town)
X = dfle[['town', 'area']].values
y=df['price']

ohe = OneHotEncoder(categorical_features=[0])
X = ohe.fit_transform(X).toarray()
X = X[:,1:]

model.fit(X,y)
model.predict([0,1,3400])


