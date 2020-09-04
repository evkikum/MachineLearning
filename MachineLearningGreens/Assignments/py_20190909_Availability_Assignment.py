# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 00:17:22 2019

@author: evkikum
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols # https://www.statsmodels.org/stable/index.html
from sklearn.linear_model import LinearRegression # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
#from sklearn.model_selection import train_test_split # latest version of sklearn
from sklearn.cross_validation import train_test_split # older version of sklearn
import os


##Below is sample Multiple Non Linear Regression
##https://www.kaggle.com/alokevil/non-linear-regression/comments


def MAPE(actual, predicted):
    actual_np = np.array(actual)
    predicted_np = np.array(predicted)
    ape = abs(actual_np - predicted_np)/actual_np
    mean_ape = np.mean(ape)
    median_ape = np.median(ape)
    return (pd.Series([mean_ape, median_ape], index = ["Mean APE", "Median APE"]))
    
    
os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\GreenInstitute_Course")

df = pd.read_csv("data/Availability.csv")
df.info()
df.isnull().sum()


####1. What is the correlation between Bid Price and Availability?
df["Bid"].corr(df["Availability"])  ### .62

####2. Is there is multi-collinearity between Spot Price and Availability?
df_corr = df.corr()  ## No collinearity

####3. Can regression techniques help predict Availability

## value_counts is python equivalent to table in R
df["Bid"].value_counts()
df["Spotprice"].value_counts()


plt.scatter(df["Bid"], df["Availability"])
plt.scatter(df["Spotprice"], df["Availability"])

sns.lmplot("Bid", "Availability", data = df, fit_reg = False, hue = "Spotprice")


df["Bid"].corr(df["Availability"])
df["Spotprice"].corr(df["Availability"])

df["Bid_sq"] = df["Bid"]**2
df["Bid_log"] = np.log(df["Bid"])
df["Spotprice_sq"] = df["Spotprice"]**2
df["Spotprice_log"] = np.log(df["Spotprice"])
df["Availability_log"] = np.log(df["Availability"])

## RSqhare 
## Bid_sq + Bid + Spotprice - 78.5 %
## Bid_sq + Spotprice ==> 53.6
## Bid + Spotprice ==> 61.3
## Bid_log + Spotprice ==> 67.7 %
df_multi_lin = ols(formula = "Availability ~ Bid_sq + Bid + Spotprice", data = df).fit()
df_multi_lin.summary()

fitted_avail = df_multi_lin.predict(df)
##fitted_avail = np.exp(fitted_avail_log)


## Mape values
## Bid_sq + Bid + Spotprice ==> 17 % Median
## Bid_sq + Spotprice  ==> 30 % Median
## Bid + Spotprice ==> 27.7%
## Bid_log + Spotprice ==> 24 %
MAPE(df["Availability"], fitted_avail)


plt.scatter(df["Bid"], df["Availability"])
plt.scatter(df["Bid"], fitted_avail, c = "red")

####4. What features will be needed for building models?




