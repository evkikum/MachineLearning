# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 18:02:07 2019

@author: evkikum
"""

## https://github.com/bhattbhavesh91/car_price_prediction/blob/master/Car_Price_Assignment_Upgrad.ipynb

import numpy as np
import pandas as pd
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
np.random.seed(0)
np.set_printoptions(precision=2)
import os

os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\GreenInstitute_Course")

def MAPE(actual, predicted):
    actual_np = np.array(actual)
    predicted_np = np.array(predicted)
    ape = abs(actual_np - predicted_np)* 100/actual_np
    mean_ape = np.mean(ape)
    median_ape = np.median(ape)
    return pd.Series([mean_ape, median_ape], index = ["Mean_APE", "Median_APE"])


def ohc_conversion(input_text):
    if input_text in "ohc":
        return 1
    else:
        return 0

def words2numconv(input_text):
    if input_text == "one":
        return 1
    elif (input_text == "two"):
        return 2
    elif (input_text == "three"):
        return 3
    elif (input_text == "four"):
        return 4
    elif (input_text == "five"):
        return 5
    elif (input_text == "six"):
        return 6
    elif (input_text == "seven"):
        return 7
    elif (input_text == "eight"):
        return 8
    elif (input_text == "nine"):
        return 9
    elif (input_text == "ten"):
        return 10
    elif (input_text == "eleven"):
        return 11
    elif (input_text == "two"):
        return 12
  
def simplify_carname(input_text):   
   input_text = input_text.split(" ")[0]
   
   if  input_text in ('porsche', 'isuzu', 'jaguar', 'alfa-romero', 'chevrolet', 'vw', 'renault','maxda','mercury', 'Nissan', 'toyouta', 'vokswagen', 'porcshce'):
       input_text = 'others'       
   
   return input_text       
   
         
df = pd.read_csv("data/CarPrice_Assignment.csv");
df.info()
df.isnull().sum()

df_stats = df.describe()  ## Based on the mean of all the columns it is confirmed that we need to implement scaling.
df_cor = df.corr()  # Based on the results it is understood that there is lot of multicollinearity exists among columns.


df["CarName"]           = df["CarName"].apply(simplify_carname)

df["CarName"].value_counts()
df["fuelsystem"].value_counts()
len(df["CarName"].unique())

df["fueltype"]          = df['fueltype'].astype('category').cat.codes
df["aspiration"]        = df['aspiration'].astype('category').cat.codes
df["enginelocation"]    = df["enginelocation"].astype("category").cat.codes

df["doornumber"]        = df["doornumber"].apply(words2numconv)
df["enginetype"]        = df["enginetype"].apply(ohc_conversion)
df["cylindernumber"]    = df["cylindernumber"].apply(words2numconv)

df["fuelsystem"].value_counts()
df = pd.get_dummies(df, columns = ["CarName","carbody","drivewheel", "fuelsystem" ])


Y = df["price"].values
df = df.drop("price", axis = 1)
X = df.values

lr = LinearRegression(normalize=True)
lr.fit(X, Y)

        



