# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:53:47 2019

@author: evkikum
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
import os
import math

os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\Practice\Linear Regression\homeprices\Multiregression")

df = pd.read_csv("homeprices.csv")
df.dtypes
df.columns = ('area', 'bedrooms', 'age', 'price')


## Fill the null values with median
df["bedrooms"] = df["bedrooms"].fillna(math.floor(df["bedrooms"].median()))

reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']], df["price"])

reg.coef_
reg.intercept_

reg.predict([3000,3,40])