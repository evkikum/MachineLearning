#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 15:55:34 2020

@author: evkikum
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

os.chdir(r"/home/evkikum/Desktop/Datascience/Python/MachineLearning/MachineLearningDatasets_old/Regression/A-ZRegression/PloynomialRegression")

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X.info()

X = X.drop("Position", axis = 1)


## Modeling using Linear Regression
model = LinearRegression()
model.fit(X, y)
model.score(X, y)   ## 66%

y_lin_pred = model.predict(X)

## Now use ploynomial Linear regression
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)
model2 = LinearRegression()
model2.fit(X_poly, y)
model2.score(X_poly, y)   ## 99%


## Now visualize the data for linear regression
plt.scatter(X["Level"], y, color = "red")
plt.plot(X["Level"], y_lin_pred, color = "blue")
plt.title("Salary info")
plt.xlabel("Lavel")
plt.ylabel("Salary")
plt.show()

## Now visualize using Polynomial regression
plt.scatter(X["Level"], y, color = "red")
plt.plot(X_poly[:,1], model2.predict(X_poly), color = "blue")
plt.title("Salary info")
plt.xlabel("Lavel")
plt.ylabel("Salary")
plt.show()  