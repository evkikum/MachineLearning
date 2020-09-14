#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 16:17:36 2020

@author: evkikum
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

os.chdir(r"/home/evkikum/Desktop/Datascience/Python/MachineLearning/MachineLearningA-Z/Part 2 - Regression/Section 4 - Simple Linear Regression/Python")

dataset = pd.read_csv('Salary_Data.csv')
dataset.info()
dataset_stats = dataset.describe()
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experence (Training set)")
plt.xlabel("Years of exp")
plt.ylabel("Salary")
plt.show()


## Visualizing the test results
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_test, y_pred, color = "blue")
plt.title("Salary vs exp (Test set)")
plt.xlabel("Year of exp")
plt.ylabel("Salary")
plt.show()
