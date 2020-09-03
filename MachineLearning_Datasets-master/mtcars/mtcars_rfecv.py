#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:10:49 2020

@author: evkikum
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
##from pydataset import data
import os
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

os.chdir(r"/home/evkikum/Desktop/Data Science/Python/Practice/mtcars")

mtcars = pd.read_csv("mtcars.csv")
mtcars.info()
mtcars["cyl"].value_counts()   ## Categorical
mtcars["vs"].value_counts()   ## vs
mtcars["am"].value_counts()   ## am
mtcars["gear"].value_counts()   ## gear
mtcars["carb"].value_counts()   ## carb



## drat


mtcars.describe(include = "all")
mtcars_summary = mtcars.describe(include = "all")

y = mtcars["mpg"].values
X = mtcars.drop("mpg", axis = 1).values

feature_names = mtcars.drop('mpg',1).columns

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.7, random_state = 1234)

lm = LinearRegression()
lm.fit(X_train, y_train)

param_df = pd.DataFrame({
                         "Coefficient" : [lm.intercept_] + list(lm.coef_),
                         "Feature" : ['intercept'] + list(feature_names) })


r_squared = r2_score(y_train, lm.predict(X_train))
predicted = lm.predict(X_test)


mean_absolute_error(pd.DataFrame(y_test), pd.DataFrame(predicted))
sqrt(mean_squared_error(pd.DataFrame(y_test), pd.DataFrame(predicted)))

mae = np.mean(abs(predicted - y_test))  
rmse = np.sqrt(np.mean((predicted - y_test)**2))    ## Root Mean Squared Error 	
rae = np.mean(abs(predicted - y_test)) / np.mean(abs(y_test - np.mean(y_test)))     ## Relative Absolute Error
rse = np.mean((predicted - y_test)**2) / np.mean((y_test - np.mean(y_test))**2)     ## Relative Squared Error

summary_df = pd.DataFrame(index = ['R-squared', 'Mean Absolute Error', 'Root Mean Squared Error',
                                   'Relative Absolute Error', 'Relative Squared Error'])
summary_df['Linear Regression, all variables'] = [r_squared, mae, rmse, rae, rse]
summary_df



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

lm = LinearRegression()

## IN THE BELOW STEP = 1 MEANS THAT AT EACH ITERATION ONLY ONE FEATURE CAN BE REMOVED. CV STANDS FOR CROSS VALIDATION.

rfecv = RFECV(estimator=lm, step = 1, cv = 5)
rfecv.fit(X_scaled, y_train)

print('Optimal number of features: {}'.format(rfecv.n_features_))
# save the selected features
print('Features selected: {}'.format(', '.join(np.array(feature_names)[rfecv.support_].tolist())))

# get the feature elimination order
ranked_features, _ = zip(*sorted(zip(feature_names, rfecv.ranking_.tolist()),
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

