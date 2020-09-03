#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 14:17:56 2020

@author: evkikum
"""

import os
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt

# Working directory
os.chdir(r"/home/evkikum/Desktop/Data Science/Python/Udemy_Shiv")
exec(open(os.path.abspath('CommonUtils.py')).read())

seed_value = 123; np.random.seed(seed_value)
catColumns = ["GENDER"]
strResponse = "HWT"

df = pd.read_csv("data/cats.csv")
df.columns = map(str.upper, df.columns)
df.info()
df.shape
df.index


catColumns = np.intersect1d(df.columns, catColumns)
df[catColumns] = df[catColumns].apply(lambda x : x.astype('category'))

print(df.describe(include = "all"))

plt.hist(df[strResponse])
plt.show()

df[strResponse].plot(kind="kde")
plt.show()


import statsmodels.api as sm

sm.qqplot(df[strResponse], line = 's')
plt.show()


from scipy.stats import shapiro

stat, p = shapiro(df[strResponse])
print('Statistics=%.2f, p=%.2f' % (stat, p))

alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (Normal) (accept H0)')
else:
	print('Sample does not look Gaussian (Normal) (reject H0)')
    
    
from scipy import stats
from scipy.special import inv_boxcox

x = df[strResponse]
xt, bx_lambda = stats.boxcox(df[strResponse])


# First plot, without any transformation
fig = plt.figure()
ax1 = fig.add_subplot(211)
prob = stats.probplot(x, dist=stats.norm, plot=ax1)
ax1.set_xlabel(''); ax1.set_title('Probplot against normal distribution')

#We now use boxcox to transform the data so it’s closest to normal:
ax2 = fig.add_subplot(212)
prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
ax2.set_title('Probplot after Box-Cox transformation')
plt.show()

def linear_model_regr(train, strResponse):
    from sklearn import linear_model
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split
    from math import sqrt

    # Devide in train and test
    train, test = train_test_split(train, test_size=0.15, random_state=123)
    train.shape, train.head(2)
    test.shape, test.head(2)

    listAllPredictiveFeatures = np.setdiff1d(train.columns,strResponse)
    train[listAllPredictiveFeatures].head(2)

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(train[listAllPredictiveFeatures], train[strResponse])

    # The coefficients
    regr.coef_ # 0.02491209,  0.00128227, -0.00623236,  0.73590393, -0.38926939, 0.00356409

    # Predict on test (unseen) data
    pred = regr.predict(test[listAllPredictiveFeatures])

    # Error ranges
    mae = mean_absolute_error(test[strResponse], pred)
    rmse = sqrt(mean_squared_error(test[strResponse], pred)) # (2.63, 3.42)

    # R 2(square): 1 is perfect
    rsquare = round(regr.score(test[listAllPredictiveFeatures], test[strResponse]) * 100, 0) # 59%
    print("Test mean: ", np.mean(test[strResponse]))
    print("Mae: ", mae, ", Rmse: ", rmse, ", Rsquare: ", rsquare)

    # Plot outputs
    plt.scatter(range(test.shape[0]), pred,  color='red', marker = '*', s = 30)
    plt.scatter(range(test.shape[0]), test[strResponse],  color='green', marker = '*', s = 60)
    plt.show()

    # Residual plot
    ResidualPlot(test[strResponse], pred, fileImageToSave = "./Images/Residual.png")

    return # end of linear_model_regr


train = pd.read_csv("data/cats.csv")
train.columns = map(str.upper, train.columns)
train.drop(catColumns, axis = 1, inplace=True)

linear_model_regr(train, strResponse) ## Mae:  1.4131156502407185 , Rmse:  1.7922781967412666 , Rsquare:  66.0 Durbin_Watson:2.0-


train = pd.read_csv("data/cats.csv")
train.columns = map(str.upper, train.columns)

catColumns = np.intersect1d(train.columns, catColumns)
train[catColumns] = train[catColumns].apply(lambda x: x.astype('category'))

Encoding(train, 'HWT', scale_and_center = True, fileTrain = "./data/cats_train_EncodedScaled.csv", fileTest = "./data/cats_test_EncodedScaled.csv")


train = pd.read_csv("data/cats_train_EncodedScaled.csv")
train.columns = map(str.upper, train.columns)
train.info()

linear_model_regr(train, strResponse)  ## Mae:  1.4444353656067614 , Rmse:  1.8243061166390109 , Rsquare:  64.0 Durbin_Watson:2.03

from catboost import CatBoostRegressor


train = pd.read_csv("data/cats.csv")
train.columns = map(str.upper, train.columns)

listAllPredictiveFeatures = np.setdiff1d(train.columns,strResponse)

train, test = train_test_split(train, test_size=0.15, random_state=123)

train[listAllPredictiveFeatures].dtypes

categorical_features_indices = list(np.where(train[listAllPredictiveFeatures].dtypes == np.object)[0])



if len(categorical_features_indices) == 0:
    categorical_features_indices = None


model=CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE', random_seed = seed_value)
model.fit(train[listAllPredictiveFeatures], train[strResponse],cat_features=categorical_features_indices) # ,plot=True

pred = model.predict(test[listAllPredictiveFeatures])

# RMSE Error
rmse = np.sqrt(mean_squared_error(test[strResponse], pred))
rmse  ## RMSE - 1.97

mae = mean_absolute_error(test[strResponse], pred)  ## 1.57


