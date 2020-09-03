#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:22:11 2020

@author: evkikum

The main objective of Lasso/Ridge regression is to reduce 
1) Cost function or overfitting


Ridge and Lasso regression are some of the simple techniques to reduce model complexity and prevent 
over-fitting which may result from simple linear regression.

overfitting ==> When the model is tested with train data there is no error but when tested with test data there is high error
underfitting ==> When the model is tested with train/test data there is high error.

As the lambda increases then slope for all the independent variables will tend to become zero.

THE BELOW CODE IS WRITTEN FROM THE BELOW URL
https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b

Ridge and Lasso Regression Indepth Intuition- Data Science
https://www.youtube.com/watch?v=9lRv01HDU0s
https://www.youtube.com/watch?v=0yI0-r3Ly40


Note ==> A good model (or generalized model) should always have low variance and low bias
"""
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 12}) 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

boston=load_boston()
boston_df=pd.DataFrame(boston.data,columns=boston.feature_names)

# add another column that contains the house prices which in scikit learn datasets are considered as target
boston_df['Price']=boston.target

newX=boston_df.drop('Price',axis=1)
print(newX[0:3])  # check 
newY=boston_df['Price']


X_train,X_test,y_train,y_test=train_test_split(newX,newY,test_size=0.3,random_state=3)
print(len(X_test), len(y_test)) 

lr = LinearRegression()
lr.fit(X_train, y_train)
rr = Ridge(alpha=0.01) # higher the alpha value, more restriction on the coefficients; low alpha > more generalization, coefficients are barely

# restricted and in this case linear and ridge regression resembles

rr.fit(X_train, y_train)

rr100 = Ridge(alpha=100) #  comparison with alpha value
rr100.fit(X_train, y_train)

train_score=lr.score(X_train, y_train)
test_score=lr.score(X_test, y_test)

Ridge_train_score = rr.score(X_train,y_train)
Ridge_test_score = rr.score(X_test, y_test)
Ridge_train_score100 = rr100.score(X_train,y_train)
Ridge_test_score100 = rr100.score(X_test, y_test)

print( "linear regression train score:", train_score)
print("linear regression test score:", test_score) 
print("ridge regression train score low alpha:", Ridge_train_score) 
print("ridge regression test score low alpha:", Ridge_test_score) 
print( "ridge regression train score high alpha:", Ridge_train_score100)
print("ridge regression test score high alpha:", Ridge_test_score100) 

'''
Let’s understand the figure below. In X axis we plot the coefficient index and, for Boston data there are 
13 features (for Python 0th index refers to 1st feature). For low value of α (0.01), when the coefficients 
are less restricted, the magnitudes of the coefficients are almost same as of linear regression. For higher 
value of α (100), we see that for coefficient indices 3,4,5 the magnitudes are considerably less compared to 
linear regression case. This is an example of shrinking coefficient magnitude using Ridge regression.
'''

plt.plot(rr.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 0.01$',zorder=7) # zorder for ordering the markers

plt.plot(rr100.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge; $\alpha = 100$') # alpha here is for transparencyplt.plot(lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.show()