#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 14:31:06 2020

@author: evkikum

Ridge and Lasso regression are some of the simple techniques to reduce model complexity and prevent 
over-fitting which may result from simple linear regression.

overfitting ==> When the model is tested with train data there is no error but when tested with test data there is high error
underfitting ==> When the model is tested with train/test data there is high error.

THE BELOW CODE IS WRITTEN FROM THE BELOW URL
https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b

Ridge and Lasso Regression Indepth Intuition- Data Science
https://www.youtube.com/watch?v=9lRv01HDU0s
https://www.youtube.com/watch?v=0yI0-r3Ly40

Note ==> A good model (or generalized model) should always have low variance and low bias

"""

import math 
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

# difference of lasso and ridge regression is that some of the coefficients can be zero i.e. some of the features are 
# completely neglected

from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

print(cancer.keys())

cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

cancer_df.head(3)

X = cancer.data
Y = cancer.target

X_train,X_test,y_train,y_test=train_test_split(X,Y, test_size=0.3, random_state=31)

lasso = Lasso()
lasso.fit(X_train,y_train)
train_score=lasso.score(X_train,y_train)
test_score=lasso.score(X_test,y_test)
coeff_used = np.sum(lasso.coef_!=0)

## 1) The default value for regularization parameter in LASSO regression is 1.

print("======Default lasso Score====")
print("training score:", train_score ) 
print("test score: ", test_score) 
print("number of features used: ", coeff_used)  
print()

lasso001 = Lasso(alpha=0.01, max_iter=10e5)
lasso001.fit(X_train,y_train)

train_score001 = lasso001.score(X_train,y_train)
test_score001 = lasso001.score(X_test,y_test)
coeff_used001 = np.sum(lasso001.coef_!=0)

print("======alpha=0.01==============")
print("training score for alpha=0.01:", train_score001 ) 
print("test score for alpha =0.01: ", test_score001) 
print("number of features used: for alpha =0.01:", coeff_used001) 
print()

lasso00001 = Lasso(alpha=0.0001, max_iter=10e5)
lasso00001.fit(X_train,y_train)

train_score00001=lasso00001.score(X_train,y_train)
test_score00001=lasso00001.score(X_test,y_test)
coeff_used00001 = np.sum(lasso00001.coef_!=0)

print("======alpha=0.0001==============")
print("training score for alpha=0.0001:", train_score00001 ) 
print("test score for alpha =0.0001: ", test_score00001) 
print("number of features used: for alpha =0.0001:", coeff_used00001) 
print()

lr = LinearRegression()
lr.fit(X_train,y_train)
lr_train_score=lr.score(X_train,y_train)
lr_test_score=lr.score(X_test,y_test)

print("LR training score:", lr_train_score ) 
print("LR test score: ", lr_test_score) 

'''
Below are the key points for Lasso Regression
1) The default value for regularization parameter in LASSO regression is 1.
2) With this, out of 30 features in cancer data-set, only 4 features are used (non zero value of the coefficient).
3) Both training and test score (with only 4 features) are low; conclude that the model is under-fitting the cancer data-set.
4) Reduce this under-fitting by reducing alpha and increasing number of iterations. Now α = 0.01, non-zero features =10, 
training and test score increases.
5) Comparison of coefficient magnitude for two different values of alpha are shown in the left panel of figure 2. For alpha =1,
  we can see most of the coefficients are zero or nearly zero, which is not the case for alpha=0.01.
6) Further reduce α =0.0001, non-zero features = 22. Training and test scores are similar to basic linear regression case.
7) In the right panel of figure, for α = 0.0001, coefficients for Lasso regression and linear regression show close resemblance.
'''

plt.subplot(1,2,1)
plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$') # alpha here is for transparency

plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)

plt.subplot(1,2,2)
plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$') # alpha here is for transparency
plt.plot(lasso00001.coef_,alpha=0.8,linestyle='none',marker='v',markersize=6,color='black',label=r'Lasso; $\alpha = 0.00001$') # alpha here is for transparency
plt.plot(lr.coef_,alpha=0.7,linestyle='none',marker='o',markersize=5,color='green',label='Linear Regression',zorder=2)
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.tight_layout()
plt.show()