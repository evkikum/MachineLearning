# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:36:41 2019

@author: evkikum
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from pydataset import data
import os
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
import seaborn as sns


os.getcwd()
os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\Practice\Linear Regression\mtcars")

df = pd.read_csv("mtcars.csv")
df.info()


y = df.loc[:, "mpg"]
X = df.loc[:, df.columns != "mpg"]
feature_names = df.loc[:,df.columns != "mpg"].columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.7, random_state = 123)
X_train.shape


lm = LinearRegression().fit(X_train, y_train)

## BASED ON THE BELOW R SQUARE IT SHOWS THAT THE MODEL PERFORMS VERY WELL ON THE TRAINING DATA TO WHICH IT WAS FITTED.
r2_score1 = lm.score(X_train, y_train) ## R_square = 1 , 
lm.intercept_
lm.coef_


##Now lest the model performance on test data;
predicted_test = lm.predict(X_test)
r2_score_test_date = lm.score(X_test, y_test)  ### -15.88. In the above R_Square # -15.88 which is much lower than training data which shows strong overfitting.Granted, our test set is not very large, so some fluctuation is expected
mean_absolute_error = np.mean(abs(y_test - predicted_test ))   ## 15.6%
rmse = np.sqrt(np.mean((y_test - predicted_test)** 2))  ## 22% - Root Mean square error


## One way to reduce overfitting is to remove some predictive features from the model. Ideally we would be able to examine many or all possible subsets of features and 
## select the subset of features that gives the best performance, but that is usually impractical due to the large number of possible subsets. A common alternative is to 
## start from the full list of features and recursively remove one that seems to be contributing least to the model's performance (i.e., the feature whose removal has the 
## least negative/most positive effect on model performance). This process is called recursive feature elimination (RFE).

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# recursive feature elimination with cross validation, using r-squared score as metric
lm = LinearRegression()
rfecv = RFECV(estimator=lm, step=1, cv=5)
rfecv.fit(X_scaled, y_train)

print("Optimal no of features : {0}".format(rfecv.n_features_))
print("Features selected : {}".format(','.join(np.array(feature_names)[rfecv.support_].tolist())))



## GET THE FEATURES ELIMINATION ORDER
rank_feat = pd.DataFrame(feature_names, rfecv.ranking_.tolist()).reset_index()
rank_feat.columns = ["Rank", "Feature_name"]
rank_feat = rank_feat.sort_values("Rank", ascending = False)

print("Suggested order of feature removal {}".format(rank_feat["Feature_name"].tolist()))

rfecv.grid_scores_
plt.xlabel("No of features selected")
plt.ylabel("Score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


X_train_subset = X_train.loc[:, rfecv.support_]
lm2 = LinearRegression().fit(X_train_subset, y_train)
##r2_score2 = lm2.score(X_train_subset, y_train) ## 77.4 % R_Square
r2_score2 = r2_score(y_train, lm2.predict(X_train_subset) )  ## 0.77 R Square

predicted_test2 = lm2.predict(X_test.loc[:, rfecv.support_])
r2_score_test_date = r2_score(y_test, predicted_test2)  ##  -0.22 R Square
mae = np.mean(abs(y_test - predicted_test2))  ## 4.94 % 
rmse = np.sqrt(np.mean((y_test - predicted_test2)**2))






