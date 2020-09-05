# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:50:14 2019

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

### https://notebooks.azure.com/aditya-swami/projects/is5152/html/Evaluating%20Multiple%20Models%20in%20a%20Python%203%20Notebook.ipynb
### https://www.kaggle.com/rishisharma/linear-regression-eda-with-python
### https://www.kaggle.com/rishisharma/linear-regression-eda-with-python



####THE PURPOSE OF THIS EXAMPLE IS TO EVALUATE THE 3 MODELS;
####1) LINEAR MODEL USING ALL VARIABLES
####2) LINEAR MODEL AFTER VARIABLE SELECTION
####3) GRADIANT BOOSTING MACHINE (GBM) MODEL

os.getcwd()
os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\Practice\Linear Regression\mtcars")

df = pd.read_csv("mtcars.csv")
df.info()
df.describe()
data('mtcars', show_doc=True)
df.head()

##NOW WILL SPLIT THE DATA INTO TRAINING AND TESTING DATA;
y = df["mpg"].values
## in the below axis = 0 means index and 1 means columns
## DELETE THE mpg COLUMN in DATAFRAME df
X = df.drop("mpg",axis = 1).values

feature_names = df.drop('mpg',1).columns

##SAVE 30% OF THE RECORDS FOR THE TEST SET.
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.7, random_state = 123)


lm = LinearRegression()
lm.fit(X_train, y_train)

param_df = pd.DataFrame({
                         "Coefficient" : [lm.intercept_] + list(lm.coef_),
                         "Feature" : ['intercept'] + list(feature_names) })

predicted = lm.predict(X_test)
r_squared = r2_score(y_train,lm.predict(X_train))


mae = np.mean(abs(predicted - y_test))              ## Mean Absolute Error 	
rmse = np.sqrt(np.mean((predicted - y_test)**2))    ## Root Mean Squared Error 	
rae = np.mean(abs(predicted - y_test)) / np.mean(abs(y_test - np.mean(y_test)))     ## Relative Absolute Error
rse = np.mean((predicted - y_test)**2) / np.mean((y_test - np.mean(y_test))**2)     ## Relative Squared Error

summary_df = pd.DataFrame(index = ['R-squared', 'Mean Absolute Error', 'Root Mean Squared Error',
                                   'Relative Absolute Error', 'Relative Squared Error'])
summary_df['Linear Regression, all variables'] = [r_squared, mae, rmse, rae, rse]
summary_df

####Notice that the R-squared value for true vs. predicted mpg of the test set is much lower than it was for the training set. (Granted, our test set is not very large, so some fluctuation is expected.) This is indicative of model overfitting.


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
lm = LinearRegression() 
##IN THE BELOW STEP=1 MEANS THAT AT EACH ITERATION ONLY ONE FEATURE CAN BE REMOVED. CV STANDS FOR CROSS VALIDATION.
rfecv = RFECV(estimator=lm, step=1, cv=5)

rfecv.fit(X_scaled, y_train)

print('Optimal number of features: {}'.format(rfecv.n_features_))

print('Features selected: {}'.format(', '.join(np.array(feature_names)[rfecv.support_].tolist())))

ranked_features, _ = zip(*sorted(zip(feature_names, rfecv.ranking_.tolist()),
                                 key=lambda x: x[1],
                                 reverse=True))
print('Suggested order of feature removal: {}'.format(', '.join(ranked_features)))

sns.set_style("darkgrid")
plt.figure(figsize=(8, 4))
plt.xlabel("Number of features selected")
plt.ylabel("Score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()




    X_train_subset = X_train[:, rfecv.support_]
lm2 = LinearRegression()
lm2.fit(X_train_subset, y_train)


X_test_part = X_test[:, rfecv.support_]
predicted = lm2.predict(X_test_part)

r_squared = r2_score(y_test, predicted)
mae = np.mean(abs(predicted - y_test))
rmse = np.sqrt(np.mean((predicted - y_test)**2))
rae = np.mean(abs(predicted - y_test)) / np.mean(abs(y_test - np.mean(y_test)))
rse = np.mean((predicted - y_test)**2) / np.mean((y_test - np.mean(y_test))**2)
