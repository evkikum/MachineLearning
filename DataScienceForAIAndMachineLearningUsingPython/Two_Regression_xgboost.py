#%% xgboost
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# Constants
catColumns = ['ORIGIN']; strResponse = 'MPG'

# Working directory
os.chdir(r"/home/evkikum/Desktop/Data Science/Python/Udemy_Shiv")
exec(open(os.path.abspath('CommonUtils.py')).read())

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value)

# Read data afresh
train = pd.read_csv("./data/mpg_train_EncodedScaled.csv")
train.columns = map(str.upper, train.columns)

# Devide in train and test
train, test = train_test_split(train, test_size=0.15, random_state=123)

listAllPredictiveFeatures = np.setdiff1d(train.columns,strResponse)

# Get data in form DMatrix as required by xgboost
d_train = xgb.DMatrix(train[listAllPredictiveFeatures], label=train[strResponse])

# https://xgboost.readthedocs.io/en/latest/parameter.html
params = {'eval_metric' : 'mae', 'eta': 0.3, 'colsample_bytree': 1,
          'subsample': 1,  'max_depth': 6,  'nthread' : 4,
          'booster' : 'gbtree', 'objective' : "reg:linear", 'seed' : seed_value}

#Build model on training data
regressor = xgb.train(dtrain = d_train, params = params, num_boost_round = 500)

#Self Prediction
pred = regressor.predict(d_train)

# MAE
mean_absolute_error(train[strResponse], pred)

#plot importance, use plot_importance. This function requires matplotlib to be installed.
xgb.plot_importance(regressor)
#xgb.plot_tree(regressor, num_trees=2)
#xgb.to_graphviz(regressor, num_trees=2)

# Predict on test data
d_test = xgb.DMatrix(test[listAllPredictiveFeatures])
pred = regressor.predict(d_test)

# MAE
mean_absolute_error(test[strResponse], pred) # 2.04

# Class work: Draw scatter plot of actual and pred
# Class work: Call Residual plot and do Analysis
