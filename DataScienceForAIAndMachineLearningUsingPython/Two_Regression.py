#%%    # Import libraries
import os
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

# Working directory
os.chdir(r"/home/evkikum/Desktop/Data Science/Python/Udemy_Shiv")
exec(open(os.path.abspath('CommonUtils.py')).read())

# Some standard settings
plt.rcParams['figure.figsize'] = (13, 9) #(16.0, 12.0)
plt.style.use('ggplot')


#Set PANDAS to show all columns in DataFrame
pd.set_option('display.max_columns', None)
#Set PANDAS to show all rows in DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('precision', 2)

# Constants
catColumns = ['ORIGIN']; strResponse = 'MPG'

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value)
#%%  Read data, Descriptive Analysis
train = pd.read_csv("./data/mpg.csv")
train.columns = map(str.upper, train.columns)

# First view
train.info()
train.dtypes
train.head(2)
train.index # get row index # train.index[0] # 1st row label # train.index.tolist() # get as a list
train.shape

# Change data types
#catColumns = list(set(train.columns).intersection(catColumns));
catColumns = np.intersect1d(train.columns, catColumns)
train[catColumns] = train[catColumns].apply(lambda x: x.astype('category'))

# View summary
print(train.describe(include = 'all'))
#%% How to see whether response distribution is near to Normal

# histogram plot
plt.hist(train[strResponse])
plt.show()

# Density plot - Similar to hist
train[strResponse].plot(kind='kde')
plt.show()

# q-q plot
import statsmodels.api as sm
sm.qqplot(train[strResponse], line='s')
plt.show()

# Shapiro-Wilk Test for normality test
from scipy.stats import shapiro
stat, p = shapiro(train[strResponse])
print('Statistics=%.2f, p=%.2f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (Normal) (accept H0)')
else:
	print('Sample does not look Gaussian (Normal) (reject H0)')

# Many more - D’Agostino’s K^2 Test, Anderson-Darling Test,
#%% How to make response distribution near to Normal

#You can try for X^2, X^3, sqrt(x), log(x), exp(x), sin(x) and so on

# Box-Cox power transformation for "positive" response ONLY.
#y = (x**lmbda - 1) / lmbda,  for lmbda > 0
#    log(x),                  for lmbda = 0

from scipy import stats
from scipy.special import inv_boxcox

x = train[strResponse] # Untransformed
xt, bx_lambda = stats.boxcox(x) # Box cox transformed and Lambda stored

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

# Class work: 1. Do inverse Box Cox transformation. 2. Print first 10 data and see any difference
# 3. Take sum of difference of original and inverse transformed data.
#%% # Build basic Liner model
#Functions specific to this file
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
#%% Linear Regression without categorical and without scaling of data
#  Read data afresh
train = pd.read_csv("./data/mpg.csv")
train.columns = map(str.upper, train.columns)

# Drop categorical columns
train.drop(catColumns, axis=1, inplace=True)

# Run Linear regression
linear_model_regr(train, strResponse)

#%% Liner Regression with hotcoding of categorical and scaling of numeric data
# Read data afresh
train = pd.read_csv("./data/mpg.csv")
train.columns = map(str.upper, train.columns)

# Change data types
catColumns = np.intersect1d(train.columns, catColumns)
train[catColumns] = train[catColumns].apply(lambda x: x.astype('category'))

# View summary
print(train.describe(include = 'all'))

## get all numeric features for scalling
#list_numeric_features = list(set(train.columns) - set([strResponse]).union(catColumns))
#train[listAllPredictiveFeatures].head(2) # train.head(2)

#Dummy Encoding of categorical data, scale and center numerical data
Encoding(train, strResponse, scale_and_center = True, fileTrain = "./data/mpg_train_EncodedScaled.csv", fileTest = "./data/mpg_test_EncodedScaled.csv")
del(train);

#%% start afresh with encoded and scale data
# Read data afresh
train = pd.read_csv("./data/mpg_train_EncodedScaled.csv")
train.columns = map(str.upper, train.columns)

linear_model_regr(train, strResponse) # Mae:  2.42990043841 , Rmse:  2.9897013064129854 , Rsquare:  83.0
#%% Generic Tree flow - PPT

#%%CatBoost: A machine learning library to handle categorical (CAT) data automatically
# https://tech.yandex.com/catboost/
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor

#  Read data
train = pd.read_csv("./data/mpg.csv")
train.columns = map(str.upper, train.columns)

# First view
train.dtypes
train.head(2)

# Change data types
#catColumns = list(set(train.columns).intersection(catColumns));
#train[catColumns] = train[catColumns].apply(lambda x: x.astype('category'))

# Get list of IV
listAllPredictiveFeatures = np.setdiff1d(train.columns,strResponse)

# Devide in train and test
train, test = train_test_split(train, test_size=0.15, random_state=123)

#The model need index of category features
train[listAllPredictiveFeatures].dtypes

# Through object identification
categorical_features_indices = list(np.where(train[listAllPredictiveFeatures].dtypes == np.object)[0])

## In generic: with assuption that non numeric is category
#categorical_features_indices_f = np.where(train[listAllPredictiveFeatures].dtypes != np.float64)[0]
#categorical_features_indices_i = np.where(train[listAllPredictiveFeatures].dtypes != np.int64)[0]
#categorical_features_indices = list(set(categorical_features_indices_f).intersection(categorical_features_indices_i))

# Basic sanity test
if len(categorical_features_indices) == 0:
    categorical_features_indices = None

#building model
model=CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE', random_seed = seed_value)
model.fit(train[listAllPredictiveFeatures], train[strResponse],cat_features=categorical_features_indices) # ,plot=True

#Now, the next task is to predict the outcome for test data set.
pred = model.predict(test[listAllPredictiveFeatures])

# RMSE Error
rmse = np.sqrt(mean_squared_error(test[strResponse], pred))
rmse
#%% XgBoost - PPT
#Hands on with Two_Regression_xgboost.py
