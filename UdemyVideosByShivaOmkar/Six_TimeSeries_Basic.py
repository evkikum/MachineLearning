#%% Loading and Handling Time Series in Pandas
#Pandas has dedicated libraries for handling TS objects, particularly the datatime
#class which stores time information and allows us to perform some operations really fast.
import os
os.chdir("/home/evkikum/Desktop/Data Science/Python/Udemy_Shiv")
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
from datetime import datetime
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value)

#Now, we can load the data set and look at some initial rows and data types of the columns:
data = pd.read_csv('./data/AirPassengers.csv')
data.head()
data.dtypes

#date time conversion
data['TravelDate'] =  pd.to_datetime(data['TravelDate'], format='%m/%d/%Y')
data['Passengers'] = data['Passengers'].astype(float)
data.dtypes # Notice the dtype=’datetime[ns]’

#Index: getting time to index
data.set_index('TravelDate', inplace=True)
data.head()

# Convert to series as convinint and required by TS
ts = data['Passengers']
ts.head(10)

# Some exploration
#Specific the index as a string constant
ts['1949-01-01']

#using 'datetime' function
ts[datetime(1949,1,1)]

#Specify the entire range:
ts['1949-01-01':'1949-05-01']

#':' if one of the indices is at ends
ts[:'1949-05-01']

#Consider another instance where you need all the values of the year 1949. This can be done as:
ts['1949']

# Theoritical frequency
pd.infer_freq(ts.index)

#simple plot the data and analyze visually. The data can be plotted using following command:
plt.plot(ts)
plt.show()

#By default the labels are set to the right edge of the window, but a center keyword is available
#so the labels can be set at the center (instead of starting from window location)
moving_avg = ts.rolling(window = 12, center=False).mean()
#Class work: Calculate Moving Avg through array calculation and match results

#Calculate Moving Rolling std and exponentially weighted moving average
moving_std = ts.rolling(window = 12, center=False).std()
moving_ewma = ts.ewm(span = 12).mean() # Span is an “N-day EW moving average”.

#Class work: Plot all 4 in one graph

#Dickey-Fuller Test: This is one of the statistical tests for checking stationarity. Here
# the null hypothesis is that the TS is non-stationary.
#Perform Dickey-Fuller test:
print('Results of Dickey-Fuller Test:')
dftest = adfuller(ts, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value

dfoutput # p is 0.99 and hence H0 can not be rejected. Hence TS is non-stationary

#How to make a Time Series Stationary? #1. Trend #2. Seasonality

#Decomposing
decomposition = seasonal_decompose(ts, freq = 12)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#decompose
#Here we can see that the trend, seasonality are separated out from data and we can model the residuals
#. Lets check stationarity of residuals:
residual.dropna(inplace=True)

print('Results of Dickey-Fuller Test:')
dftest = adfuller(residual, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value

dfoutput # p is 0.0002 and hence H0 can be rejected. Hence TS is stationary

# Class work: Take one diff and perform Perform Dickey-Fuller test:

#%%  Missing value imputation
# see how many are missing
ts.isnull().sum()

# None missing, let us put some dummy NaN
ts[2] = np.NaN
ts[:5]

# Fill in forward way
ts.fillna(method='ffill', inplace=True)
ts[:5]

# None missing, let us put some dummy NaN
ts[2] = np.NaN

# Fill in backward way
ts.fillna(method='bfill', inplace=True)
ts[:5]

# Interpolation ways
ts[2] = np.NaN
ts.plot()
ts.interpolate().plot()
#ts.interpolate(method='time') # For a floating-point index, use method='values
#If you are dealing with a time series that is growing at an increasing rate, method='quadratic' or 'cubic' may be appropriate.
#If you have values approximating a cumulative distribution function, then method='pchip' should work well.
#To fill missing values with goal of smooth plotting, use method='akima'.
