#%%Forecasting a Time Series
## acf vs pacf
## https://www.youtube.com/watch?v=5Q5p6eVM7zM  
## https://www.youtube.com/watch?v=DeORzP0go5I
## (The below explains the codes)
## https://www.youtube.com/watch?v=y8opUEd05Dg    
## (The below provides the codes for the above video)
## https://github.com/ritvikmath/Time-Series-Analysis  

import os
import sys
import pandas as pd
import numpy as np

os.chdir(r"/home/evkikum/Desktop/Data Science/Python/Udemy_Shiv")
exec(open(os.path.abspath('CommonUtils.py')).read())

import matplotlib.pylab as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value)

#%% ARIMA and ARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA, ARMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
import warnings
import itertools
from pandas.plotting import autocorrelation_plot

#Now, we can load the data set and look at some initial rows and data types of the columns:
data = pd.read_csv('./data/AirPassengers.csv')

#date time conversion
data['TravelDate'] =  pd.to_datetime(data['TravelDate'], format='%m/%d/%Y')
data['Passengers'] = data['Passengers'].astype(float)
data.dtypes # Notice the dtype=’datetime[ns]’

#Index: getting time to index
data.set_index('TravelDate', inplace=True)
data.head()

# Convert to series as convinint and required by TS
ts = data['Passengers']

# First view
plt.plot(ts)
plt.show()
plt.close(); plt.gcf().clear()

# Manual forecasting

# First let us see if TS is stationary
#Perform Dickey-Fuller test:
print('Results of Dickey-Fuller Test:')
dftest = adfuller(ts, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value

if dfoutput['p-value'] > 0.05:
    print("Accept the null hypothesis (H0), the data has a unit root and is non-stationary.")
elif dfoutput['p-value'] <= 0.05:
    print("Reject the null hypothesis (H0), the data does not have a unit root and is stationary.")

dfoutput # p is 0.99 and hence H0 can not be rejected. Hence TS is non stationary

# Generally after diff, tss becomes stationary. Let us see
# Take the diff to draw acf and pacf
ts_diff = ts - ts.shift()
plt.plot(ts_diff) # See the stationarity

# ts diff. Remove the first NaN as DF test does not like NaN
ts_diff.dropna(inplace=True)

#Perform Dickey-Fuller test:
print('Results of Dickey-Fuller Test:')
dftest = adfuller(ts_diff, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value

if dfoutput['p-value'] > 0.05:
    print("Accept the null hypothesis (H0), the data has a unit root and is non-stationary.")
elif dfoutput['p-value'] <= 0.05:
    print("Reject the null hypothesis (H0), the data does not have a unit root and is stationary.")

dfoutput # p is 0.054 and hence H0 can not be rejected. Hence TS is non stationary for 5% although
#stationary at 10% p value

# Now determine ARIMA(p,d,q)

#Autocorrelation Function (ACF): It is a measure of the correlation between the the TS with a
# lagged version of itself. For instance at lag 5, ACF would compare series at time instant ‘t1’…’t2’ with
#series at instant ‘t1-5’…’t2-5’ (t1-5 and t2 being end points).

#Partial Autocorrelation Function (PACF): This measures the correlation between the TS with a lagged
#version of itself but after eliminating the variations already explained by the intervening comparisons.
# Eg at lag 5, it will check the correlation but remove the effects already explained by lags 1 to 4.

season = 12

# Let us plot acf and pacf

#In this plot, the two dotted lines on either sides of 0 are the confidence interevals.
#These can be used to determine the ‘p’ and ‘q’ values as:

#Plot ACF: q – The lag value where the ACF chart crosses the upper confidence interval for the first
# time. If you notice closely, in this case q=2.

#Plot PACF: p – The lag value where the PACF chart crosses the upper confidence interval for the
#first time. If you notice closely, in this case p=2.
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(ts, lags=20, ax=ax1)
ax1.xaxis.set_ticks_position('bottom')
plt.title('Autocorrelation Function')
fig.tight_layout();

ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(ts, lags=20, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
plt.title('Partial Autocorrelation Function')
fig.tight_layout();
plt.show()
plt.close(); plt.gcf().clear()

#We need to load the ARIMA model first:
#The p,d,q values can be specified using the order argument of ARIMA which take a tuple (p,d,q).
model = ARIMA(ts, order=(2, 1, 2))
results_ARIMA = model.fit()  # disp=-1

# Plot ARIMA fitted value
plt.plot(ts, color='blue')
plt.plot(results_ARIMA.fittedvalues, color='red')
self_pred = np.abs(results_ARIMA.fittedvalues); self_pred.dropna(inplace=True)
plt.title('RMSE: %.4f'% np.sqrt(sum((self_pred-ts[1:])**2)/len(ts))) # 285
plt.show()
plt.close(); plt.gcf().clear()

# Need improvement???

# Use SARIMAX to include seasonality
# Define the p, d and q parameters to take any value between 0 and 2
p = q = range(2, 5); d = range(0, 1)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
len(pdq) == len(p)*len(q)*len(d)

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], season) for x in pdq]

#The code chunk below iterates through combinations of parameters and uses the SARIMAX function from
#statsmodels to fit the corresponding Seasonal ARIMA model. Here, the order argument specifies the
#(p, d, q) parameters, while the seasonal_order argument specifies the (P, D, Q, S) seasonal component
#of the Seasonal ARIMA model. After fitting each SARIMAX()model, the code prints out its respective
#AIC score.
warnings.filterwarnings("ignore") # specify to ignore warning messages
best_aic = np.inf; param_best = pdq[0]; param_seasonal_best = seasonal_pdq[0]; at_least_one_success = False
for param in pdq: # param = pdq[5]
    for param_seasonal in seasonal_pdq: # param_seasonal = seasonal_pdq[5]
        try: # Explanation of enforce_stationarity and enforce_invertibility is at bottom
            mod = SARIMAX(ts, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
            results = mod.fit()
            print('SARIMAX{}x{}{} - AIC:{}'.format(param, param_seasonal, season, np.round(results.aic,0)))
            if results.aic < best_aic: # numpy.isnan
                best_aic = results.aic; param_best = param; param_seasonal_best = param_seasonal; at_least_one_success = True
                # Save in config
                #str_param = ','.join([','.join(str(p) for p in param_best), ','.join(str(p) for p in param_seasonal_best)])
                print('Latest best: ' + str(param_best) + str('x') +str(param_seasonal_best))
            del(mod, results)
        except:
            print('Exception -> ARIMAX{}x{}, Error: {}'.format(param, param_seasonal,  sys.exc_info()[0]))
            continue
        # for param_seasonal
    # for param

if not at_least_one_success:
    print('SARIMAX has failed')

print('Best observation is ARIMA{}x{} - AIC:{}'.format(param_best, param_seasonal_best, best_aic))
#Best observation is ARIMA(5, 0, 2)x(5, 0, 2, 12)12 - AIC:624.1499710804894
#Best observation is ARIMA(4, 0, 4)x(4, 0, 2, 12)12 - AIC:718.0640628188848
#param_best = (4, 0, 4); seasonal_order=(4, 0, 2, 12)

#Fitting an ARIMA Time Series Model
mod = SARIMAX(ts, order=param_best, seasonal_order=param_seasonal_best,
              enforce_stationarity=False, # Whether or not to transform the AR parameters to enforce stationarity in the autoregressive component of the model. Default is True.
              enforce_invertibility=False) # Whether or not to transform the MA parameters to enforce invertibility in the moving average component of the model. Default is True.
results = mod.fit()

# Self forecast
self_pred = np.abs(results.fittedvalues)

# Graphical view
plt.plot(ts, color = 'red')
plt.plot(self_pred, color = 'green')
plt.show()
plt.close(); plt.gcf().clear()

#Residual Analysis and various error terms
mae, rmse, mape = ResidualPlot(ts, self_pred, "./Images/residual_sarimax.png")
print(''.join(['mean:', str(np.round(ts.mean(), 2)), ', mae:', str(mae),', rmse:', str(rmse),', mape:',str(mape)]))
# Durbin_Watson:1.43, mean:280.3, mae:11.85, rmse:19.98, mape:6.36

#Rolling Forecast ARIMA Model. forecast is good for original level
#forecast of the very next time step or given step in the sequence from the available
#data used to fit the model.
pred = results.forecast(steps = 12)
pred.head(2)
#1961-01-01    459.236058
#1961-02-01    430.688488

#A line plot is created showing the expected values (blue) compared to the rolling forecast predictions
# (red). We can see the values show some trend and are in the correct scale.
plt.plot(ts, color = 'blue')
plt.plot(pred, color = 'red')
plt.show()
plt.close(); plt.gcf().clear()

# Looks Good

# For future prediction (start, end)
#The predict function can be used to predict arbitrary in-sample and out-of-sample
#time steps, including the next out-of-sample forecast time step
nforecast = 10
pred = results.predict(start=mod.nobs, end=mod.nobs + nforecast)
pred.head(10)

#%% Time Series Forecasts using Facebook’s Prophet
# conda install -c conda-forge fbprophet
# Theory on PPT

from fbprophet import Prophet

#Now, we can load the data set and look at some initial rows and data types of the columns:
df_prophet = pd.read_csv('./data/AirPassengers.csv')

#date time conversion
df_prophet['TravelDate'] =  pd.to_datetime(df_prophet['TravelDate'], format='%m/%d/%Y')
df_prophet['Passengers'] = df_prophet['Passengers'].astype(float)

#Prophet requires the variable names in the time series to be:
#y – Target
#ds – Datetime

df_prophet = df_prophet.rename(columns={'TravelDate': 'ds', 'Passengers': 'y'})
df_prophet.head(5)

#Fitting the prophet model:
#seasonality_prior_scale: The strength of the seasonality model. Larger values allow the model
#to fit larger seasonal fluctuations, smaller values dampen the seasonality
model = Prophet(growth='linear', yearly_seasonality = True, seasonality_prior_scale=0.5)
model.fit(df_prophet)

# to predict future, create dummy DF
df_future = model.make_future_dataframe(periods=12, freq='MS') # It will have actual + 12 steps
df_future.tail(12)

# Forecast the future data
forecast = model.predict(df_future)

#Prophet returns a large DataFrame with many interesting columns
forecast.tail(12)

#subset our output to the columns most relevant to forecasting, which are:
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)

# We can look at the various components using the following command:
model.plot(forecast, uncertainty=True)
plt.show()
plt.close(); plt.gcf().clear()

#Plot components: daily, weekly and yearly patterns contribute to the overall forecasted values:
model.plot_components(forecast)
plt.show()
plt.close(); plt.gcf().clear()

# Putting all together
plt.plot(df_prophet['ds'].values, df_prophet['y'].values,  color='red', label='y') # marker = '*', s = 30,
plt.plot(forecast['ds'].values, forecast['yhat'].values,  color='green', label='yhat')
plt.plot(forecast['ds'].values, forecast['yhat_upper'].values,  color='yellow', label='yhat_upper')
plt.plot(forecast['ds'].values, forecast['yhat_lower'].values,  color='blue', label='yhat_lower')
plt.plot(forecast['ds'].values, forecast['trend'].values,  color='black', label='trend')
plt.legend(loc='best')
plt.show()
plt.close(); plt.gcf().clear()

#Residual Analysis and various error terms
mae, rmse, mape = ResidualPlot(df_prophet['y'].values, forecast.loc[:df_prophet['y'].shape[0]-1,'yhat'].values, "./Images/residual_prophet.png")
print(''.join(['mean:', str(np.round(df_prophet['y'].mean(), 2)), ', mae:', str(mae),', rmse:', str(rmse),', mape:',str(mape)]))
#Durbin_Watson:0.55, mean:280.3, mae:17.36, rmse:22.45, mape:7.25

# Optional for future: Add holidays
holidays = pd.DataFrame([['Christmas','1949-12-25'], ['Christmas','1950-12-25']], columns = ['holiday', 'ds'])
holidays['ds'] =  pd.to_datetime(holidays['ds'], format='%Y-%m-%d')
#holidays['lower_window'] = 0
#holidays['upper_window'] = 0
holidays

# Start from here
#model = Prophet(growth='linear', yearly_seasonality = True, seasonality_prior_scale=0.5, holidays=holidays)

# add_regressor is for extra variables

# Class work: plot original, Sarimax pred and Prophet pred in one graph