# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 10:53:50 2020

@author: evkikum
Below is the URL that I emulated the below pgm;
https://www.youtube.com/watch?v=1C67EYcoxvM&list=PLubVnfIHABZQZXwT7Nx3M4Joixwe3V824

"""

import pylab
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
import os


os.chdir(r"C:\Users\evkikum\OneDrive - Ericsson AB\Python Scripts\GreenInstitute_Course")

df = pd.read_csv("Data/ambient_temperature_system_failure.csv")
df.info()

df.head()

value_final = df['value']
data = df['value']


## UNIVARATE ANALYSIS
## Create box plot to display univarate outliers on df['value']
## Get quantile values and IQR for outliers 
qv1 = data.quantile(0.25)
qv2 = data.quantile(0.50)
qv3 = data.quantile(0.75)

qv_limit = 1.5 * (qv3 - qv1)

## GET POSITION OF OUTLIERS AND USE df['value'] for labels
un_outliers_mask = (data > qv3 + qv_limit) | (data < qv1 - qv_limit)
un_outliers_data = data[un_outliers_mask]

## BELOW FLAG AND BOLXPLOT SHOWS THAT THERE ARE LOT ANOMALIES/OUTLIERS IN DATAFRAME IN value field.
df['Outlier_flag'] = un_outliers_mask
plt.boxplot(data)


## BIVARATE ANALYSIS



##below is used for multivariate outliers
ocsvm = OneClassSVM(nu = 0.25, gamma=0.05)
ocsvm.fit(data)

df_ts = df['timestamp']

df_ts[ocsvm.predict(data) == -1]






