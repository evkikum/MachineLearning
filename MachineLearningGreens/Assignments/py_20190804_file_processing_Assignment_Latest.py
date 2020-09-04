# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:21:17 2019

@author: evkikum
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 09:56:47 2019

@author: evkikum
"""



import os
import pandas as pd
import numpy as np


os.getcwd()
os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\GreenInstitute_Course\Assignments")

air = pd.read_csv("airquality.csv")
air.dtypes



#How many rows in the data?
air.shape[0]


##How many columns in the data?
air.shape[1]

##What are the column names?
air.columns

##How many null values in Ozone column

#OPTION-1
sum(air['Ozone'].isnull())  ## 37 

#OPTION-1 -- THE BELOW IS PANDAS SUM
air["Ozone"].isnull().sum()

##Slice from airquality a dataframe which only has rows with valid entries for Solar.R. Remove rows which has null values in Solar.R column

##option-1

air.dtypes
air.shape[0]
sum(air["Solar.R"].notnull())
sum(air["Solar.R"].isnull())
df_Solar = air.loc[air["Solar.R"].notnull(),:]


##option-2 
df = air.dropna()  # any record with missing values
df_solar = air.dropna(subset = ["Solar.R"])

##What is the average value of Ozone column?

##OPTION-1
air["Ozone"].mean() 
df_Solar["Ozone"].mean() 


##OPTION-2
np.mean(air["Ozone"])

np.median(air["Ozone"])
np.nanmedian(air["Ozone"])

## What is the average value of Solar.R on days with temperature above average temperature?

sum(df_Solar["Solar.R"] > df_Solar["Solar.R"].mean())
df_Solar.loc[df_Solar["Solar.R"] > df_Solar["Solar.R"].mean(),"Solar.R"].mean()

air.loc[air["Solar.R"] > air["Solar.R"].mean(),"Solar.R"].mean()

##Slice only records of 15th day of each month

air.dtypes
air.loc[air["Day"] == 15,:]
df_Solar.loc[df_Solar["Day"] == 15,:]

##Slice records of 6th and 8th month alone
##OPTION 1 
df = air.loc[(air["Month"] == 6) | (air["Month"] == 8),:]

##option2 
include_values = [6,8]

df = air.loc[air["Month"] in include_values, :]
df = air.loc[air["Month"].isin([6,8])]

## What is the average ozone values of the days where both Solar.R and Temperature are above their averages?
air.dtypes
df1 = air.loc[(air["Solar.R"] > air["Solar.R"].mean()) & (air["Temp"] > air["Temp"].mean()),["Ozone","Day"]]
df1.groupby('Day')['Ozone'].mean()

df1 = df_Solar.loc[(df_Solar["Solar.R"] > df_Solar["Solar.R"].mean()) & (df_Solar["Temp"] > df_Solar["Temp"].mean()),["Ozone","Day"]]
df1.groupby('Day')['Ozone'].mean()



## Calculate average values of Ozone, Solar, Wind and Temperature and save in a list/array/series
##save in list
include_values = [air["Ozone"].mean(), air["Solar.R"].mean(), air["Wind"].mean(), air["Temp"].mean()]
l1 = []

for i in include_values:
    l1.append(i)

print(l1)

#save in array

np_array = np.array(include_values)
print(np_array)

##save in series 
pd_series = pd.Series(include_values)
print(pd_series)

## Calculate month-wise average Ozone and save in a list/array/series

##option1 
df2 = air.groupby("Month")["Ozone"].mean()
print(df2)

##option2
df2 = air.groupby("Month", as_index = False)["Ozone"].mean()

## save in list
l1 = []

for i in df2["Ozone"]:
    l1.append(i)
        
print(l1)


## save in array 
np_array = np.array(df2["Ozone"])
print(np_array)
type(np_array)

##save in series
pd_series = pd.Series(df2["Ozone"])
print(pd_series)



##Calculate month-wise average Ozone, Solar, Wind and Temperature and save in a matrix/data

##OPTION1
air.dtypes
df3 = air.groupby("Month", as_index = False)["Ozone", "Solar.R", "Wind", "Temp"].mean()

np_matrix = np.column_stack([df3["Ozone"], df3["Solar.R"], df3["Wind"], df3["Temp"]])
type(np_matrix)


##OPTION2
air.groupby("Month").agg(np.mean)

