# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 15:26:50 2019

@author: evkikum
"""


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

os.getcwd()
os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\GreenInstitute_Course")

airquality = pd.read_csv("Data/airquality.csv")


###Airquality


####1. Get the histogram distribution of Solar.R

airclean = airquality.dropna(subset = ["Ozone","Solar.R" ])

##OPTION1
plt.figure()
plt.hist(airclean["Solar.R"])
plt.xlabel("Solar R Values")
plt.ylabel("Count of Solar R")
plt.title("Distribution of Solar vs count")


## option2
airclean["Solar.R"].plot.hist()

##Option3 
pd.DataFrame.hist(airclean, column = "Solar.R")

####2. Get the boxplot distribution of temperature

##option1
plt.boxplot(airclean["Temp"])


##option2
airclean["Temp"].plot.box()

##option2
airclean.boxplot(column = "Temp")

####3. Generate a scatter plot between temperature and solar.R
##option1
plt.scatter(airclean["Temp"], airclean["Solar.R"], s = 10, c = "red")

##option2 
airclean.plot.scatter("Temp", "Solar.R")

####4. Generate a line plot of Solar.R. Create a date column and use that as x axis in line plot. Note:
####Date and time available in data frame. You may have to refer the documentation of data to
####know the year.

plt.plot(airclean["Solar.R"])

pd.date_range(start = "2017-08-01", periods = 153, freq = "D")
pd.date_range(start = "2017-08-01", periods = 145, freq = "D")

airclean.index =  pd.date_range(start = "2017-08-01", periods = 111, freq = "D")
airclean["Solar.R"].plot.line()



###Mtcars
mtcars = pd.read_csv("Data/mtcars.csv")

###1. Compare the mpg boxplot distribution of automatic vs manual transmission cars

mtcars.boxplot(column = "mpg", by = "am")

###2. Compare the boxplot distribution of mpg of cars by gears and transmission. One mpg
###distribution box per gear-am combination

mtcars.boxplot(column = "mpg", by = ["gear", "am"])

###3. Generate a scatter plot between mpg and weight of the car

plt.scatter(mtcars["mpg"], mtcars["wt"])

