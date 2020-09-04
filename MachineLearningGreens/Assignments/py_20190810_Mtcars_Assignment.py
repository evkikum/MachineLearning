# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 20:25:37 2019

@author: evkikum
"""

import os
import pandas as pd
import numpy as np



os.getcwd()
os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\GreenInstitute_Course\Data")

mtcars = pd.read_csv("mtcars.csv")


##3) How many rows in the data?
mtcars.shape[0]

##4) How many columns in the data?
mtcars.shape[1]

##5) What are the column names?
mtcars.columns

##6) Use describe command to understand the statistical summary.
mtcars.describe()


##7) Average miles per gallon (mpg) of all cars
mtcars["mpg"].mean()
mtcars["mpg"].median()
np.percentile(mtcars["mpg"],50)  ## same as median
np.percentile(mtcars["mpg"],25)  ## 25% of population below this value
np.percentile(mtcars["mpg"],75)  ## 75% of population below this value


##8) Average mpg of automatic transmission cars
mtcars.loc[mtcars["am"] == 0,"mpg"].mean()

##9) Average mpg of manual transmission cars
mtcars.loc[mtcars["am"] == 1,"mpg"].mean()

##10) Average Displacement of cars with 4 gears

mtcars.loc[mtcars["gear"] == 4,"disp"].mean()

## 11) Average Horse power of cars with 3 carb
mtcars.loc[mtcars["carb"] == 3,"hp"].mean()

##12) Average mpg of automatic cars with 4 gears
cond1 = mtcars["am"] == 1
cond2 = mtcars["gear"] == 4
mtcars.loc[cond1 & cond2 ,"mpg"].mean()

##13) Average qsec of cars with mpg above average mpg and weight below average weight
cond1 = mtcars["mpg"] > mtcars["mpg"].mean()
cond2 = mtcars["wt"] < mtcars["wt"].mean()
mtcars.loc[cond1 & cond2, "qsec"].mean()


##14) Entire row of the vehicle which has the highest miles per gallon
mtcars_sorted = mtcars.sort_values("mpg", ascending = False)
mtcars_sorted.iloc[0,:]

##15) Entire row of vehicle with the highest horsepower
mtcars_sorted_1 = mtcars.sort_values("hp", ascending = False)
mtcars_sorted_1.iloc[0,:]

##16) Mileage and hp of car with highest weight
##OPTION1
mtcars_sorted_2 = mtcars.sort_values("wt", ascending = False)
mtcars_sorted_2.iloc[0,:]

##option2
cond1 = mtcars["wt"] == mtcars["wt"].max()
mtcars.loc[cond1,["mpg", "hp"]]

## 17) Calculate ratio of mpg to carb for each car and calculate the average of ratio
mtcars["ratio_mpg_carb"] = mtcars["mpg"]/mtcars["carb"]
mtcars["ratio_mpg_carb"].mean()

##18) Weight of the car with the minimum displacement
cond1 = mtcars["disp"] == mtcars["disp"].min()
mtcars.loc[cond1, "wt"]

##19) Slice all columns of 3 gear cars
cond1 = mtcars["gear"] == 3

mtcars.loc[cond1,:]

## 20) Slice mpg, displacement and hp columns of manual transmission cars

cond1 = mtcars["am"] == 0
mtcars.loc[cond1, ["mpg", "disp", "hp"]]

##21) What is
## a. average mpg for 3 gear cars
## b. average mpg for 4 gear cars
## c. average mpg for 5 gear cars
## Save result in a list/array/series

##OPTION1
df_gp_gear = mtcars.groupby("gear")
df_avg_mpg = df_gp_gear["mpg"].agg(np.mean)
df_avg_mpg_array = np.array(df_avg_mpg)

##option2
avg_mpg_per_gear = pd.Series(0.0, index = mtcars["gear"].unique())

for i in mtcars["gear"].unique():
    avg_mpg_per_gear[i] = mtcars.loc[mtcars["gear"] == i, "mpg"].mean()



##22) What is
##a. average hp, average wt, average sec, average vs for 3 gear cars
##b. average hp, average wt, average sec, average vs for 4 gear cars
##c. average hp, average wt, average sec, average vs for 5 gear cars
##Save list in a matrix/data frame

##OPTION1
df_gp_gear = mtcars.groupby("gear")
df_avg_hpwtsecvs = df_gp_gear['hp',"wt","qsec", "vs"].agg(np.mean)
type(df_avg_hpwtsecvs)

##option2



##25) average hp, median and average wt, average vs for different gear-transmission combinations

mtcars["gear"].unique()

df_gp_gear = mtcars.groupby(["gear", "am"])

df_gp_gear.agg({
                "hp" : np.mean,
                "wt" : [np.mean, np.median],
                "vs" : np.mean
                })


