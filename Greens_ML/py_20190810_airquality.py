import pandas as pd
import numpy as np

#File Reading
#1.	Read airquality.csv file. Note: you may have to use double slash for path (“data\\airquality.csv”) as ‘\a’ is an escape character.  
airquality = pd.read_csv("data/airquality.csv")
#2.	Understand about the data from airquality.pdf
airquality.dtypes

#Data Frame properties and quality check
#3.	How many rows in the data?
airquality.shape[0]

#4.	How many columns in the data?
airquality.shape[1]

#5.	What are the column names?
airquality.columns

#6.	How many null values in Ozone column (Note: nans are treated as nulls. There is a pandas function to catch nulls)
airquality["Ozone"].isnull().sum() # Option 1
airquality.shape[0] - airquality["Ozone"].count() # Option 2

#Data Frame slicing
#7.	Slice from airquality a dataframe which only has rows with valid entries 
  # for Solar.R. Remove rows which has null values in Solar.R column
aq_solarna_removed = airquality[airquality["Solar.R"].notnull()]
aq_clean = airquality.dropna() # any record with missing values will be removed
aq_solarna_removed = airquality.dropna(subset = ["Solar.R"]) # mentionign the columns to be checked for missing values

#8.	What is the average value of Ozone column?
airquality["Ozone"].mean()
np.mean(airquality["Ozone"])

airquality["Ozone"].median()
#np.median(airquality["Ozone"]) # numpy median doesn't work if there are nans
np.nanmedian(airquality["Ozone"]) # numpy median function which ignores nan


#9.	What is the average value of Solar.R on days with temperature 
   # above average temperature?
# SQL: selct average(Solar.R) from airquality where Temp > average(Temp)
avg_temp = airquality["Temp"].mean()
cond = airquality["Temp"] > avg_temp
airquality.loc[cond,"Solar.R"].mean()
# last 3 lines in 1 line
airquality.loc[airquality["Temp"] > airquality["Temp"].mean(),"Solar.R"].mean()

#10.Slice only records of 15th day of each month
# SQL: Select * from airquality where Day = 15
aq_day15 = airquality[airquality["Day"] == 15]

#11.Slice records of 6th and 8th month alone

# Option 1
cond1 = airquality["Month"] == 6
cond2 = airquality["Month"] == 8
aq_months_68 = airquality[cond1 | cond2]

# Option 2
aq_months_68 = airquality[airquality["Month"].isin([6,8])]


#12.What is the average ozone values of the days where both 
 # Solar.R and Temperature are above their averages?
# Select average(Ozone) from airquality 
  # where Solar.R > average(Solar.R) and Temp > average(Temp)
cond1 = airquality["Solar.R"] > airquality["Solar.R"].mean()
cond2 = airquality["Temp"] > airquality["Temp"].mean()
airquality.loc[cond1 & cond2, "Ozone"].mean()

#### For loop
#13.Calculate average values of Ozone, Solar, Wind and Temperature and save in a list/array/series
aq_avg_multicols = pd.Series([airquality["Ozone"].mean(),
                airquality["Solar.R"].mean(),
                airquality["Wind"].mean(),
                airquality["Temp"].mean()],
        index = airquality.columns[:4])
# Above approach is manual. Cannot be done if there are too many columns

# Using for - looping across multiple columns
cols_needed = airquality.columns[:4]
aq_avg_multicols = pd.Series(0.0, index = cols_needed) # creating dummy series
for i in cols_needed:
    aq_avg_multicols[i] = airquality[i].mean()
print(aq_avg_multicols)

#14.Calculate month-wise average Ozone and save in a list/array/series
airquality.loc[airquality["Month"] == 5,"Ozone"].mean()
airquality.loc[airquality["Month"] == 6,"Ozone"].mean()
airquality.loc[airquality["Month"] == 7,"Ozone"].mean()
airquality.loc[airquality["Month"] == 8,"Ozone"].mean()
airquality.loc[airquality["Month"] == 9,"Ozone"].mean()
# above approach is too manual

unique_months = airquality["Month"].unique()
monthwise_avg_ozone = pd.Series(0.0, index = unique_months) # dummy series
for i in unique_months:
    monthwise_avg_ozone[i] = airquality.loc[
            airquality["Month"] == i,"Ozone"].mean()
print(monthwise_avg_ozone)


#15.Calculate month-wise average Ozone, Solar, Wind and Temperature and 
 # save in a matrix/data frame

# So many lines of code if you have to do this without for loop
airquality.loc[airquality["Month"]== 5,"Ozone"].mean()
airquality.loc[airquality["Month"]== 6,"Ozone"].mean()
airquality.loc[airquality["Month"]== 7,"Ozone"].mean()
airquality.loc[airquality["Month"]== 8,"Ozone"].mean()
airquality.loc[airquality["Month"]== 9,"Ozone"].mean()
airquality.loc[airquality["Month"]== 5,"Solar.R"].mean()
airquality.loc[airquality["Month"]== 6,"Solar.R"].mean()
airquality.loc[airquality["Month"]== 7,"Solar.R"].mean()
airquality.loc[airquality["Month"]== 8,"Solar.R"].mean()
airquality.loc[airquality["Month"]== 9,"Solar.R"].mean()    
airquality.loc[airquality["Month"]== 5,"Wind"].mean()
airquality.loc[airquality["Month"]== 6,"Wind"].mean()
airquality.loc[airquality["Month"]== 7,"Wind"].mean()
airquality.loc[airquality["Month"]== 8,"Wind"].mean()
airquality.loc[airquality["Month"]== 9,"Wind"].mean()
airquality.loc[airquality["Month"]== 5,"Temp"].mean()
airquality.loc[airquality["Month"]== 6,"Temp"].mean()
airquality.loc[airquality["Month"]== 7,"Temp"].mean()
airquality.loc[airquality["Month"]== 8,"Temp"].mean()
airquality.loc[airquality["Month"]== 9,"Temp"].mean()  

unique_months = airquality["Month"].unique()
cols_needed = airquality.columns[:4]
monthwise_avg_multicols = pd.DataFrame(0.0,
                                       columns = cols_needed,
                                       index = unique_months)
for i in unique_months:
    for j in cols_needed:
        monthwise_avg_multicols.loc[i,j] = airquality.loc[
                airquality["Month"]== i,j].mean()  
print(monthwise_avg_multicols)

