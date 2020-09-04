import pandas as pd
import numpy as np
#File Reading
#1.	Read mtcars.csv file.
mtcars = pd.read_csv("data/mtcars.csv")
#2.	Understand about the data from mtcars.pdf

#Data Frame properties and quality check
#3.	How many rows in the data?
mtcars.shape[0]
#4.	How many columns in the data?
mtcars.shape[1]
#5.	What are the column names?
mtcars.columns
#6.	Use describe command to understand the statistical summary. 
mtcars_stats_summary = mtcars.describe()

# Data Frame slicing
#7.	Average miles per gallon (mpg) of all cars
mtcars["mpg"].mean()
mtcars["mpg"].median()
np.percentile(mtcars["mpg"],50) # same as median; 50% of the population below this value
np.percentile(mtcars["mpg"],25) # 25% of population below this value
np.percentile(mtcars["mpg"],75) # 75% of population below this value

#8.	Average mpg of automatic transmission cars
mtcars.loc[mtcars["am"] == 0,"mpg"].mean()

#9.	Average mpg of manual transmission cars
mtcars.loc[mtcars["am"] == 1,"mpg"].mean()

#10.	Average Displacement of cars with 4 gears
#11.	Average Horse power of cars with 3 carb

#12.	Average mpg of automatic cars with 4 gears
cond1 = mtcars["am"] == 0
cond2 = mtcars["gear"] == 4
mtcars.loc[cond1 & cond2, "mpg"].mean()

#13.	Average qsec of cars with mpg above average mpg and weight below average weight
cond1 = mtcars["mpg"] > mtcars["mpg"].mean()
cond2 = mtcars["wt"] < mtcars["wt"].mean()
mtcars.loc[cond1 & cond2, "qsec"].mean()

#14.	Entire row of the vehicle which has the highest miles per gallon
mtcars[mtcars["mpg"] == mtcars["mpg"].max()]
mtcars.loc[mtcars["mpg"].idxmax(),:]

#15.	Entire row of vehicle with the highest horsepower
#16.	Mileage and hp of car with highest weight
#17.	Calculate ratio of mpg to carb for each car and calculate the average of ratio
(mtcars["mpg"]/mtcars["carb"]).mean()

#18.	Weight of the car with the minimum displacement
#19.	Slice all columns of 3 gear cars
#20.	Slice mpg, displacement and hp columns of manual transmission cars
mtcars.loc[mtcars["am"] == 1,["mpg","disp","hp"]]

#For loops
#21.	What is 
#a.	average mpg for 3 gear cars
#b.	average mpg for 4 gear cars 
#c.	average mpg for 5 gear cars
#      Save result in a list/array/series
mtcars.loc[mtcars["gear"] == 3,"mpg"].mean()
mtcars.loc[mtcars["gear"] == 4,"mpg"].mean()
mtcars.loc[mtcars["gear"] == 5,"mpg"].mean()

unique_gears = np.unique(mtcars["gear"])
gearwise_avg_mpg = pd.Series(0.0, index = unique_gears)
for i in unique_gears:
    gearwise_avg_mpg[i] = mtcars.loc[mtcars["gear"] == i,"mpg"].mean()
print(gearwise_avg_mpg)

#22.	What is 
#a.	average hp, average wt, average sec, average vs for 3 gear cars
#b.	average hp, average wt, average sec, average vs for 4 gear cars 
#c.	average hp, average wt, average sec, average vs for 5 gear cars 
#Save list in a matrix/data frame
cols_needed = ["hp","wt","qsec","vs"]
gearwise_avg_multi_cols = pd.DataFrame(0.0, index = unique_gears,
                                       columns = cols_needed)
for i in unique_gears:
    for j in cols_needed:
        gearwise_avg_multi_cols.loc[i,j] = mtcars.loc[
                mtcars["gear"] == i,j].mean()
print(gearwise_avg_multi_cols)


#Apply Aggregate
#23.	Solve 21 without for loop
mtcars.groupby("gear")["mpg"].agg(np.mean)
#24.	Solve 22 without for loop
mtcars.groupby("gear")[cols_needed].agg(np.mean)

#25.	average hp, median and average wt, average vs for different gear-transmission combinations
mtcars.groupby(["gear","am"]).agg({"hp": np.median,
                "wt": [np.mean, np.median],
                "vs": np.mean})
