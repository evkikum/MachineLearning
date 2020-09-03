# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 16:42:09 2019

@author: evkikum
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\GreenInstitute_Course")

##1) Read the csv file and check the data types. Note that certain columns has numbers with
##commas in between which might have been read as a non-numeric data type. You can't
##just convert the data type; it will then have junk numbers. You have to remove commas.


ssamatab1 = pd.read_csv("Data/ssamatab1.csv")
ssamatab1.dtypes

ssamatab1["Civilian Labor Force"] = ssamatab1["Civilian Labor Force"].str.replace(",","")
ssamatab1["Employment"] = ssamatab1["Employment"].str.replace(",","")
ssamatab1["Unemployment"] = ssamatab1["Unemployment"].str.replace(",","")

ssamatab1["Civilian Labor Force"] = pd.to_numeric(ssamatab1["Civilian Labor Force"])
ssamatab1["Employment"] = pd.to_numeric(ssamatab1["Employment"])
ssamatab1["Unemployment"] = pd.to_numeric(ssamatab1["Unemployment"])


##2. Which Area had the highest unemployment rate in December 2015?

cond1 = (ssamatab1["Year"] == 2015) & (ssamatab1["Month"] == 12)
df_Dec2015 = ssamatab1.loc[cond1,:]
df_Dec2015.loc[df_Dec2015["Unemployment"] == df_Dec2015["Unemployment"].max(),"Area"]


##3. Which area had the highest ever unemployment rate and when did that happen?

ssamatab1.loc[ssamatab1["Unemployment"] == ssamatab1["Unemployment"].max(),["Area", "Year", "Month"]]

##4. Which state had the highest ever unemployment rate and when did that happen?

ssamatab1[['City', 'State']] = ssamatab1["Area"].str.split(",", expand=True,)
ssamatab1["State"] = ssamatab1["State"].str.lstrip()
ssamatab1["State"] = ssamatab1["State"].str.rstrip()
ssamatab1.loc[ssamatab1["Unemployment"] == ssamatab1["Unemployment"].max(),"State"]

##5. Obtain Yearly Unemployment rate by aggregating the data. One way would be to take
##average of unemployment rate column directly. But that's not mathematically right. You
##need to sum up the Unemployed and Civilian labor force by Year and then calculate the
##ratio for calculation of Unemployment rate

df_unemploy_ratio =  ssamatab1.groupby("Year")["Unemployment", "Civilian Labor Force"].agg(sum)
df_unemploy_ratio["unemploy_ratio"] = df_unemploy_ratio["Unemployment"]/df_unemploy_ratio["Civilian Labor Force"]




##6. Repeat a similar aggregation as previous point for State Level unemployment rate

df_state_unemployratio =  ssamatab1.groupby("State")["Unemployment", "Civilian Labor Force"].agg(sum).reset_index()
df_state_unemployratio["unemploy_ratio"] = df_state_unemployratio["Unemployment"]/df_state_unemployratio["Civilian Labor Force"]


##7. Plot the histogram and boxplot of unemployment rate

##Unemployment rate per year
plt.figure()
plt.hist(df_unemploy_ratio["unemploy_ratio"])
plt.xlabel("Unemployee ratio")
plt.ylabel("Count ")
plt.ylim([0,7])
plt.title("Yearly unemployee rate info ")

plt.boxplot(df_unemploy_ratio["unemploy_ratio"])
df_unemploy_ratio.boxplot(column = "unemploy_ratio")


##Unemployment rate per state
plt.figure()
plt.hist(df_state_unemployratio["unemploy_ratio"])
plt.xlabel("Unemployee ratio")
plt.ylabel("Count")
plt.ylim([0,40])
plt.title("State level unemployee rate info")

plt.boxplot(df_state_unemployratio["unemploy_ratio"])
df_state_unemployratio.boxplot(column = "unemploy_ratio")


##8. Compare the boxplot distribution of unemployment rate between top 4 states with highest
##civilian labor force

df_civilLaborForces =  ssamatab1.groupby("State")["Civilian Labor Force"].agg(np.median).reset_index()
df_civilLaborForces = df_civilLaborForces.sort_values("Civilian Labor Force", ascending = False).head(4)

ssamatab1["unemploy_ratio"] = ssamatab1["Unemployment"]/ssamatab1["Civilian Labor Force"]
df2 = ssamatab1.loc[np.in1d(ssamatab1["State"], df_civilLaborForces["State"]),: ]
df2.boxplot(column = "unemploy_ratio", by = "State" )


##9. Visualize the relationship between civilian labor force and unemployment rate using
##scatter plot

##OPTION 1
plt.figure()
plt.scatter(ssamatab1["Civilian Labor Force"], ssamatab1["unemploy_ratio"])
plt.xlabel("Civilian Labor Force")
plt.ylabel("Unemployment")
plt.title("Civilian Labor Force / Unemployment")


##option2 

ssamatab1.plot.scatter("Civilian Labor Force", "unemploy_ratio")

##10. Draw line plot of yearly unemployment rate of US (Year in xaxis and unemployment rate
##of US in yaxis)

ssamatab1_v2 = ssamatab1
ssamatab1_v2.index = ssamatab1["Year"]
ssamatab1_v2["unemploy_ratio"].plot.line()