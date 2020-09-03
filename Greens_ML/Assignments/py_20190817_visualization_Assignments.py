import pandas as pd
import numpy as np
import os
import warnings
import math

os.getcwd()
os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\GreenInstitute_Course")

wg = pd.read_csv("data/wg.csv")
############### EXPLORATORY ANALYSIS AND DATA QUALITY CHECKS #############
#1. number of nans in wg column

##option1 
wg["wg"].isnull().sum()

##option2
wg.shape[0] - wg["wg"].count()


#2. number of nans in metmin

##option1 
wg["metmin"].isnull().sum()

##option2 
wg.shape[0] - wg["metmin"].count()

#3. how many observations had a null either in wg or in metmin column?

132/134

#4. extract the observations which don't have null in "wg" as well as "metmin" columns

wgclean = wgdata.dropna(subset = ["wg", "metmin"])




#5. how many people have gained weight above average weight gain of the data

cond1 = wg["wg"] > wg["wg"].mean() 
df1_wg =  wg.loc[cond1, :]
df1_wg.shape[0]


#6. what is the gender of the person with the highest weight gain?
cond1 = wg["wg"] == wg["wg"].max()
wg.loc[cond1, "Gender"]


#6a. In this data, who gained more weight, male or female?

Female

#7. get the count of above avg wg people by shift

df_abv_avg_wg = wg.loc[wg["wg"] > wg["wg"].mean(), :]
wg_gp_df =  df_abv_avg_wg.groupby("Shift")
wg_gp_df["Shift"].count()


#8. percentage of male in each shift

wg_male = wg.loc[wg["Gender"] == "M",:]

total_male_count = wg_male.shape[0]

wg_gp_shift =  wg_male.groupby("Shift")

per_male_function = lambda x, y : (x/y) * 100

per_male_function(wg_gp_shift["Gender"].count(), total_male_count)


#9. Create a copy of the data frame and impute missing values with mean of the column

##option1        
count_rows = wg.shape[0]
count_columns = wg.shape[1]

for i in range(0,count_columns):    
    if wg.iloc[:,i].isnull().sum() > 0:
        for j in range(0, count_rows):
            if math.isnan(wg.iloc[j,i]):
                wg.iloc[j,i] = np.mean(wg.iloc[:,i])
    
