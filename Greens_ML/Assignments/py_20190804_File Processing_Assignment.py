# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 11:07:16 2019

@author: evkikum
"""

import pandas as pd
import os

print("Hello")

print("Hell\no")
print("He\tllo")

print(r"Hell\no")

os.getcwd()
os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\GreenInstitute_Course\Data")
os.getcwd()

acs2013 = pd.read_csv("ACS_13_5YR_S1903.csv")
acs2008 = pd.read_csv("ACS_08_3YR_S1903.csv")

acs2013.dtypes


acs2013 = pd.read_csv("ACS_13_5YR_S1903.csv", dtype = {"GEO.id2":str})

# Q1. slice the first 7 columns and save as acs_2013_s

acs_2013_s = acs2013.iloc[:,0:7]
# Q2. rename the column names as follows
#["ID","FIPS","State",
#                    "Total Household", "Total Household MOE",
#                    "Income","Income MOE"]

acs_2013_s.columns = ["ID","FIPS","State","Total Household", "Total Household MOE","Income","Income MOE"]


# Q3. calculate average income of US
acs2013["Income"].mean()


# Q4. what is the maximum income and which state is that?

acs_2013_s[acs_2013_s["Income"] == acs_2013_s["Income"].max()]["State"]

# Q5. what is the minimum income and which state is that?
acs_2013_s.loc[acs_2013_s["Income"] == acs_2013_s["Income"].min(),"State"]

# Q6. get the list of states which are above average in household income

acs_2013_s.loc[acs_2013_s["Income"] > acs_2013_s["Income"].mean(),"Income"]

# Q7. get the income of texas state

acs_2013_s.loc[acs_2013_s["State"] == "Texas",["Income"]]

# Q8. what is the state which has the 2nd highest income
df4 = acs_2013_s.loc[acs_2013_s["Income"] < acs_2013_s["Income"].max() ,["State","Income"]].sort_values("Income", ascending = False)
                  
            
            