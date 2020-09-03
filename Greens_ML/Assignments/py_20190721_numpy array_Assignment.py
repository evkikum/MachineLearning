# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:20:03 2019

@author: evkikum
"""


import numpy as np
import pandas as pd


xVec = np.array([42,85,84,23,11,55,14,96,13,30])
yVec = np.array([13,8,85,71, 1,7,55, 2,34,24])


## Subset xVec with values greater than 60
xVec[xVec > 60]

## Subset yVec with values less than mean of yVec
yVec[yVec < np.mean(yVec)]

## How many odd numbers in xVec?
len(xVec[xVec%2 != 0])

## Subset values in yVec which are between minimum and maximum values of xVec (yes,xVec)
min = np.min(xVec)
max = np.max(xVec)

yVec[(yVec > np.min(xVec))  & (yVec < np.max(xVec))]



###Date 07/21/2019

math_score_array = np.array([95,67,88,45,84])
eng_score_array = np.array([78,67,45,39,67])
gender_array = np.array(["M","M","F","M","F"])


# Create a data frame (score_df) with above 3 arrays as columns
# Add "R1001","R1002",...."R1005" as row indexes
# Add "Maths","English","Gender" as column indexes

score_df = pd.DataFrame({"Maths" : math_score_array, "English" : eng_score_array , "Gender": gender_array}, index = ["R1001", "R1002", "R1003", "R1004", "R1005"])

# Slice the following
# Maths column
score_df["Maths"]

# Maths and English 
score_df.loc[:, ["English", "Maths"]]

# "Maths" column of "R1001"
score_df.loc["R1001", "Maths"]

# "Maths" and English column values of "R1001" and "R1003"
score_df.loc[["R1001", "R1003"], ["Maths", "English"]]

# All rows, 2nd column
score_df.loc[:,"English"]

# 0th and 3rd row, 0th and 1st column
score_df.iloc[[0,2], [0,1]]

# data frame of Male students alone
cond = score_df["Gender"] == "M"
score_df[cond]

# english and maths score of Male students
score_df_v2 = score_df[score_df["Gender"] == "M"]
score_df_v2.loc[:,["Maths", "English"]]

# all columns of students who score above 70 in Maths
score_df[score_df["Maths"] > 70]

# average maths core of students who got above 60 in English
score_df[score_df["Maths"] > 60].mean()

# average english score of students who are above average in maths

df = score_df["English"]
df[df > score_df["Maths"].mean()]


# all columns of male students who scores above 60 in maths

score_df[(score_df["Maths"] > 60) & (score_df["Gender"] == "M")]

