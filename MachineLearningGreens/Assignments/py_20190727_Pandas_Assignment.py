# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:12:40 2019

@author: evkikum
"""

import pandas as pd
import numpy as np


df = pd.DataFrame({
        "A":np.arange(1,25) ,
        "B":np.arange(26,50),
        "C":np.arange(51,75),
        "D":np.arange(76,100)
        }, index = np.arange(1001,1025))


##Slice column ‘A’ from df and save it as a series ‘s’

s = df["A"]

##Slice column ‘A’ and column ‘C’ and save it as df2

## OPTION-1
df2 = df.loc[:,["A","C"]]

##OPTION-2
cols_needed = ["A","C"]
df2 = df[cols_needed]

## Slice 0th and 2nd column using column number and save it as df3

df3 = df.iloc[:,[0,2]]

##Slice from 0 till 5th position in series ‘s’
type(s)
s[1:5]

## Slice all columns from rows 3 till 19 and save it as df4

df4 = df.iloc[3:19,:]

## Create df5 which has subset of data from df where column A values are above median of column A. Note: slice entire columns based on 
##condition on column A

df5 = df[df["A"] > np.median(df["A"])]

