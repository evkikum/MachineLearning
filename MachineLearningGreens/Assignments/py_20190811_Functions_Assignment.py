# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:50:28 2019

@author: evkikum
"""

import os
import numpy as np
import pandas as pd
import warnings



##Write a function which accepts a list/array/series as input and returns the difference between mean and median
mean_median_diff = lambda a : np.mean(a) - np.median(a)


## Write a function (max_var2_corresponding) which accepts a data frame (df) as input along with 2 column names (var1, var2) in the data frame. Calculate the maximum value
## in var1 column of df. Return the value of var2 corresponding to maximum value of var1

max_var2_corresponding = lambda df, var1, var2 : df.loc[df[var1] == df[var1].max(), var2]
    
##a Test Case 1:
    
math_score_array = np.array([95,67,88,45,84])
eng_score_array = np.array([78,67,45,39,67])
gender_array = np.array(["M","M","F","M","F"])

score_df = pd.DataFrame({
'Maths':math_score_array,
'English':eng_score_array,
'Gender':gender_array})
score_df.index = ["R1001","R1002","R1003","R1004","R1005"]

max_var2_corresponding(score_df, "Maths","English")  ## 78

## 

emp_details_dict = {
'Age': [25,32,28],
'Income': [1000,1600,1400]
}

emp_details = pd.DataFrame(emp_details_dict)
emp_details.index = ['Ram','Raj','Ravi']


max_var2_corresponding(emp_details,"Income","Age")  ## 32