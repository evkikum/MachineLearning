# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:02:44 2019

@author: evkikum
"""


import numpy as np

br_yr = [1986, 1989, 1975, 1981, 1978]


##age using reguler loops

age = []
yrs = 0

for i in br_yr:
    yrs = 2017 - i
    age.append(yrs)
    
print(age)

## age using list comprehension

age = [(2017 - i) for i in br_yr]

## age using numpy operation.

np_br_yr = np.array(br_yr)
2017 - np_br_yr




str_msg = "this is a python excercise which is neither too easy nor too hadr to be solved in the given amount of time"

## using regular loops

split_str = []

for i in str_msg.split():
    if (i == "is" or i == "a" or i == "the"):
        continue
    else:
        split_str.append(i)

print(split_str)

## List comprehension

split_str = [i for i in str_msg.split() if (i != "is" and i != "a" and i != "the")]

## Using numpy

split_str = np.array(str_msg.split())
cond1 = (split_str != 'is')
split_str = split_str[cond1]
cond2 = (split_str != 'a')
split_str = split_str[cond2]
cond3 = (split_str != 'the')
split_str = split_str[cond3]



