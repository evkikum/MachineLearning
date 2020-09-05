# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:17:52 2019

@author: evkikum
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.model_selection import train_test_split, cross_val_score # latest version of sklearn
from sklearn.cross_validation import train_test_split, cross_val_score # older version of sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os

os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\GreenInstitute_Course")

irisdata = pd.read_csv("data/iris.csv")
irisdata_stats = irisdata.describe()



## OPTION1 
for i in irisdata.columns[:4]:
    irisdata.boxplot(column = i, by = "Species")
    

## OPTION2
irisdata.boxplot(column = "Sepal.Length") 
irisdata.boxplot(column = "Sepal.Length", by = "Species")


    

