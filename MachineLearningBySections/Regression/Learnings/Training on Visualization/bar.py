# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:57:22 2019

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
irisdata.info()
irisdata_stats = irisdata.describe()

iris_class_summary = irisdata.groupby("Species").agg(np.mean)

for i in iris_class_summary.columns:
    plt.figure()
    iris_class_summary[i].plot.bar()
    
iris_class_summary["Sepal.Length"].plot.bar()




###================================================================
company = ["GOOGL","AMZN", "MSFT", "FB"]
revenue = [90,136, 89,27]
profit=[40,2,34,12]
xpos = np.arange(len(company))

## THE BELOW WILL HELP PLOT BOTH REVENUE/PROFIT IN SAME BAR
plt.xticks(xpos, company)
plt.xlabel("Company")
plt.ylabel("revenue")
plt.bar(xpos, revenue,label="Revenue generated in $")
plt.bar(xpos, profit, label="Profit in $")
plt.legend()

##Alternatively

plt.bar(company, revenue,label="Revenue generated in $")
plt.bar(company, profit, label="Profit in $")
plt.legend()




## THE BELOW WILL HELP PLOT BOTH REVENUE/PROFIT IN ADJUSTED BARS
plt.xticks(xpos, company)
plt.xlabel("Company")
plt.ylabel("revenue")
plt.bar(xpos-0.2, revenue,width=0.4,label="Revenue generated in $")
plt.bar(xpos+0.2 , profit,width=0.4, label="Profit in $")
plt.legend()


## Horizontal view

plt.yticks(xpos, company )
plt.title("USA Tech Stocks")
plt.barh(xpos-0.2, revenue, label="Revenue")
plt.barh(xpos+0.2, profit, label="Profit")
plt.legend()
