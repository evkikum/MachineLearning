# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:51:21 2019

@author: evkikum
"""



import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt


tips = sns.load_dataset('tips')
sns.barplot(x='day', y="tip", data = tips)
sns.barplot(x='day', y="total_bill", data=tips)

sns.barplot(x='day',y='tip', data = tips, hue = 'sex')
sns.barplot(x='day',y='tip', data = tips, hue = 'sex', palette='winter_r')
sns.barplot(x='day', y='total_bill', data = tips, hue='smoker')

sns.barplot(x='day', y='total_bill', data = tips, palette='spring')
sns.barplot(x='total_bill', y='day', data = tips)
## ORDER CHANGE
sns.barplot(x='day', y='tip', data = tips, order=["Sat", 'Fri', 'Sun', "Thur"])


## THE BELOW IS NOT WORKING
sns.barplot(x='day', y='total_bill', data = tips)
sns.barplot(x='day', y='total_bill', data = tips, estimator=np.median)
sns.barplot(x='day', y='total_bill', data = tips, estimator=np.median, palette='spring')
sns.barplot(x='smoker', y='tip', data=tips, hue='sex', palette='coolwarm',estimator=np.median)
sns.barplot(x='smoker', y='tip', data=tips, hue='sex', palette='coolwarm',estimator=np.mean)
sns.barplot(x='smoker', y='tip', data=tips, hue='sex', palette='coolwarm',estimator=np.max)
    

sns.barplot(x='smoker', y='tip', data = tips)
sns.barplot(x='smoker', y='tip', data = tips, ci=99)
sns.barplot(x='smoker', y='tip', data = tips, ci=50)
sns.barplot(x='smoker', y='tip', data = tips, ci=30)

sns.barplot(x='smoker', y='tip', data=tips,ci=34,palette='winter_r')
sns.barplot(x='smoker', y='tip', data=tips,ci=34,palette='winter_r', estimator=np.median)

sns.barplot(x='day', y='total_bill', data = tips, palette='spring', capsize=0.3)

sns.barplot(x='day', y='total_bill', data = tips,hue="sex", palette='spring', capsize=0.1)



