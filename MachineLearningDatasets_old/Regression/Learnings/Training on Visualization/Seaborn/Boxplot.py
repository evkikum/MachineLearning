# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:11:57 2019

@author: evkikum
"""


import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
iris = sns.load_dataset('iris')

sns.boxplot(x=tips["size"])

sns.boxplot(x='sex',y='total_bill', data = tips )
sns.boxplot(x='day', y='total_bill', data = tips)
sns.boxplot(x='day', y='total_bill', data = tips, hue='sex')

sns.boxplot(x='day', y='total_bill', data = tips, hue='sex', palette='spring')

sns.boxplot(x='day', y='total_bill', data = tips, hue='smoker', palette='coolwarm')

sns.boxplot(x='day', y='total_bill', data = tips, hue='time')

sns.boxplot(x='day', y='total_bill', data = tips, hue='time', order=['Sun','Sat', 'Fri', 'Thur'])

sns.boxplot(x='sex', y='tip', data = tips, order=['Female', 'Male'])


####PLOTTING ON IRIS
sns.boxplot(data=iris)

sns.boxplot(data=iris, palette='coolwarm')
sns.boxplot(data=iris, palette='coolwarm', orient='horizontal')  ## horizontal or h will work
sns.boxplot(data=iris, palette='coolwarm', orient='v')  ## Vertical or v will work


## Ensure both bolxplot/swarmplot run together
sns.boxplot(x='day', y='total_bill', data = tips, palette='husl')
sns.swarmplot(x='day', y='total_bill', data = tips)





