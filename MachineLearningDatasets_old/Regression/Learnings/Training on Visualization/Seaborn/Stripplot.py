# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:14:32 2019

@author: evkikum
"""

import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

%matplotlib inline

tips = sns.load_dataset('tips')
iris = sns.load_dataset('iris')

sns.stripplot(x=tips['tip'], color = 'green')

sns.stripplot(x=tips['total_bill'])

sns.stripplot(x='day',y='total_bill' , data=tips)
sns.stripplot(x='day',y='total_bill', data = tips, jitter=True)

sns.stripplot(x='day',y='total_bill', data = tips, jitter=0.2)
sns.stripplot(x='day',y='total_bill', data = tips, jitter=0.1)
sns.stripplot(x='day',y='total_bill', data = tips, jitter=0.02)
sns.stripplot(x='total_bill', y='day', data = tips)


sns.stripplot(x='total_bill', y='day', data = tips, jitter=1, linewidth=1.2)

sns.stripplot(x='day', y='total_bill', data = tips, hue='sex')
sns.stripplot(x='day', y='total_bill', data=tips,hue='smoker', jitter=True, split=True)
sns.stripplot(x='day', y='total_bill', data=tips,hue='smoker', jitter=True, split=True, order=['Sun', 'Sat', 'Fri', 'Thur'])

sns.stripplot(x='day', y='total_bill', data=tips, marker='D')
sns.stripplot(x='day', y='total_bill', data=tips, marker='D', size=15)

sns.stripplot(x='day', y='total_bill', data=tips, marker='D', size=15, hue='sex')
sns.stripplot(x='day', y='total_bill', data=tips, marker='D', size=15, hue='sex',split=True)


## THE BELOW 2 SHOULD BE RUN TOGETHER
sns.stripplot(x='tip', y='day', data=tips, jitter=True, palette='winter_r')
sns.boxplot(x='tip', y='day', data=tips)


sns.stripplot(x='tip', y='day', data=tips, jitter=True)
sns.violinplot(x='tip', y='day', data=tips)



