# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:30:58 2019

@author: evkikum
"""


import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

%matplotlib inline

tips = sns.load_dataset('tips')


sns.violinplot(x=tips['tip'])
sns.violinplot(x=tips['total_bill'])

sns.violinplot(x=tips['size'], y=tips['total_bill'])
sns.violinplot(x='day', y='total_bill', data=tips)
sns.violinplot(x='day', y='total_bill', data=tips, hue='sex')

sns.violinplot(x='day', y='total_bill', data=tips, hue='sex', palette='spring')
sns.violinplot(x='day', y='total_bill', data=tips, hue='smoker', palette='spring', split=True)

sns.violinplot(x='day', y='total_bill', data=tips, palette='spring', order=['Sat','Thur', 'Sun', 'Fri'])

sns.violinplot(x='day', y='total_bill', data=tips, palette='spring', order=['Sat','Thur', 'Sun', 'Fri'], hue='smoker', split=True)

sns.violinplot(x='day', y='total_bill', data=tips, palette='coolwarm', hue='smoker', inner='quartile')
sns.violinplot(x='day', y='total_bill', data=tips, palette='coolwarm', hue='smoker', inner='quartile', split=True)


====STILL TO CONTINE PART2
