# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:44:52 2019

@author: evkikum
"""


import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt


num = np.random.randn(150)
sns.distplot(num)

sns.distplot(num, color ='red')

label_dist = pd.Series(num, name='variable x')


## HERE IN THIS PLOT THE Y AXIS REPRESENTS PROBABILITY DENSITY FUNCTION
## x axis represent the value mentioned.
sns.distplot(label_dist)
sns.distplot(label_dist, vertical=True)
sns.distplot(label_dist, vertical=True, color='red')
sns.distplot(label_dist, vertical=True, color="red", hist=False)

sns.distplot(label_dist, color='green', hist=False, rug=True)