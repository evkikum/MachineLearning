# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 16:58:48 2019

@author: evkikum
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from matplotlib.pyplot import xticks
import os

os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\GreenInstitute_Course")

wgdata = pd.read_csv("data/wg.csv")
wgdata.isnull().sum()

wgclean = wgdata.dropna(subset = ["metmin", "wg"])
wgmale = wgclean.loc[wgclean["Gender"] == 'M', :]
wgfemale = wgclean.loc[wgclean["Gender"] == 'F', :]

fig, (ax_m, ax_f) = plt.subplots(1,2 , sharex = True, sharey = True)
ax_f.hist(wgfemale["wg"], color = "red")
ax_m.hist(wgmale["wg"], color = "red")


fig, (ax_m, ax_f) = plt.subplots(1,2 , sharex = True, sharey = True)
ax_f.hist(wgfemale["wg"], color = "blue", bins=[0,10,20,30,50,70,80,100], rwidth = .95, label="Female")
ax_m.hist(wgmale["wg"], color = "red", bins=[0,10,20,30,50,70,80,100], rwidth = .95, label="Male")
ax_f.legend()
ax_m.legend()



f, (ax1, ax2) = plt.subplots(2, 1, sharex = True, sharey = True)
ax1.hist(wgfemale["wg"], color = "red")
ax2.hist(wgmale["wg"], color = "blue")

plt.subplots(2, 2)
plt.subplots(3, 1)



os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\GreenInstitute_Course")
df = pd.read_csv("data/CarPrice_Assignment.csv");

fig, axs = plt.subplots(2,2,figsize=(15,10))
plt1 = sns.scatterplot(x = 'carlength', y = 'price', data = df, ax = axs[0,0])
plt1.set_xlabel('Length of Car (Inches)')
plt1.set_ylabel('Price of Car (Dollars)')
plt2 = sns.scatterplot(x = 'carwidth', y = 'price', data = df, ax = axs[0,1])
plt2.set_xlabel('Width of Car (Inches)')
plt2.set_ylabel('Price of Car (Dollars)')
plt3 = sns.scatterplot(x = 'carheight', y = 'price', data = df, ax = axs[1,0])
plt3.set_xlabel('Height of Car (Inches)')
plt3.set_ylabel('Price of Car (Dollars)')
plt4 = sns.scatterplot(x = 'curbweight', y = 'price', data = df, ax = axs[1,1])
plt4.set_xlabel('Weight of Car (Pounds)')
plt4.set_ylabel('Price of Car (Dollars)')
plt.tight_layout()


fig,axs= plt.subplots(3,2,figsize=(15,10))
plt1=sns.scatterplot(x='enginesize', y='price',data = df, ax=axs[0,0])
plt1.set_xlabel('Size of engine')
plt1.set_ylabel('Price of car')
plt2=sns.scatterplot(x='boreratio',y='price', data=df, ax=axs[0,1])
plt2.set_xlabel('Bore ratio')
plt2.set_ylabel('Price of car')
plt3=sns.scatterplot(x='stroke', y='price', data=df, ax=axs[1,0])
plt3.set_xlabel('Stroke')
plt3.set_ylabel('Price of car')
plt4=sns.scatterplot(x='compressionratio', y='price', data=df, ax=axs[1,1])
plt4.set_xlabel('compressionratio')
plt4.set_ylabel('Price of car')
plt5=sns.scatterplot(x='horsepower', y='price', data=df, ax=axs[2,0])
plt5.set_xlabel('horsepower')
plt5.set_ylabel('Price of car')
plt6=sns.scatterplot(x='peakrpm', y='price', data=df, ax=axs[2,1])
plt6.set_xlabel('peakrpm')
plt6.set_ylabel('Price of car')

