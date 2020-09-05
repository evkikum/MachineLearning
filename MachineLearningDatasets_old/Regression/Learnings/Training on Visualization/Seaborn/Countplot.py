# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:52:26 2019

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


df = pd.read_csv("data/CarPrice_Assignment.csv");

sns.countplot(df["symboling"], order=pd.value_counts(df["symboling"]).index)


df['brand'] = df["CarName"].str.split(" ").str.get(0).str.upper()

sns.countplot(df["brand"], order=pd.value_counts(df['brand']).index)
xticks(rotation = 90)
