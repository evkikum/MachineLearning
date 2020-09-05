# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 12:56:09 2019

@author: evkikum
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import os

os.getcwd()
os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\Practice\Linear Regression\homeprices")


homeprices = pd.read_csv("homeprices.csv")
homeprices.columns = ["area", "price"]


##Option1
homeprices.plot.scatter("area", "price")

##Option2
plt.figure()
plt.scatter(homeprices.area, homeprices.price)
##plt.plot(homeprices["area"], reg.predict(homeprices[["area"]]))
plt.xlabel("Area ")
plt.ylabel("Price ")
plt.title("House Price")


reg = linear_model.LinearRegression().fit(homeprices[["area"]], homeprices["price"])


reg.predict(26000)

reg.coef_
reg.intercept_



plt.figure()
plt.scatter(homeprices.area, homeprices.price)
plt.plot(homeprices["area"], reg.predict(homeprices[["area"]]))
plt.xlabel("Area ")
plt.ylabel("Price ")
plt.title("House Price")

## Ways to find the price for list of areas in areas file.

areas = pd.read_csv("areas.csv")
d = reg.predict(areas)
areas["price"] = d