# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 20:33:14 2019

@author: evkikum
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

blood_sugar = [113, 85,90, 150, 149, 88, 93, 115, 135, 80, 77, 82, 129]

## IN HISTOGRAM ONLY SINGLE DIMENSION VARIABLE IS ENOUGH
## BY DEFAULT HISTOGRAMS DISPLAYS 10 BINS
plt.hist(blood_sugar)  ## by default the no of bins are 10

plt.hist(blood_sugar, bins = 3)  ## This will reduce the bins to 3.


plt.hist(blood_sugar, bins = 3, rwidth=0.95)  ## rwidth means relative width of bar compared to bin size.


## Alternat way of using bins
plt.hist(blood_sugar, bins=[80,100,125,150], rwidth=.95, color = "green")  


## How to avoid hard rigid bar
plt.hist(blood_sugar, bins=[80,100,125,150], rwidth=.95, color = "green", histtype= "step")  




blood_sugar_men = [113, 85,90, 150, 149, 88, 93, 115, 135, 80, 77, 82, 129]
blood_sugar_women = [67,98,89, 120, 133, 150, 84, 69, 89, 79, 120, 112, 100 ]


plt.figure()
plt.xlabel("Sugar Range")
plt.ylabel("Total no of patients")
plt.hist([blood_sugar_men, blood_sugar_women], bins=[80, 100,125,150] ,color = ["green", "orange"], label=["Men", "Women"])
plt.legend()



### FOR HORIZONTAL ORIENTATION

plt.figure()
plt.xlabel("Sugar Range")
plt.ylabel("Total no of patients")
plt.hist([blood_sugar_men, blood_sugar_women], bins=[80, 100,125,150] ,color = ["green", "orange"], label=["Men", "Women"], orientation="horizontal")
plt.legend()
plt.grid()








