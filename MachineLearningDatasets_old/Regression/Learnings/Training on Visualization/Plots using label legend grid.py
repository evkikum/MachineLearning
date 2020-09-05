# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:05:23 2019

@author: evkikum
"""

import matplotlib.pyplot as plt
%matplotlib inline


x = [1,2,3,4,5,6,7]
y = [50,51,52,48,47,49,46]



plt.xlabel('Day')
plt.ylabel('Temperature')
plt.title('Weather')
## linestyle - dashed/solid/dotted
plt.plot(x,y, color = 'blue',linewidth=5, linestyle='dotted')   
## the scale of alpha is (0-1)
plt.plot(x,y, color = 'blue',linewidth=5, linestyle='dotted', alpha=.5)   
plt.plot(x,y, color = 'blue',linewidth=5, linestyle='dotted', alpha=1)   



## How to plot mutiple plot all in one chart

days = [1,2,3,4,5,6,7]
max_t = [50,51,52,48,47,49,46]
min_t = [43, 42,40, 44,33, 35, 37]
avg_t = [45,48, 48, 46, 40, 42,41]

plt.xlabel("Days")
plt.ylabel("Temperature")
plt.title('Weather')
plt.plot(days, max_t)
plt.plot(days, min_t)
plt.plot(days, avg_t)



## legend - to name the individual plots

days = [1,2,3,4,5,6,7]
max_t = [32,51,52,48,47,49,46]
min_t = [43, 42,40, 44,33, 35, 37]
avg_t = [45,48, 48, 46, 40, 42,41]

plt.xlabel("Days")
plt.ylabel("Temperature")
plt.title("Weather")
plt.plot(days, max_t, label="Max")
plt.plot(days, min_t, label="Min")
plt.plot(days, avg_t, label="Avg")
plt.legend()


## grid

days = [1,2,3,4,5,6,7]
max_t = [32,51,52,48,47,49,46]
min_t = [43, 42,40, 44,33, 35, 37]
avg_t = [45,48, 48, 46, 40, 42,41]

plt.xlabel("Days")
plt.ylabel("Temperature")
plt.title("Weather")
plt.plot(days, max_t, label="Max")
plt.plot(days, min_t, label="Min")
plt.plot(days, avg_t, label="Avg")
plt.legend()
plt.grid()

