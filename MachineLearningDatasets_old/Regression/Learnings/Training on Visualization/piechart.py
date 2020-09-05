# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:23:00 2019

@author: evkikum
"""

import matplotlib.pyplot as plt
%matplotlib inline


exp_vals = [1400, 600, 300, 410, 250]
exp_labels = ["Home Rent", "Food", "Phone/Inter bill", "Car", "Other Utilities"]

plt.pie(exp_vals, labels= exp_labels)




plt.axis("equal")
plt.pie(exp_vals, labels=exp_labels)


plt.axis("equal")
plt.pie(exp_vals, labels=exp_labels)
plt.show()


plt.axis("equal")
plt.pie(exp_vals, labels=exp_labels, radius=1.5)
plt.show()


plt.axis("equal")
plt.pie(exp_vals, labels=exp_labels, radius=1.5, autopct="%0.0f%%")   ## No decimal and plain
plt.show()

plt.axis("equal")
plt.pie(exp_vals, labels=exp_labels, radius=1.5, autopct="%0.2f%%")   ## 2 decimal
plt.show()


plt.axis("equal")
plt.pie(exp_vals, labels=exp_labels, radius=1.5, autopct="%0.2f%%", shadow=True)   ## It thickens the ring
plt.show()

plt.axis("equal")
plt.pie(exp_vals, labels=exp_labels, radius=1.5, autopct="%0.2f%%", shadow=True, explode=[0,0.1,0.1,0,0])   ## in explode the selected pieces will be deattached
plt.show()

plt.axis("equal")
plt.pie(exp_vals, labels=exp_labels, radius=1.5, autopct="%0.2f%%", shadow=True, explode=[0,0.1,0.1,0,0], startangle=180)  ## IT ROTATES THE PIECHART BY 180
plt.show()