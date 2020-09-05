# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 18:54:00 2019

@author: evkikum
"""


import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

%matplotlib inline

iris = sns.load_dataset('iris')

x = sns.PairGrid(iris)
x = x.map(plt.scatter)


x = sns.PairGrid(iris)
x = x.map_diag(plt.hist)
x = x.map_offdiag(plt.scatter)


x= sns.PairGrid(iris, hue='species')
x = x.map_diag(plt.hist)
x = x.map_offdiag(plt.scatter)
x.add_legend()

x= sns.PairGrid(iris, hue='species', palette='coolwarm')
x = x.map_diag(plt.hist)
x = x.map_offdiag(plt.scatter)
x.add_legend()


x= sns.PairGrid(iris, hue='species', palette='autumn')
x = x.map_diag(plt.hist, histtype='step')
x = x.map_offdiag(plt.scatter)
x.add_legend()


x= sns.PairGrid(iris, hue='species', palette='autumn')
x = x.map_diag(plt.hist, histtype='step', linewidth=4)
x = x.map_offdiag(plt.scatter)
x.add_legend()

x=sns.PairGrid(iris, vars=['petal_length', 'petal_width'])
x=x.map_diag(plt.hist)
x=x.map_offdiag(plt.scatter)


x=sns.PairGrid(iris, hue='species',vars=['petal_length', 'petal_width'])
x=x.map_diag(plt.hist, edgecolor='black')
x=x.map_offdiag(plt.scatter, edgecolor='white')
x=x.add_legend()

x=sns.PairGrid(iris,  x_vars=['petal_length', 'petal_width'], y_vars=['sepal_length', 'sepal_width'])
x=x.map(plt.scatter)

x=sns.PairGrid(iris)
x=x.map_diag(plt.hist)
x=x.map_upper(plt.scatter)
x=x.map_lower(sns.kdeplot)

x= sns.PairGrid(iris, hue='species', palette='coolwarm')
x=x.map_diag(plt.hist)
x=x.map_upper(plt.scatter)
x=x.map_lower(sns.kdeplot)

x=sns.PairGrid(iris, hue='species', hue_kws={'marker':['D','s','+']})
x=x.map(plt.scatter, s=30, edgecolor='black')
x=x.add_legend()

