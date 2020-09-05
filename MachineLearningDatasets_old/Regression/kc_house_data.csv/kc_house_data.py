# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 18:29:02 2019

@author: evkikum
"""


### https://www.kaggle.com/arthurtok/feature-ranking-rfe-random-forest-linear-models


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import os

os.getcwd()
os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\Practice\Linear Regression\kc_house_data.csv")

house = pd.read_csv("kc_house_data.csv")
house.info()
house.head()

house.isnull().sum()
house.dtypes


## Dropping the id and date columns
house = house.drop(['id', 'date'], axis = 1)

sns.pairplot(house[['sqft_lot','sqft_above','price','sqft_living','bedrooms']], hue='bedrooms', palette='afmhot',size=1.4)

house.plot.scatter("sqft_lot", "price")
house.plot.scatter("sqft_above", "price")
house.plot.scatter("sqft_living", "price")
house.plot.scatter("bedrooms", "price")


str_list = []
for colname, colvalue in house.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)
         
num_list = house.columns.difference(str_list) 
house.info()
house_num  = house[num_list]


f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation of features')
# Draw the heatmap using seaborn
#sns.heatmap(house_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="PuBuGn", linecolor='k', annot=True)
sns.heatmap(house_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="cubehelix", linecolor='k', annot=True)


### USAGE OF Randomized Lasso
Y = house.price.values
house = house.drop(['price'], axis = 1)
X = house.as_matrix()
colnames = house.columns

ranks = {}

def ranking(ranks, names, order = 1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))    
    
    
rlasso = RandomizedLasso(alpha=0.04)
rlasso.fit(X,Y)
ranks["rlasso/Stability"] = ranking(np.abs(rlasso.scores_), colnames)
print('finished')

###Recursive Feature Elimination ( RFE )

lr = LinearRegression(normalize = True)
lr.fit(X,Y)
#stop the search when only the last feature is left
rfe = RFE(lr, n_features_to_select=1, verbose =3 )
rfe.fit(X,Y)
ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)


###linear model feature Ranking

##Using linear regression
lr = LinearRegression(normalize=True)
lr.fit(X,Y)

ranks["LinReg"] = ranking(np.abs(lr.coef_), colnames)

# Using Ridge
ridge = Ridge(alpha = 7)
ridge.fit(X,Y)
ranks['Ridge'] = ranking(np.abs(ridge.coef_), colnames)

##Using Lasso
lasso = Lasso(alpha=.05)
lasso.fit(X, Y)
ranks["Lasso"] = ranking(np.abs(lasso.coef_), colnames)

##Using Random Forest
rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=3)
rf.fit(X,Y)
ranks["RF"] = ranking(rf.feature_importances_, colnames);



r = {}
for name in colnames:
    r[name] = round(np.mean([ranks[method][name] 
                             for method in ranks.keys()]), 2)
 
methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")
 
print("\t%s" % "\t".join(methods))
for name in colnames:
    print("%s\t%s" % (name, "\t".join(map(str, 
                         [ranks[method][name] for method in methods]))))
    
meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])
meanplot = meanplot.sort_values('Mean Ranking', ascending=False)


sns.factorplot(x="Mean Ranking", y="Feature", data = meanplot, kind="bar", 
               size=14, aspect=1.9, palette='coolwarm')