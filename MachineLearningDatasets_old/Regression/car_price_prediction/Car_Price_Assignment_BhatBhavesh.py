# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:56:42 2019

@author: evkikum
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
np.random.seed(0)
np.set_printoptions(precision=2)
import os

os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\Practice\Linear Regression\car_price_prediction")

def pretty_print_linear(coefs, names = None, sort = False):
    if names is None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst, key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name) for coef, name in lst)


def root_mean_square_error(y_pred,y_test):
    rmse_train = np.sqrt(np.dot(abs(y_pred-y_test),abs(y_pred-y_test))/len(y_test))
    return rmse_train

def ohc_conversion(input_text):
    if "ohc" in input_text:
        return 1
    else:
        return 0
    
def plot_real_vs_predicted(y_pred,y_test):
    plt.plot(y_pred,y_test,'ro')
    plt.plot([0,50],[0,50], 'g-')
    plt.xlabel('predicted')
    plt.ylabel('real')
    plt.show()
    return plt

df = pd.read_csv("CarPrice_Assignment.csv")

print ("Number of rows in dataset = ", df.shape[0])
print ("Number of features in dataset = ", df.shape[1])

df.drop('car_ID',axis=1, inplace=True)
df.isnull().sum()
df.describe()

categorical_columns = list(df.select_dtypes(['object']).columns)

for cols in categorical_columns:
    print ([cols], " : ", df[cols].unique())

len(df.CarName.unique())


'''
147 different car models would give rise to 147 different columns when they are one-hot encoded.
We can group different car models based on their 1st name & since the count of porsche, isuzu, jaguar, alfa-romero, chevrolet, vw, renault etc are less than 6 we can combine it into others category
'''

def car_type(input_text):
    other_cars = set(['porsche', 'isuzu', 'jaguar', 'alfa-romero', 'chevrolet',
              'vw', 'renault','maxda','mercury', 'Nissan', 'toyouta',
              'vokswagen', 'porcshce'])
    if (len(input_text.split(' '))>1) and (input_text.split(' ')[0] not in other_cars):
        return input_text.split(' ')[0]
    else:
        return 'others'    


df.CarName.apply(car_type).value_counts()
df.CarName = df.CarName.apply(car_type)

df.head()

word2num_mapping = {'one': 1,'two':2,'three':3,'four':4, 'five':5, 'six':6, 'seven':7,
                    'eight':8, 'nine':9, 'twelve':12}

df.doornumber.replace(word2num_mapping,inplace=True)
df.doornumber = df.doornumber.astype(int)

df.cylindernumber.replace(word2num_mapping,inplace=True)
df.cylindernumber = df.cylindernumber.astype(int)

df.fueltype       = df.fueltype.astype('category').cat.codes
df.aspiration     = df.aspiration.astype('category').cat.codes
df.enginelocation = df.enginelocation.astype('category').cat.codes

df["enginetype"].value_counts()

df.enginetype = df.enginetype.apply(ohc_conversion)

Y = df.price.values
df.drop('price',axis=1,inplace=True)

df = pd.get_dummies(df, columns=["CarName","carbody", "drivewheel", "fuelsystem"], prefix=["make", "body","drive", "fuel"])

X = df.values
names = df.columns

####Feature Selection using Multiple methods

ranks = {}
r = {}

def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))

lr = LinearRegression(normalize=True)
lr.fit(X, Y)
ranks["Linear reg"] = rank_to_dict(np.abs(lr.coef_), names)

ridge = Ridge(alpha=7)
ridge.fit(X, Y)
ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)

lasso = Lasso(alpha=.05)
lasso.fit(X, Y)
ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), names)

rlasso = RandomizedLasso(alpha=0.04)
rlasso.fit(X, Y)
ranks["Stability"] = rank_to_dict(np.abs(rlasso.scores_), names)

#stop the search when 5 features are left (they will get equal scores)
rfe = RFE(lr, n_features_to_select=5)
rfe.fit(X,Y)
##ranks["RFE"] = rank_to_dict(map(float, rfe.ranking_), names, order=-1)


print('Optimal no of features : {} '.format(rfe.n_features_))
print("Featured selected {}".format(', '.join(np.array(df.columns)[rfe.support_].tolist())))


rf = RandomForestRegressor()
rf.fit(X,Y)
ranks["RF"] = rank_to_dict(rf.feature_importances_, names)

f, pval  = f_regression(X, Y, center=True)
ranks["Corr."] = rank_to_dict(f, names)

for name in names:
    r[name] = round(np.mean([ranks[method][name]for method in ranks.keys()]), 2)
    
methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")

print "\t%s" % "\t".join(methods)
for name in names:
    print "%s\t%s" % (name, "\t".join(map(str, [ranks[method][name] for method in methods])))


mean_plot = pd.DataFrame(list(r.items()),columns=['Feature', 'Mean Ranking'])
mean_plot = mean_plot.sort_values('Mean Ranking', ascending=False)

mean_plot

sns.factorplot(x="Mean Ranking", y="Feature", data = mean_plot, kind="bar", 
               size=14, aspect=1.9, palette='coolwarm')

# Create linear regression object
linreg = LinearRegression()

# Train the model using the training sets
linreg.fit(X,Y)

print( "Linear model: ", pretty_print_linear(linreg.coef_, names, sort = True))

# Predict the values using the model
Y_lin_predict = linreg.predict(X)

# Print the root mean square error 
print ("Root Mean Square Error: ", root_mean_square_error(Y_lin_predict,Y))

plot_real_vs_predicted(Y,Y_lin_predict)




