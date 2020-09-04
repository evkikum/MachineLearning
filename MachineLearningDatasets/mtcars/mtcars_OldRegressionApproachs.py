#!/usr/bin/env python3
# -*- coding: utf-8 -*---
"""
Created on Wed Aug  5 05:27:17 2020

@author: evkikum

https://www.kaggle.com/ajinkyaa/mtcars-feature-selection-and-transformations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols # https://www.statsmodels.org/stable/index.html
from sklearn.linear_model import LinearRegression # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
from sklearn.model_selection import train_test_split # latest version of sklearn
#from sklearn.cross_validation import train_test_split # older version of sklearn
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso, ElasticNet
from sklearn.feature_selection import RFE


os.chdir(r"/home/evkikum/Desktop/Data Science/Python/Practice/mtcars")


def MAPE(actual, predicted):   
    actual_np = np.array(actual)
    predicted_np = np.array(predicted)
    ape = abs(actual_np - predicted_np)*100/actual_np  
    ape = ape[np.isfinite(ape)] # removes records with infinite percentage error
    mean_ape = np.mean(ape)
    median_ape = np.median(ape)
    return(pd.Series([mean_ape,median_ape],
                     index = ["Mean APE", "Median APE"]))

mtcars = pd.read_csv("mtcars.csv")
mtcars.info()
mtcars["cyl"].value_counts()   ## Categorical
mtcars["vs"].value_counts()   ## vs
mtcars["am"].value_counts()   ## am
mtcars["gear"].value_counts()   ## gear
mtcars["carb"].value_counts()   ## carb
mtcars.columns = map(str.upper, mtcars.columns)


mtcars_train_data, mtcars_test_data = train_test_split(mtcars, test_size = 0.3, random_state = 1234)

mtcars_simp_lin_model = ols(formula = "MPG ~ CYL + DISP + HP + DRAT + WT + QSEC + VS + AM + GEAR + CARB ", data = mtcars_train_data).fit()
mtcars_simp_lin_model.params
mtcars_simp_lin_model.rsquared   ## 86.9%

fitted_mpg_tr_data = mtcars_simp_lin_model.predict(mtcars_train_data)
MAPE(mtcars_train_data["MPG"], fitted_mpg_tr_data)   ## Mean - 10.2, Median - 9.58 %

fitted_mpg_te_data = mtcars_simp_lin_model.predict(mtcars_test_data)
Mape_mean, Mape_median = MAPE(mtcars_test_data["MPG"], fitted_mpg_te_data)   ## Mean - 17.06, Median - 12.775 %


rmae_1 = mean_absolute_error(mtcars_test_data["MPG"], fitted_mpg_te_data)  ### 3.08 %
rmse_1 = sqrt(mean_squared_error(mtcars_test_data["MPG"], fitted_mpg_te_data))   ## 4.34


output_df = pd.DataFrame(index = ["RSquared", "rmae","rmse", "Mape_mean", "Mape_median"])
output_df["Basic Test"] = [mtcars_simp_lin_model.rsquared, rmae_1, rmse_1, Mape_mean, Mape_median]



### Treating categorical variables;
mtcars = pd.read_csv("mtcars.csv")
mtcars.info()
mtcars.columns = map(str.upper, mtcars.columns)

mtcars["CYL"] = mtcars["CYL"].astype("category")
mtcars["VS"] = mtcars["VS"].astype("category")
mtcars["AM"] = mtcars["AM"].astype("category")
mtcars["GEAR"] = mtcars["GEAR"].astype("category")
mtcars["CARB"] = mtcars["CARB"].astype("category")

y = mtcars["MPG"]

mtcars.drop("MPG", axis = 1, inplace = True)
mtcars = pd.get_dummies(mtcars)
mtcars = pd.concat([y, mtcars], axis = 1)


mtcars_train_data, mtcars_test_data = train_test_split(mtcars, test_size = 0.3, random_state = 1234)

mtcars_simp_lin_model = ols(formula = 'MPG ~ DISP + HP + DRAT + WT + QSEC + CYL_4 + CYL_6 + CYL_8 + VS_0 + VS_1 + AM_0 + AM_1 + GEAR_3 + GEAR_4 + GEAR_5 + CARB_1 + CARB_2 + CARB_3 + CARB_4 + CARB_6 + CARB_8', data = mtcars_train_data).fit()
mtcars_simp_lin_model.params
mtcars_simp_lin_model.summary()
mtcars_simp_lin_model.rsquared   ## 91.44 %

fitted_mpg_tr_data = mtcars_simp_lin_model.predict(mtcars_train_data)
MAPE(mtcars_train_data["MPG"], fitted_mpg_tr_data)   ## Mean - 7.87, Median - 6.12 %

fitted_mpg_te_data = mtcars_simp_lin_model.predict(mtcars_test_data)
Mape_mean, Mape_median = MAPE(mtcars_test_data["MPG"], fitted_mpg_te_data)

rmae_2 = mean_absolute_error(mtcars_test_data["MPG"], fitted_mpg_te_data)  ### 5.49 %
rmse_2 = sqrt(mean_squared_error(mtcars_test_data["MPG"], fitted_mpg_te_data))   ## 7.54 % 

output_df["Category test"] = [mtcars_simp_lin_model.rsquared, rmae_2, rmse_2, Mape_mean, Mape_median]



### Backward feature elimination

## https://www.kaggle.com/ajinkyaa/mtcars-feature-selection-and-transformations

mtcars = pd.read_csv("mtcars.csv")

X=mtcars.drop(['mpg'],axis=1)
Y=mtcars.mpg
model=sm.OLS(Y,X).fit()
model.pvalues

lin=LinearRegression()
cols=list(X.columns)
select_feat=[]

while(len(cols)>0):
    p=[]
    X1=X[cols]
    model1=sm.OLS(Y,X1).fit()
    p=pd.Series(model1.pvalues,index=X1.columns)
    pmax=max(p)
    if(pmax>0.05):
        feature_with_p_max = p.idxmax()
        print("feature_with_p_max ", feature_with_p_max)
        cols.remove(feature_with_p_max)
    else:
        break
select_feat=cols
select_feat

mtcars1 = mtcars[["mpg","wt", "qsec", "am"]]

mtcars_train_data, mtcars_test_data = train_test_split(mtcars1, test_size = 0.3, random_state = 1234)

mtcars_simp_lin_model = ols(formula = 'mpg ~ wt + qsec + am ', data = mtcars_train_data).fit()
mtcars_simp_lin_model.params
mtcars_simp_lin_model.summary()
mtcars_simp_lin_model.rsquared   ## 84.33 %

fitted_mpg_te_data = mtcars_simp_lin_model.predict(mtcars_test_data)
Mape_mean, Mape_median = MAPE(mtcars_test_data["mpg"], fitted_mpg_te_data)

rmae_2 = mean_absolute_error(mtcars_test_data["mpg"], fitted_mpg_te_data)  ### 5.49 %
rmse_2 = sqrt(mean_squared_error(mtcars_test_data["mpg"], fitted_mpg_te_data))   ## 7.54 % 


output_df["Bckwrd_ftur_elm"] = [mtcars_simp_lin_model.rsquared, rmae_2, rmse_2, Mape_mean, Mape_median]

### Backward feature elimination using categorical features

mtcars = pd.read_csv("mtcars.csv")
mtcars.info()

mtcars["cyl"] = mtcars["cyl"].astype("category")
mtcars["vs"] = mtcars["vs"].astype("category")
mtcars["am"] = mtcars["am"].astype("category")
mtcars["gear"] = mtcars["gear"].astype("category")
mtcars["carb"] = mtcars["carb"].astype("category")

y = mtcars["mpg"]

mtcars.drop("mpg", axis = 1, inplace = True)
mtcars = pd.get_dummies(mtcars)
mtcars = pd.concat([y, mtcars], axis = 1)
mtcars.info()


mtcars_train_data, mtcars_test_data = train_test_split(mtcars, test_size = 0.3, random_state = 1234)

mtcars_simp_lin_model = ols(formula = 'mpg ~ disp + hp + drat + wt + qsec + cyl_4 + cyl_6 + cyl_8 + vs_0 + vs_1 + am_0 + am_1 + gear_3 + gear_4 + gear_5 + carb_1 + carb_2 + carb_3 + carb_4 + carb_6 + carb_8', data = mtcars_train_data).fit()
mtcars_simp_lin_model.params
mtcars_simp_lin_model.summary()
mtcars_simp_lin_model.rsquared   ## 91.44 %

X=mtcars.drop(['mpg'],axis=1)
Y=mtcars.mpg
model=sm.OLS(Y,X).fit()
model.pvalues

lin=LinearRegression()
cols=list(X.columns)
select_feat=[]

while(len(cols)>0):
    p=[]
    X1=X[cols]
    model1=sm.OLS(Y,X1).fit()
    p=pd.Series(model1.pvalues,index=X1.columns)
    pmax=max(p)
    if(pmax>0.05):
        feature_with_p_max = p.idxmax()
        print("feature_with_p_max ", feature_with_p_max)
        cols.remove(feature_with_p_max)
    else:
        break
select_feat=cols
select_feat

mtcars1 = pd.concat([mtcars["mpg"],mtcars[select_feat]], axis = 1) 
mtcars1.info()

mtcars_train_data, mtcars_test_data = train_test_split(mtcars1, test_size = 0.3, random_state = 1234)

mtcars_simp_lin_model = ols(formula = 'mpg ~ wt + hp + wt + cyl_4 + cyl_6 + cyl_8 + vs_0 + vs_1 + am_0 + am_1 + gear_3 + gear_4 + gear_5', data = mtcars_train_data).fit()
mtcars_simp_lin_model.params
mtcars_simp_lin_model.summary()
mtcars_simp_lin_model.rsquared   ## 85.77 %

fitted_mpg_te_data = mtcars_simp_lin_model.predict(mtcars_test_data)
Mape_mean, Mape_median = MAPE(mtcars_test_data["mpg"], fitted_mpg_te_data)

rmae_2 = mean_absolute_error(mtcars_test_data["mpg"], fitted_mpg_te_data)  ### 5.49 %
rmse_2 = sqrt(mean_squared_error(mtcars_test_data["mpg"], fitted_mpg_te_data))   ## 7.54 % 

output_df["Bckwrd_ftr_elm_category"] = [mtcars_simp_lin_model.rsquared, rmae_2, rmse_2, Mape_mean, Mape_median]


## USING RECURSIVE FEATURE SELECTION

mtcars = pd.read_csv("mtcars.csv")

X = mtcars.drop(['mpg'],axis=1)
Y = mtcars.mpg
model=sm.OLS(Y,X).fit()
model.pvalues

lin=LinearRegression()
X.columns
highsc=0
nof=0
support_score=[]
noflist=np.arange(1,11)
for n in noflist:
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=1)
    rfe=RFE(lin,n)
    X_train_rfe=rfe.fit_transform(x_train,y_train)
    X_test_rfe=rfe.transform(x_test)
    lin.fit(X_train_rfe,y_train)
    score=lin.score(X_test_rfe,y_test)
    if(score>highsc):
        highsc=score
        nof=n
        support_score=rfe.support_
        
temp=pd.Series(support_score,index=X.columns)
print('No of optimum features:',n)
print('SCore for optimum features:',highsc)
print('Features Selected:\n')
temp[temp==True].index

X2=mtcars[['drat', 'wt', 'gear', 'carb']]
Y2=mtcars.mpg
x_train,x_test,y_train,y_test=train_test_split(X2,Y2,test_size=0.3,random_state=1)
lin1=LinearRegression()
lin1.fit(x_train,y_train)
print('R2 for train:',lin1.score(x_train,y_train))
print('R2 for test:',lin1.score(x_test,y_test))

r_squared = lin1.score(x_test,y_test)

predicted = lin1.predict(x_test)

rmae = mean_absolute_error(y_test, predicted)  
rmse = sqrt(mean_squared_error(y_test, predicted))  

Mape_mean, Mape_median = MAPE(y_test, predicted)

output_df["RFE"] = [r_squared, rmae, rmse, Mape_mean, Mape_median]



'''
## Using recursive feature elimination

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score

mtcars = pd.read_csv("mtcars.csv")
mtcars.columns = map(str.upper, mtcars.columns)

feature_names = mtcars.drop("MPG", axis = 1).columns

mtcars["CYL"] = mtcars["CYL"].astype("category")
mtcars["VS"] = mtcars["VS"].astype("category")
mtcars["AM"] = mtcars["AM"].astype("category")
mtcars["GEAR"] = mtcars["GEAR"].astype("category")
mtcars["CARB"] = mtcars["CARB"].astype("category")

z  = mtcars["MPG"]
mtcars.drop("MPG", axis = 1, inplace = True)

mtcars = pd.get_dummies(mtcars)
mtcars = pd.concat([z, mtcars], axis = 1)

X  = mtcars.drop("MPG", axis = 1).values
y  = mtcars["MPG"].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.7, random_state = 123)

lm = LinearRegression()
lm.fit(X_train, y_train)

predicted = lm.predict(X_test)
r_squared = r2_score(y_train,lm.predict(X_train))

rmae_3 = mean_absolute_error(y_test, predicted)  ### 6.84 %
rmse_3 = sqrt(mean_squared_error(y_test, predicted))   ## 8.15 % 

Mape_mean, Mape_median = MAPE(y_test, predicted)

output_df["RFECV before"] = [r_squared, rmae_3, rmse_3, Mape_mean, Mape_median]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
lm = LinearRegression() 

rfecv = RFECV(estimator=lm, step=1, cv=5)

rfecv.fit(X_scaled, y_train)

print('Optimal number of features: {}'.format(rfecv.n_features_))

ranked_features, _ = zip(*sorted(zip(feature_names, rfecv.ranking_.tolist()),
                                 key=lambda x: x[1],
                                 reverse=True))
print('Suggested order of feature removal: {}'.format(', '.join(ranked_features)))

X_train_subset = X_train[:, rfecv.support_]
lm2 = LinearRegression()
lm2.fit(X_train_subset, y_train)


X_test_part = X_test[:, rfecv.support_]
predicted = lm2.predict(X_test_part)

r_squared = r2_score(y_train,lm2.predict(X_train_subset))   ## .85%
##r_squared = r2_score(y_test, predicted)
mae = np.mean(abs(predicted - y_test))
rmse = np.sqrt(np.mean((predicted - y_test)**2))

rmae_4 = mean_absolute_error(y_test, predicted)  ### 4.7 %
rmse_4 = sqrt(mean_squared_error(y_test, predicted))   ## 6.33 % 

Mape_mean, Mape_median = MAPE(y_test, predicted)

output_df["RFECV after"] = [r_squared, rmae_4, rmse_4, Mape_mean, Mape_median]
'''




#### Peason correlation !###################

mtcars = pd.read_csv("mtcars.csv")
mtcars.info()

cor=mtcars.corr()
cor1=cor['mpg']

featu=cor1[abs(cor1)>0.5][1:]

mult_cor=mtcars[['cyl', 'disp', 'hp', 'drat', 'wt', 'vs', 'am', 'carb','mpg']].corr()
cor_max=max(abs(featu.values))
final=featu[abs(featu.values)==cor_max]
final


X2=mtcars[['wt']]
Y2=mtcars.mpg
x_train,x_test,y_train,y_test=train_test_split(X2,Y2,test_size=0.3,random_state=1)
lin1=LinearRegression()
lin1.fit(x_train,y_train)
print('R2 for train:',lin1.score(x_train,y_train))
print('R2 for test:',lin1.score(x_test,y_test))

r_squared = lin1.score(x_test,y_test)

predicted = lin1.predict(x_test)

rmae = mean_absolute_error(y_test, predicted)  ### 4.7 %
rmse = sqrt(mean_squared_error(y_test, predicted))   ## 6.33 % 

Mape_mean, Mape_median = MAPE(y_test, predicted)


output_df["Peason correlation"] = [r_squared, rmae, rmse, Mape_mean, Mape_median]




## FEATURE SELECTION ON BASIS OF VARIANCE INFLATION FACTOR
## VIF - IT IS MEASURE OF COLLINEARITY AMONG PREDICTOR VARIABLES WITHIN MULTIPLE REGRESSION.
'''
Steps for Implementing VIF

1) Run a multiple regression.
2) Calculate the VIF factors.
3) Inspect the factors for each predictor variable, if the VIF is between 5-10, 
    multicolinearity is likely present and you should consider dropping the variable.
'''


mtcars = pd.read_csv("mtcars.csv")
mtcars.info()

X = mtcars.drop("mpg", axis = 1)
Y = mtcars.mpg

vif = [variance_inflation_factor(X.values, j) for j in range(X.shape[1])] 
vif_pd = pd.Series(vif, index = X.columns)
vif_pd


def calculate_vif(x):    
    output = pd.DataFrame()
    vif = [variance_inflation_factor(x.values, j) for j in range(x.shape[1])]
    cols = x.shape[1]
    print('cols ', cols)
    thresh = 5.0
    
    for i in range(cols):
        print('Iteration: ', i)
        a = np.argmax(vif)
        print('Max vif found at : ', a)
        
        if(vif[a]>thresh):
            if i == 0:
                output = x.drop(x.columns[a], axis = 1)
                vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
            else:
                output = output.drop(output.columns[a], axis = 1)
                vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
        else:
            break
        
    return output.columns


calculate_vif(X).values    

X2 = mtcars[['disp', 'vs', 'am']]
Y2 = mtcars[['mpg']]

x_train, x_test, y_train, y_test = train_test_split(X2, Y2, test_size=0.3, random_state = 1)
lin1 = LinearRegression()
lin1.fit(x_train, y_train)

print('R2 for train:',lin1.score(x_train,y_train))
print('R2 for test:',lin1.score(x_test,y_test))

r_squared = lin1.score(x_test,y_test)

predicted = lin1.predict(x_test)

rmae = mean_absolute_error(y_test, predicted)  ### 4.7 %
rmse = sqrt(mean_squared_error(y_test, predicted))   ## 6.33 % 

Mape_mean, Mape_median = MAPE(y_test, predicted)


output_df["VIF"] = [r_squared, rmae, rmse, Mape_mean, Mape_median]



### LASSO

mtcars = pd.read_csv("mtcars.csv")
mtcars.info()

X = mtcars.drop("mpg", axis = 1)
Y = mtcars.mpg

reg = LassoCV()
reg.fit(X, Y)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,Y))

coef = pd.Series(reg.coef_, index = X.columns)

coeff=coef.sort_values()
coeff.plot(kind='bar')
plt.show()

X2=mtcars[['disp', 'hp']]
Y2=mtcars.mpg
x_train,x_test,y_train,y_test=train_test_split(X2,Y2,test_size=0.3,random_state=1)
lin1=LinearRegression()
lin1.fit(x_train,y_train)
print('R2 for train:',lin1.score(x_train,y_train))
print('R2 for test:',lin1.score(x_test,y_test))

r_squared = lin1.score(x_test,y_test)

predicted = lin1.predict(x_test)

rmae = mean_absolute_error(y_test, predicted)  ### 4.7 %
rmse = sqrt(mean_squared_error(y_test, predicted))   ## 6.33 % 

Mape_mean, Mape_median = MAPE(y_test, predicted)


output_df["LASSO"] = [r_squared, rmae, rmse, Mape_mean, Mape_median]



### ELASTICNET

mtcars = pd.read_csv("mtcars.csv")
mtcars.info()

X = mtcars.drop("mpg", axis = 1)
Y = mtcars.mpg

reg1 = ElasticNet()
reg1.fit(X,Y)

print("Best alpha using built-in ElasticNet: %f" % reg1.alpha)
print("Best score using built-in ElasticNet: %f" %reg1.score(X,Y))

coef_elastic = pd.Series(reg1.coef_, index = X.columns)

coeff=coef_elastic.sort_values()
coeff.plot(kind='bar')
plt.show()

X2=mtcars[['wt','carb','cyl','disp', 'hp','qsec']]
Y2=mtcars.mpg
x_train,x_test,y_train,y_test=train_test_split(X2,Y2,test_size=0.3,random_state=1)
lin1=LinearRegression()
lin1.fit(x_train,y_train)
print('R2 for train:',lin1.score(x_train,y_train))
print('R2 for test:',lin1.score(x_test,y_test))


r_squared = lin1.score(x_test,y_test)
predicted = lin1.predict(x_test)

rmae = mean_absolute_error(y_test, predicted)  ### 4.7 %
rmse = sqrt(mean_squared_error(y_test, predicted))   ## 6.33 % 

Mape_mean, Mape_median = MAPE(y_test, predicted)


output_df["ELASTICNET"] = [r_squared, rmae, rmse, Mape_mean, Mape_median]


output_df.to_csv("/home/evkikum/Desktop/Data Science/Python/Practice/mtcars/images/" + "OldRegressionApproaches.csv", index=True)
