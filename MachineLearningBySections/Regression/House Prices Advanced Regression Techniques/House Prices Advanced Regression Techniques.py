# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:47:16 2019

@author: evkikum
"""


#### https://www.kaggle.com/akihirokkkkk/house-prize

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder


os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\Practice\Linear Regression\House Prices Advanced Regression Techniques")

test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")


test_id = test["Id"]

train = train.drop("Id",axis=1)
test = test.drop("Id",axis=1)

# poolQC
train["PoolQC"] = train["PoolQC"].fillna("None")
test["PoolQC"] = test["PoolQC"].fillna("None")

#MiscFeature
train["MiscFeature"] = train["MiscFeature"].fillna("None")
test["MiscFeature"] = test["MiscFeature"].fillna("None")

#Alley
train["Alley"] = train["Alley"].fillna("None")
test["Alley"] = test["Alley"].fillna("None")

#Fence
train["Fence"] = train["Fence"].fillna("None")
test["Fence"] = test["Fence"].fillna("None")

#fireplace
train["FireplaceQu"] = train["FireplaceQu"].fillna("None")
test["FireplaceQu"] = test["FireplaceQu"].fillna("None")

train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
test["LotFrontage"] = test.groupby("Neighborhood")["LotFrontage"].transform(lambda x : x.fillna(x.median()))


for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    train[col] = train[col].fillna('None')
    test[col] = test[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)
del col

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)
    
for col in ("BsmtQual",'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    train[col] = train[col].fillna("None")
    test[col] = test[col].fillna("None")
del col

# masvnr系
train["MasVnrType"] = train["MasVnrType"].fillna("None")
test["MasVnrType"] = test["MasVnrType"].fillna("None")

train["MasVnrArea"] = train["MasVnrArea"].fillna(0)
test["MasVnrArea"] = test["MasVnrArea"].fillna(0)

#MSZoning - the purpose of mode is to fetch the most occuruing element
train['MSZoning'] = train['MSZoning'].fillna(train['MSZoning'].mode()[0])
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])


###below command is equivalent to table in R
train["MSZoning"].value_counts()
#############

train = train.drop("Utilities", axis = 1)
test = test.drop("Utilities", axis = 1)


train["Functional"] = train["Functional"].fillna("Typ")
test["Functional"] = test["Functional"].fillna("Typ")

train["Electrical"] = train["Electrical"].fillna("SBrkr")
test["Electrical"] = test["Electrical"].fillna("SBrkr")


train["KitchenQual"] = train["KitchenQual"].fillna(train["KitchenQual"].mode()[0])
test["KitchenQual"] = test["KitchenQual"].fillna(test["KitchenQual"].mode()[0])

train["Exterior1st"] = train["Exterior1st"].fillna(train["Exterior1st"].mode()[0])
test["Exterior1st"] = test["Exterior1st"].fillna(test["Exterior1st"].mode()[0])

train["Exterior2nd"] = train["Exterior2nd"].fillna(train["Exterior2nd"].mode()[0])
test["Exterior2nd"] = test["Exterior2nd"].fillna(test["Exterior2nd"].mode()[0])

train["SaleType"] = train["SaleType"].fillna(train["SaleType"].mode()[0])
test["SaleType"] = test["SaleType"].fillna(test["SaleType"].mode()[0])

train["MSSubClass"] = train["MSSubClass"].fillna('None')
test["MSSubClass"] = test["MSSubClass"].fillna("None")


train["MSSubClass"] = train["MSSubClass"].astype(str)
test['MSSubClass'] = test['MSSubClass'].astype(str)

train["OverallCond"] = train["OverallCond"].astype(str)
test["OverallCond"] = test["OverallCond"].astype(str)

train['YrSold'] = train['YrSold'].astype(str)
test['YrSold'] = test['YrSold'].astype(str)

train['MoSold'] = train['MoSold'].astype(str)
test['MoSold'] = test['MoSold'].astype(str)


# year
train['YrBltAndRemod']=train['YearBuilt']+train['YearRemodAdd']
test['YrBltAndRemod']=test['YearBuilt']+test['YearRemodAdd']

#total
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

train['Total_sqr_footage'] = (train['BsmtFinSF1'] + train['BsmtFinSF2'] + train['1stFlrSF'] + train['2ndFlrSF'])
test['Total_sqr_footage'] = (test['BsmtFinSF1'] + test['BsmtFinSF2'] + test['1stFlrSF'] + test['2ndFlrSF'])

train['Total_Bathrooms'] = (train['FullBath'] + (0.5 * train['HalfBath']) + train['BsmtFullBath'] + (0.5 * train['BsmtHalfBath']))
test['Total_Bathrooms'] = (test['FullBath'] + (0.5 * test['HalfBath']) + test['BsmtFullBath'] + (0.5 * test['BsmtHalfBath']))

train['Total_porch_sf'] = (train['OpenPorchSF'] + train['3SsnPorch'] + train['EnclosedPorch'] + train['ScreenPorch'] + train['WoodDeckSF'])
test['Total_porch_sf'] = (test['OpenPorchSF'] + test['3SsnPorch'] + test['EnclosedPorch'] + test['ScreenPorch'] + test['WoodDeckSF'])

train = train.drop('Street', axis = 1)
test = test.drop('Street', axis = 1)

train = train.fillna(train.median())
test = test.fillna(test.median())

## IN THE BELOW frac = 1 means that 100% of the records should be copied from the dataframe. if frac = .5 then only 50% should be copied from dataframe.
## say in the below sample command if it has n= 10 then only records should be copied from data frame.
## https://www.geeksforgeeks.org/python-pandas-dataframe-sample/
train = train.sample(frac=1, random_state=0)
train.info();

co_box = []

for co in train.columns:
    try:
        sumup = train[co].sum()
        if(type(sumup) == type("dokabenman")):
            co_box.append(co)
    except:
        print(co)
        
   
for obj_col in co_box:
    le = LabelEncoder()
    train[obj_col] = train[obj_col].apply(lambda x:str(x))
    train[obj_col] = pd.DataFrame({obj_col:le.fit_transform(train[obj_col])})
    
    test[obj_col] = test[obj_col].apply(lambda x:str(x))
    test[obj_col] = pd.DataFrame({obj_col:le.fit_transform(test[obj_col])}) 
    

