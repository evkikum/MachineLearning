# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:12:23 2019

@author: evkikum
"""

### Need to implement the below solution present in below Kaggle URL
## https://www.kaggle.com/ashydv/car-price-prediction-linear-regression

## below usage VIF for model building
## https://www.kaggle.com/ankitakaggle/car-price-prediction


## EDA STEP BY STEP
## https://www.kaggle.com/prakharrathi25/exploratory-data-analysis-step-by-step
## https://www.kaggle.com/punith02/linear-regression-to-predict-car-price

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from matplotlib.pyplot import xticks
from sklearn.cross_validation import train_test_split # older version of sklearn
import os
from sklearn.preprocessing import MinMaxScaler
from scipy import stats


os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\GreenInstitute_Course")

def MAPE(actual, predicted):
    actual_np = np.array(actual)
    predicted_np = np.array(predicted)
    ape = abs(actual_np - predicted_np)* 100/actual_np
    mean_ape = np.mean(ape)
    median_ape = np.median(ape)
    return pd.Series([mean_ape, median_ape], index = ["Mean_APE", "Median_APE"])


df = pd.read_csv("data/CarPrice_Assignment.csv");
df_prev = df
df.info()
## Checking dupliactes
sum(df.duplicated(subset = 'car_ID'))  ## 0 duplicates
df.isnull().sum()  ## 0 nulls

### BELOW DISPLAYS THE VALUD COUNTS FOR ALL CATEGORICAL VARIABLES
for k in df.columns:
    if (df[k].dtype not in [np.float64, np.int64]):
        print("Value counts ",
              df[k].value_counts())


## CHECK FOR OUTLIERS-- THE BELOW PRINT STATEMENT SHOWS THAT THIER ARE OUTIERS IN DATAFRAME.        
for k, v in df_prev.items():
    if (df[k].dtype in [np.float64, np.int64]):
        q1 = df_prev[k].quantile(.25)
        q3 = df_prev[k].quantile(.75)
        iqr = q3 - q1
        uwl = q3 + (iqr * 1.5)
        lwl = q1 - (iqr * 1.5)
        df_prev = df_prev[(df_prev[k] >= lwl) & (df_prev[k] <= uwl)]
        print("k ", k,
              "COun to records ", df_prev.shape)

## THE BELOW HISTOGRAM SHOWS THAT THE MOST CARS ARE PRICED AT  Rs (6000-7000) price range
plt.hist(df["price"], bins=[5000,6000,7000,8000, 9000, 10000, 15000, 20000, 25000, 30000], rwidth = .9)
sns.distplot(df['price'])

## Price values are right skewed, most cars are priced ate the lower end (9000) of the price range
##sns.distplot(df["price"])
df_sym = pd.DataFrame(df["symboling"].value_counts()) 

sns.countplot(df["symboling"], order=pd.value_counts(df["symboling"]).index)
xticks(rotation=90)
    
plt.axis("equal")
plt.pie(df_sym["symboling"], labels = df_sym.index.values, autopct="%0.2f%%")
plt.show()

##LET US SEE THE AVERAGE PRICE OF CAR IN EACH SYMBOL CATEGORY
sns.barplot(x='symboling', y='price', data = df, estimator=np.mean)

## Based on tyhe above graph the average price of car is lower for 0,1,2 symbol category

df['brand'] = df['CarName'].str.split(' ').str.get(0).str.upper()

plt1 = sns.countplot(df["brand"], order=pd.value_counts(df['brand']).index)
xticks(rotation = 90)

df['brand'] = df['brand'].replace(['VW', 'VOKSWAGEN'], 'VOLKSWAGEN')
df['brand'] = df['brand'].replace(['MAXDA'], 'MAZDA')
df['brand'] = df['brand'].replace(['PORCSHCE'], 'PORSCHE')
df['brand'] = df['brand'].replace(['TOYOUTA'], 'TOYOTA')

plt1 = sns.countplot(df["brand"], order=pd.value_counts(df['brand']).index)
xticks(rotation = 90)

df_comp_avg_price = df[['brand','price']].groupby("brand", as_index = False).mean().rename(columns={'price':'brand_avg_price'})
df = df.merge(df_comp_avg_price, on = 'brand')

sns.barplot(x='brand',y='price', data=df, estimator=np.mean)
xticks(rotation=90)




def dec_brand_category(x):
    if x < 10000:
        return "Budget"
    elif (x >= 10000 and x < 20000):
        return "Mid_Range"
    else:   
        return "Luxury"

df['brand_category'] = df["brand_avg_price"].apply(dec_brand_category)
sns.countplot(df['brand_category'])

df_fuel_avg_price  = df.groupby("fueltype",as_index = False)["price"].agg(np.mean)
df_fuel_avg_price.columns = ["fueltype", "fuel_avg_price"]
sns.barplot(x='fueltype', y='price', data=df, estimator=np.mean)   ## Diesal cars are priced more than gas cars
sns.barplot(x='aspiration',y='price', data=df, estimator=np.mean)
sns.barplot(x='doornumber', y='price', data =df, estimator=np.mean)  ## No of doors do not have much impact on price
sns.barplot(x='carbody', y='price', data=df, estimator=np.mean) ## COnverible and Hardtop  are very expensive
sns.barplot(x='drivewheel', y='price', data=df, estimator=np.mean) ## Cars is rear wheel is expensive
sns.barplot(x='enginetype', y='price', data=df, estimator=np.mean) ## DOHCV and OHCV engine types are priced high.
xticks(rotation = 90)

sns.barplot(x="cylindernumber", y='price', data = df, estimator=np.mean)  ## Eight and twelve cylinder cars have higher price.
xticks(rotation=90)

sns.barplot(x="fuelsystem", y='price', data = df, estimator=np.mean) ## IDI and MPFI fuel system have higher price.
xticks(rotation=90)

sns.barplot(x='enginetype', y='price', data=df, estimator=np.mean) 
xticks(rotation = 90)

sns.scatterplot(x='wheelbase', y='price', data = df)

fig, axs = plt.subplots(2,2,figsize=(15,10))
plt1 = sns.scatterplot(x = 'carlength', y = 'price', data = df, ax = axs[0,0])
plt1.set_xlabel('Length of Car (Inches)')
plt1.set_ylabel('Price of Car (Dollars)')
plt2 = sns.scatterplot(x = 'carwidth', y = 'price', data = df, ax = axs[0,1])
plt2.set_xlabel('Width of Car (Inches)')
plt2.set_ylabel('Price of Car (Dollars)')
plt3 = sns.scatterplot(x = 'carheight', y = 'price', data = df, ax = axs[1,0])
plt3.set_xlabel('Height of Car (Inches)')
plt3.set_ylabel('Price of Car (Dollars)')
plt4 = sns.scatterplot(x = 'curbweight', y = 'price', data = df, ax = axs[1,1])
plt4.set_xlabel('Weight of Car (Pounds)')
plt4.set_ylabel('Price of Car (Dollars)')
plt.tight_layout()

sns.barplot(x='enginetype', y='price', data = df, estimator=np.mean) # DOHCV and OHCV engine types are priced high.
sns.barplot(x='cylindernumber', y='price', data = df, estimator=np.mean)  ## Eight and twelve cylinder cars have higher price.
sns.barplot(x='fuelsystem', y='price', data=df, estimator=np.mean)  ## IDI and MPFI fuel system have higher price.

fig,axs= plt.subplots(3,2,figsize=(15,10))
plt1=sns.scatterplot(x='enginesize', y='price',data = df, ax=axs[0,0])
plt1.set_xlabel('Size of engine')
plt1.set_ylabel('Price of car')
plt2=sns.scatterplot(x='boreratio',y='price', data=df, ax=axs[0,1])
plt2.set_xlabel('Bore ratio')
plt2.set_ylabel('Price of car')
plt3=sns.scatterplot(x='stroke', y='price', data=df, ax=axs[1,0])
plt3.set_xlabel('Stroke')
plt3.set_ylabel('Price of car')
plt4=sns.scatterplot(x='compressionratio', y='price', data=df, ax=axs[1,1])
plt4.set_xlabel('compressionratio')
plt4.set_ylabel('Price of car')
plt5=sns.scatterplot(x='horsepower', y='price', data=df, ax=axs[2,0])
plt5.set_xlabel('horsepower')
plt5.set_ylabel('Price of car')
plt6=sns.scatterplot(x='peakrpm', y='price', data=df, ax=axs[2,1])
plt6.set_xlabel('peakrpm')
plt6.set_ylabel('Price of car')
plt.tight_layout()
plt.show()

# Size of Engine, bore ratio, and Horsepower has positive correlation with price.

df['mileage'] = df['citympg']*0.55 + df['highwaympg']*0.45

plt1 = sns.scatterplot(x='mileage', y='price', data=df)
plt1.set_xlabel('Mileage')
plt1.set_ylabel('pricve')

## Mileage and price are negatively correlated

pl1 = sns.scatterplot(x='mileage', y = 'price', hue='brand_category', data = df)
plt1.set_xlabel('Mileage')
plt1.set_ylabel('price')
plt.show()

plt1 = sns.scatterplot(x='mileage',y='price', data = df, hue='fueltype')
plt1.set_xlabel('mileage')
plt1.set_ylabel('price')
plt.show()

plt1 = sns.scatterplot(x='horsepower', y='price', data=df, hue='fueltype')
plt1.set_xlabel('horsepower')
plt1.set_ylabel('price')
plt.show()

sns.scatterplot(x='mileage', y='price', data = df, hue = 'fueltype')
sns.scatterplot(x='horsepower', y = 'price', data = df, hue = 'fueltype')



'''

Summary Univariate and Bivriate Analysis:

From the above Univariate and bivariate analysis we can filter out variables which does not affect price much.
The most important driver variable for prediction of price are:-

    Brand Category
    Fuel Type
    Aspiration
    Car Body
    Drive Wheel
    Wheelbase
    Car Length
    Car Width
    Curb weight
    Engine Type
    Cylinder Number
    Engine Size
    Bore Ratio
    Horsepower
    Mileage
'''

auto = df[['fueltype', 'aspiration', 'carbody', 'drivewheel', 'wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginetype',
       'cylindernumber', 'enginesize',  'boreratio', 'horsepower', 'price', 'brand_category', 'mileage']]
auto.head()

sns.pairplot(auto)

plt.figure(figsize=(10, 20))
plt.subplot(4,2,1)
sns.boxplot(x = 'fueltype', y = 'price', data = auto)
plt.subplot(4,2,2)
sns.boxplot(x = 'aspiration', y = 'price', data = auto)
plt.subplot(4,2,3)
sns.boxplot(x = 'carbody', y = 'price', data = auto)
plt.subplot(4,2,4)
sns.boxplot(x = 'drivewheel', y = 'price', data = auto)
plt.subplot(4,2,5)
sns.boxplot(x = 'enginetype', y = 'price', data = auto)
plt.subplot(4,2,6)
sns.boxplot(x = 'brand_category', y = 'price', data = auto)
plt.subplot(4,2,7)
sns.boxplot(x = 'cylindernumber', y = 'price', data = auto)
plt.tight_layout()
plt.show()


## DATA PREPARATION 
## DUMMY VARIABLES
cyl_no = pd.get_dummies(auto['cylindernumber'], drop_first=True)
auto = pd.concat([auto, cyl_no], axis=1)

brand_cat  = pd.get_dummies(auto['brand_category'], drop_first=True)
auto = pd.concat([auto, brand_cat], axis = 1)

eng_typ = pd.get_dummies(auto['enginetype'], drop_first=True)
auto = pd.concat([auto, eng_typ], axis = 1)

drwh  = pd.get_dummies(auto['drivewheel'], drop_first=True)
auto = pd.concat([auto, drwh], axis=1)

carb  = pd.get_dummies(auto['carbody'], drop_first=True)
auto = pd.concat([auto, carb], axis = 1)

asp = pd.get_dummies(auto['aspiration'], drop_first=True)
auto = pd.concat([auto, asp], axis = 1)

fuelt = pd.get_dummies(auto['fueltype'], drop_first=True)
auto = pd.concat([auto, fuelt], axis = 1)

auto = auto.drop(['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginetype', 'cylindernumber','brand_category'], axis = 1)



sns.barplot(x = 'fueltype', y = 'price', data = df, estimator=np.mean)
sns.barplot(x='aspiration', y = 'price', data = df, estimator=np.mean)
sns.barplot(x='doornumber', y = 'price', data = df, estimator=np.mean)          ## NO IMPACT
sns.barplot(x='carbody', y='price', data = df, estimator=np.mean)
sns.barplot(x='drivewheel', y = 'price', data = df, estimator=np.mean)
sns.barplot(x='enginelocation', y = 'price', data = df, estimator=np.mean)      ==> 
sns.scatterplot(x='wheelbase', y='price', data = df)  ## Postive 
sns.scatterplot(x='carlength', y = 'price', data = df) ## Positive
sns.scatterplot(x='carwidth', y = 'price', data = df) ## Positive
sns.scatterplot(x='carheight', y = 'price', data = df)            ## NO IMPACT
sns.scatterplot(x='curbweight', y='price', data = df) ## Positive
sns.barplot(x='enginetype', y='price', data = df)
sns.barplot(x='cylindernumber', y='price', data = df)
sns.scatterplot(x='enginesize', y='price', data = df) ## Positive
sns.barplot(x='fuelsystem', y='price', data = df)                               ==> 
sns.scatterplot(x='boreratio', y='price', data = df) ## Positive
sns.scatterplot(x='stroke', y='price', data = df)        ## Rearely positive
sns.barplot(x='compressionratio', y='price', data = df)                     ==>
xticks(rotation = 90)
sns.scatterplot(x='horsepower', y='price', data = df)  ## Positive
sns.barplot(x='peakrpm', y='price', data = df)                  ==> 
xticks(rotation = 90)
sns.scatterplot(x='citympg', y='price', data = df)  ## -ve          ==> 
sns.scatterplot(x='highwaympg', y = 'price', data = df) ## -ve          ==>
sns.barplot(x='brand', y='price', data = df)
xticks(rotation = 90)


sns.regplot(x='enginelocation', y='price', data = df)

peakrpm_f, p_value = stats.pearsonr(df['peakrpm'], df['price'])
horsepower_f, p_value = stats.pearsonr(df['horsepower'], df['price'])
citympg_f, p_value = stats.pearsonr(df['citympg'], df['price'])
highwaympg_f, p_value = stats.pearsonr(df['highwaympg'], df['price'])

pearson_coef, p_value = stats.pearsonr(df['highwaympg'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of  = ", p_value ) 


## Model building
## Splitting the Data into Training and Testing sets

np.random.seed(0)
df_train, df_test = train_test_split(auto, train_size=0.7, test_size=0.3, random_state=100)

num_vars = ['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize','boreratio', 'horsepower', 'price','mileage']
scaler = MinMaxScaler()

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

plt.figure(figsize=(16,10))
sns.heatmap(df_train.corr(),annot=True ,cmap='YlGnBu')

y_train = df_train.pop('price')
X_train = df_train

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Running RFE with the output number of the variable equal to 10
lm = LinearRegression()
lm.fit(X_train, y_train)
rfe = RFE(lm, 10)             # running RFE
rfe = rfe.fit(X_train, y_train)

list(zip(X_train.columns, rfe.support_, rfe.ranking_))

col = X_train.columns[rfe.support_]

# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]

import statsmodels.api as sm  
X_train_rfe = sm.add_constant(X_train_rfe)

lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model
lm.summary()

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending=False)



