# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:33:59 2019

@author: evkikum
"""

## https://github.com/bhattbhavesh91/GA_Sessions/blob/master/ga_dsmp_5jan2019/16_feature_selection.ipynb

## The below Video explains about the below program
## https://www.youtube.com/watch?v=xlHk4okO8Ls

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.cross_validation import train_test_split
import os

os.getcwd()
os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\Practice\Linear Regression\diabetes")

np.set_printoptions(precision=3)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
warnings.filterwarnings('ignore')
np.random.seed(8)
%matplotlib inline

def generate_accuracy_and_heatmap(model, x, y):
#     cm = confusion_matrix(y,model.predict(x))
#     sns.heatmap(cm,annot=True,fmt="d")
    ac = accuracy_score(y,model.predict(x))
    f_score = f1_score(y,model.predict(x))
    print('Accuracy is: ', ac)
    print('F1 score is: ', f_score)
    print ("\n")
    print (pd.crosstab(pd.Series(model.predict(x), name='Predicted'),
                       pd.Series(y['Outcome'],name='Actual')))
    return 1
    
    

df = pd.read_csv("diabetes.csv")
df.shape
df.columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
df.Outcome.value_counts()
df['BloodPressureSquare'] = np.square(df['BloodPressure'])
df['BloodPressureCube'] = df['BloodPressure']**3
df['BloodPressureSqrt'] = np.sqrt(df['BloodPressure'])

df['GlucoseSquare'] = np.square(df['Glucose'])
df['GlucoseCube'] = df['Glucose']**3
df['GlucoseSqrt'] = np.sqrt(df['Glucose'])

df['GlucoseBloodPressure'] = df['BloodPressure'] * df['Glucose']
df['AgeBMI'] = df['Age'] * df['BMI']


categorical_feature_columns  = list(set(df.columns) - set(df._get_numeric_data().columns))
numerical_feature_columns  = df._get_numeric_data().columns
target = 'Outcome'

k = 15 ## No of variables for heat map

cols = df[numerical_feature_columns].corr().nlargest(k, target)[target].index
cm = df[cols].corr()
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True, cmap = 'viridis')

X = df.loc[:, df.columns != target]
Y = df.loc[:, df.columns == target]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1234)

clf_lr = LogisticRegression()      
lr_baseline_model = clf_lr.fit(x_train,y_train)

generate_accuracy_and_heatmap(lr_baseline_model,x_train, y_train )
ac1 = accuracy_score(y_train ,lr_baseline_model.predict(x_train))
f_score1 = f1_score(y_train, lr_baseline_model.predict(x_train))










