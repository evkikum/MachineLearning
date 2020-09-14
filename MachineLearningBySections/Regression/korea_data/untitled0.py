# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:57:47 2019

@author: evkikum
"""

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
%matplotlib inline
import os

os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\Practice\Linear Regression\korea_data")

econ_df = pd.read_excel("korea_data.xlsx")
econ_df= econ_df.replace("..", 'nan')
econ_df = econ_df.set_index('Year')
econ_df = econ_df.astype(float)
econ_df = econ_df.loc['1969':'2016']
econ_df.count()


column_names = {'unemployment','gdp_growth','gross_capital_formation','pop_growth','birth_rate','broad_money_growth','final_consum_gdp','final_consum_growth','gov_final_consum_growth','gross_cap_form_growth','hh_consum_growth'}

econ_df.info()
econ_df = econ_df.rename(columns = column_names)
econ_df.columns = column_names
econ_df.info()
                                                      
## econ_df = econ_df.rename(columns = column_names)
econ_df["final_consum_gdp"]

print('-'*100)
print(econ_df.isnull().any())

corr = econ_df.corr()

unemployment, gdp_growth
gdp_growth, gross_cap_form_growth
unemployment, gross_cap_form_growth
gov_final_consum_growth, birth_rate

final_consum_growth, birth_rate
broad_money_growth, birth_rate
gross_capital_formation, final_consum_gdp
broad_money_growth, gov_final_consum_growth






econ_df["birth_rate"].corr(econ_df["pop_growth"])
econ_df["final_consum_growth"].corr(econ_df["hh_consum_growth"])

,  (.994)
final_consum_growth, gdp_growth (.)
gross_cap_form_growth, gdp_growth
hh_consum_growth, gdp_growth
unemployment, gross_capital_formation
final_consum_growth, gross_cap_form_growth
gross_capital_formation, hh_consum_growth


##PLOT THE CORRELATION HEATMAP
sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, cmap= 'RdBu')

econ_df_before = econ_df
econ_df_after = econ_df.drop(['gdp_growth','birth_rate', 'final_consum_growth','gross_capital_formation'], axis = 1)

X1 = sm.tools.add_constant(econ_df_before)
X2 = sm.tools.add_constant(econ_df_after)

series_before = pd.Series([variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])], index = X1.columns) 
series_after  = pd.Series([variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])], index = X2.columns)


