#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 11:57:51 2020

@author: evkikum
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


os.chdir(r"/home/evkikum/Desktop/Data Science/Python/GreenInstitute_Course")

catsdata = pd.read_csv("Data/cats.csv") # Predict Hwt

def MAPE(actual, predicted):   
    actual_np = np.array(actual)
    predicted_np = np.array(predicted)
    ape = abs(actual_np - predicted_np)*100/actual_np  
    ape = ape[np.isfinite(ape)] # removes records with infinite percentage error
    mean_ape = np.mean(ape)
    median_ape = np.median(ape)
    return(pd.Series([mean_ape,median_ape],
                     index = ["Mean APE", "Median APE"]))


genderwise_count = catsdata.groupby("Gender").size()

genderwise_count["F"]/sum(genderwise_count)

# Is there any difference in Bwt between Male and Female cats?
catsdata.groupby("Gender")["Bwt"].agg([np.mean, np.median])

catsdata.boxplot(column = "Bwt", by="Gender")

catsdata.plot.scatter("Bwt", "Hwt")

cond_f = catsdata.loc[:,"Gender"] == "F"
cond_m = catsdata.loc[:,"Gender"] == "M"

cats_m = catsdata[catsdata["Gender"] == "M"]
cats_f = catsdata[catsdata["Gender"] == "F"]

fig, (ax_m, ax_f) = plt.subplots(2, 1, sharex = True, sharey = True)
ax_m.scatter(cats_m["Bwt"],cats_m["Hwt"], color = "blue")
ax_f.scatter(cats_f["Bwt"],cats_f["Hwt"], color = "red")


sns.lmplot("Bwt", "Hwt", data=catsdata, fit_reg=False, hue="Gender")

catsdata["Hwt"].corr(catsdata["Bwt"])

plt.scatter(catsdata["Bwt"],catsdata["Hwt"])
plt.xlim([0,4])
plt.ylim([0,20])

cats_simp_lin_model = ols(formula = "Hwt ~ Bwt", data = catsdata).fit()
cats_simp_lin_model.params   ##fitted_hwt = -0.35 + 4.03 * Bwt
fitted_hwt = cats_simp_lin_model.predict(catsdata)


plt.scatter(catsdata["Bwt"], fitted_hwt, color = "red")
plt.scatter(catsdata["Bwt"], catsdata["Hwt"], color = "green")

cats_simp_lin_model_sk = LinearRegression().fit(pd.DataFrame(catsdata["Bwt"]), 
                                      catsdata["Hwt"])

cats_simp_lin_model_sk.coef_ # 4.034
cats_simp_lin_model_sk.intercept_ # -0.356
cats_simp_lin_model_sk.score(pd.DataFrame(catsdata["Bwt"]), catsdata["Hwt"])   ## 64%

MAPE(catsdata["Hwt"], fitted_hwt)   ## Mean - 11.2 % Median - 9.95 %

mae = mean_absolute_error(catsdata["Hwt"], fitted_hwt)   ## 1.17%
rmse = sqrt(mean_squared_error(catsdata["Hwt"], fitted_hwt)) #  1.44 %% 



## Train test split 

cats_train_data, cats_test_data = train_test_split(catsdata, test_size = 0.3, random_state = 1234)
cats_simp_lin_model = ols(formula = "Hwt ~ Bwt", data = cats_train_data).fit()
cats_simp_lin_model.rsquared   ## 62.49%%
cats_simp_lin_model.params   ## Hwt = -.34 + 4.06 * Bwt

fitted_hwt_tr_data = cats_simp_lin_model.predict(cats_train_data)
MAPE(cats_train_data["Hwt"],fitted_hwt_tr_data)   ## Mean - 12.07% Median - 10.74%%

pred_hwt_te_data = cats_simp_lin_model.predict(cats_test_data)
MAPE(cats_test_data["Hwt"], pred_hwt_te_data)  ## Mean - 10% Median - 9 %

mae_1 = mean_absolute_error(cats_test_data["Hwt"], pred_hwt_te_data)   ## 0.99%
rmse_1 = sqrt(mean_squared_error(cats_test_data["Hwt"], pred_hwt_te_data)) #  1.2 %% 


###Treating categorical data
cat_gender_encoded = pd.get_dummies(catsdata["Gender"])
cats2 = pd.concat([catsdata,cat_gender_encoded], axis = 1)

cats_multi_lin_model = ols("Hwt ~ Bwt + M + F", data = cats2).fit()
cats_multi_lin_model.params

catsdata.nunique()

cats_multi_lin_model.rsquared ## 64.68%%
cats_multi_lin_model.params   ## Hwt = -0.3 + 4.08 * Bwt - 0.19 * M - 0.11 * F

fitted_hwt = cats_simp_lin_model.predict(catsdata)

MAPE(catsdata["Hwt"], fitted_hwt)   ## Mean - 11.46 % Median - 10.42 %

mae_2 = mean_absolute_error(catsdata["Hwt"], fitted_hwt)   ## 1.18%
rmse_2 = sqrt(mean_squared_error(catsdata["Hwt"], fitted_hwt)) #  1.44 %% 



## Treating categorical data with Train/test split
cat_gender_encoded = pd.get_dummies(catsdata["Gender"])
cats2 = pd.concat([catsdata, cat_gender_encoded], axis = 1)

cats_train_data, cats_test_data = train_test_split(cats2, test_size = 0.3, random_state=1234)

cats_multi_lin_model = ols("Hwt ~ Bwt + M + F", data = cats_train_data).fit()
cats_multi_lin_model.params   ## Hwt = -.1 + 3.98 * Bwt + 0.18 * M - 0.12 * F
cats_multi_lin_model.rsquared   ## 62.55%

fitted_hwt_tr_data = cats_multi_lin_model.predict(cats_train_data)
MAPE(cats_train_data["Hwt"], fitted_hwt_tr_data)  ## Mean - 12.10 % Median - 10.6 %


fitted_hwt_te_data = cats_multi_lin_model.predict(cats_test_data)
MAPE(cats_test_data["Hwt"], fitted_hwt_te_data)   ## Mean - 10.16, Median - 9.09

mae_3 = mean_absolute_error(cats_test_data["Hwt"], fitted_hwt_te_data)   ## 1.004
rmse_3 = sqrt(mean_squared_error(cats_test_data["Hwt"], fitted_hwt_te_data)) #  1.22 %% 







