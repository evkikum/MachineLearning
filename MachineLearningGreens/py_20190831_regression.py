import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols # https://www.statsmodels.org/stable/index.html
from sklearn.linear_model import LinearRegression # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
#from sklearn.model_selection import train_test_split # latest version of sklearn
from sklearn.cross_validation import train_test_split # older version of sklearn


catsdata = pd.read_csv("data/cats.csv") # Predict Hwt
mtcars = pd.read_csv("data/mtcars.csv") # Predict mpg

## Step 0: Data Preparation
## Step 1: Knowing the Dependent Variable (DV) and Independent Variable (IDV)
    # DV is always continuous
    # If IDV is continuous, use as such
    # If IDV is nominal categorical (e.g: Gender, Shift, Color), do dummy encoding 
    # If IDV is ordinal categorical (e.g: Cancer level), convert them directly to ordered numbers
## Step 2: Exploratory analysis
   # Statistical summary
   # Scatter Plot between IDVs and DV
   # Correlation analysis: Quantifies the relationship btw 2 variables
     # Correlation should be high between DV and IDVs
      # Correlation is a standardized form of covariance
      # http://www.grroups.com/blog/r-mean-standard-deviation-variance-covariance-and-correlation-using-simple-examples
      # Negative: Altitude vs Temperature, Price vs Demand
      # Positive: Age of Kid vs Weight, Economy vs Stock Prices
      # Output will be between -1 and +1
          # -1: Strong negative relationship
          # 0: Weak relationship
          # +1: Strong positive relationship
      # Check for multi collinearity issue
         # The IDVs should not correlate with each other
           # One IDVs has to be selected from Correlated IDVs 
## Step 3: Build regression models
          # Simple Linear Regression Model
          # Simple Non Linear Regression
              # Variable transformation
                  # square
                  # square root
                  # log
                      # linear - linear
                      # linear - log; IDV is log transformed
                      # Log - Linear; DV is log transformed
                      # Log - Log; DV as well as IDV are log transformed  
          # Multiple Regression
          # Step wise regression
          # Regularized regression (Ridge, Lasso)
          #  https://scikit-learn.org/stable/modules/linear_model.html
## Step 4: Model Evaluation
          # Train-Test-Validate
            # Data given will be split into training and test (say 70-30) 
              # Build model on training data
              # Evaluate the model on test data
            # Test again on validation data - typically done by business
          # MAPE (Mean Absolute Percent Error, Median Absolute Percent Error)
          # R-squared: Variance explained by the model 
            # Adj R-squared for multiple regression
            # will be between 0 and 1
            # 0: Bad Model
            # 1: Good Model
         # Variable significance (p value)
           # lesser the p value; significant is the variable. cutoff is 0.05
             # remove variables with high p values (p > 0.05)
           # Hypothesis testing (t test): 
              # Null hypothesis of t test: probability of the variable to be insigificant
               # Null hypothesis of the judge: Person accused is innocent. Public prosecutor has to provide that he is a criminal
## Step 5: Go live and start predicting         


def MAPE(actual, predicted):   
    actual_np = np.array(actual)
    predicted_np = np.array(predicted)
    ape = abs(actual_np - predicted_np)*100/actual_np  
    ape = ape[np.isfinite(ape)] # removes records with infinite percentage error
    mean_ape = np.mean(ape)
    median_ape = np.median(ape)
    return(pd.Series([mean_ape,median_ape],
                     index = ["Mean APE", "Median APE"]))

## Step 1:
# DV -> Hwt:Heart weight in Grams
# IDV -> Bwt: Body Weight in Kilograms

## Step 2:
# How many male and female cats are present?
genderwise_count = catsdata.groupby("Gender").size()

# What is the percentage of female cats?
47/144 # 32.6%
genderwise_count["F"]/sum(genderwise_count)
# There are less female cats compared to male

# What is the average Bwt and Hwt
catsdata["Bwt"].mean() # 2.7 Kg
catsdata["Hwt"].mean() # 10.g grams

catsdata_stats_summary = catsdata.describe()

# Is there any difference in Bwt between Male and Female cats?
catsdata.groupby("Gender")["Bwt"].agg([np.mean, np.median])
catsdata.boxplot(column = "Bwt", by = "Gender")
# Male cats weigh more compared to female cats

# Is there any relationship between Bwt and Hwt?
catsdata.plot.scatter("Bwt","Hwt")
# As Bwt increases, Hwt increases

# If so, is the relationship different for male and female cats?
cats_m = catsdata[catsdata["Gender"] == "M"]
cats_f = catsdata[catsdata["Gender"] == "F"]

fig, (ax_m, ax_f) = plt.subplots(2, 1, sharex = True, sharey = True)
ax_m.scatter(cats_m["Bwt"],cats_m["Hwt"], color = "blue")
ax_f.scatter(cats_f["Bwt"],cats_f["Hwt"], color = "red")

## Seaborn lmplot has "hue" argument which can be used to 
  # plot different color based on a categorical variable
sns.lmplot("Bwt","Hwt", data = catsdata, fit_reg = False,
           hue = "Gender")

# Covariance
# Covariance is scale dependent
catsdata["Bwt"].cov(catsdata["Hwt"])
mtcars["mpg"].cov(mtcars["wt"])

# Correlation
# Correlation is scale independent; will always be between -1 to 1
catsdata["Bwt"].corr(catsdata["Hwt"]) # 0.804; strong positive correlation
mtcars["mpg"].corr(mtcars["wt"]) # -867; strong negative correlation

### Step 3
#predicted_mpg4 = -8*new_wt + 45
plt.scatter(catsdata["Bwt"],catsdata["Hwt"])
plt.xlim([0,4])
plt.ylim([0,20])

## Formula: DV ~ IDV
cats_simp_lin_model = ols(formula = "Hwt ~ Bwt", data = catsdata).fit()
cats_simp_lin_model.params
# Hwt = 4.034*Bwt - 0.35662
# As Bwt of a cat increases by 1Kg, Hwt will increase by 4.034 grams
fitted_hwt = 4.034*catsdata["Bwt"] - 0.35662
# instead of hard coding the model output, inbuilt predict can be used
fitted_hwt = cats_simp_lin_model.predict(catsdata)
plt.scatter(catsdata["Bwt"],catsdata["Hwt"])
plt.scatter(catsdata["Bwt"],fitted_hwt, c = "red")

# Building model using sklearn
cats_simp_lin_model_sk = LinearRegression().fit(pd.DataFrame(catsdata["Bwt"]), 
                                      catsdata["Hwt"])
cats_simp_lin_model_sk.coef_ # 4.034
cats_simp_lin_model_sk.intercept_ # -0.356

## mtcars: mpg ~ wt
# mtcars = m*wt + C
mtcars_simp_lin_model = ols(formula = "mpg ~ wt", data = mtcars).fit()
mtcars_simp_lin_model.params
# mpg = -5.34*wt + 37.28
# As weight of a car increases by 1 tonne, mileage will come down by 5.34mpg
fitted_mpg = -5.34*mtcars["wt"] + 37.28
fitted_mpg = mtcars_simp_lin_model.predict(mtcars)
plt.scatter(mtcars["wt"],mtcars["mpg"])
plt.scatter(mtcars["wt"],fitted_mpg, c = "red")

# Building model using sklearn
mtcars_simp_lin_model_sk = LinearRegression().fit(pd.DataFrame(mtcars["wt"]), 
                                      mtcars["mpg"])
mtcars_simp_lin_model_sk.coef_ # -5.34
mtcars_simp_lin_model_sk.intercept_ # 37.28

## Step 4: Model Evaluation

# R-squared
cats_simp_lin_model.rsquared  # statsmodel
cats_simp_lin_model_sk.score(pd.DataFrame(catsdata["Bwt"]), 
                                      catsdata["Hwt"]) # sklearn
# 0.6466; decent model

mtcars_simp_lin_model.rsquared 
mtcars_simp_lin_model_sk.score(pd.DataFrame(mtcars["wt"]), 
                                      mtcars["mpg"])
# 0.75; good model

# MAPE
MAPE(catsdata["Hwt"], fitted_hwt) # 11.27% Mean APE, 9.9% Median APE
 # Accuracy: Around 89%
MAPE(mtcars["mpg"], fitted_mpg) # 12.6% Mean APE, 11.07% Median APE
 # Accuracy: Around 87%
 
## Step 5
new_cats = pd.DataFrame({"Bwt": [1.8,2.9,3.8,4.2]})
new_cats["Pred_Hwt"] = cats_simp_lin_model.predict(new_cats)

new_cars = pd.DataFrame({"wt": [4.2, 4.7, 1.85, 5.7]})
new_cars["Predicted_Mileage"] = mtcars_simp_lin_model.predict(new_cars)

#############3 Train - Test ###################################
# Random sampling for train and test
  # fix the random state (set seed) to get reproducible result
cats_train_data, cats_test_data = train_test_split(catsdata,
                                test_size = 0.3, random_state = 1234)
# 100 records for training and 44 records for testing

# Build model on training data
cats_simp_lin_model = ols("Hwt ~ Bwt", data = cats_train_data).fit()
cats_simp_lin_model.rsquared # 0.624
cats_simp_lin_model.params # Hwt = 4.06*Bwt - 0.337

# Evaluate the model on training data
fitted_hwt_tr_data = cats_simp_lin_model.predict(cats_train_data)
MAPE(cats_train_data["Hwt"],fitted_hwt_tr_data)
# 12.07% Mean APE, 10.7% Median APE

# Evaluate the model on test data (unseen data)
pred_hwt_te_data = cats_simp_lin_model.predict(cats_test_data)
MAPE(cats_test_data["Hwt"],pred_hwt_te_data)
# 10.06% Mean APE, 9.02% Median APE

################ wg ####################################################
wgdata = pd.read_csv("data/wg.csv")
wgclean = wgdata.dropna() # removing records with missing values

## Step 1:
# DV: wg (weight gain in lbs)
# IDV: metmin (Activitiy levels measured in metmin)

## Step 2
wg_stats_summary = wgclean.describe()
wgclean.plot.scatter("metmin", "wg")
sns.lmplot("metmin", "wg", data = wgclean, fit_reg = False, hue = "Gender")
sns.lmplot("metmin", "wg", data = wgclean, fit_reg = False, hue = "Shift")
wgclean["wg"].corr(wgclean["metmin"]) # -0.9; strong negative correlation

## Step 3
wg_train_data, wg_test_data = train_test_split(wgclean,
                                test_size = 0.3, random_state = 1234)

## Building the model on training data
wg_simp_lin_model = ols("wg ~ metmin", data = wg_train_data).fit()
wg_simp_lin_model.params
# wg = -0.019*metmin + 54.84
# For 1 unit increase in metmin, wg will decrease by 0.019

## Step 4

# Evaluating on training data
wg_simp_lin_model.rsquared # 0.835; good model
wg_fitted_simp_lin_model = wg_simp_lin_model.predict(wg_train_data)
MAPE(wg_train_data["wg"],wg_fitted_simp_lin_model) # 34.22% Mean APE, 21.7% Median APE

plt.scatter(wg_train_data["metmin"],wg_train_data["wg"])
plt.scatter(wg_train_data["metmin"],wg_fitted_simp_lin_model, c = "red")

# Evaluating on test data
wg_pred_simp_lin_model = wg_simp_lin_model.predict(wg_test_data)
MAPE(wg_test_data["wg"],wg_pred_simp_lin_model) # 47.4% Mean APE, 22.9% Median APE

###### Non linear relationship
x = np.random.randint(1,100,50)
y = 5*x + 10
y = -5*x + 10
y = 5*x**2 + 10 # 2nd order polynomial relationship
y = -5*x**2 + 10
y = 5*x**2 + 10*x + 10
plt.scatter(x,y)

### Non Linear Transformation
# Including square term for metmin
wgclean["metmin_sq"] = wgclean["metmin"]**2
# Log transformation of IDV
wgclean["metmin_log"] = np.log(wgclean["metmin"])
# Log transformation of DV
wgclean["wg_log"] = np.log(wgclean["wg"])
wg_train_data, wg_test_data = train_test_split(wgclean,
                                test_size = 0.3, random_state = 1234)

## Non Linear Model 1: # wg = a*metmin^2 + C
wg_simp_nonlin_model1 = ols("wg ~ metmin_sq", data = wg_train_data).fit()
wg_simp_nonlin_model1.params
# wg = -0.000004*metmin^2 + 35.34
wg_simp_nonlin_model1.rsquared # 0.691; good model
wg_fitted_simp_nonlin_model1 = wg_simp_nonlin_model1.predict(wg_train_data)
MAPE(wg_train_data["wg"],wg_fitted_simp_nonlin_model1) # 43.19% Mean APE, 29.29% Median APE
plt.scatter(wg_train_data["metmin"],wg_train_data["wg"])
plt.scatter(wg_train_data["metmin"],wg_fitted_simp_nonlin_model1, c = "red")
# Error is very high

## Non Linear Model 1: # wg = a*metmin^2 + b*metmin + C
wg_simp_nonlin_model2 = ols("wg ~ metmin + metmin_sq", data = wg_train_data).fit()
wg_simp_nonlin_model2.params
# wg = 0.00001*metmin^2 - 0.06*metmin + 93.15
wg_simp_nonlin_model2.rsquared # 0.974; good model
wg_fitted_simp_nonlin_model2 = wg_simp_nonlin_model2.predict(wg_train_data)
MAPE(wg_train_data["wg"],wg_fitted_simp_nonlin_model2) # 8.8% Mean APE, 6.5% Median APE
plt.scatter(wg_train_data["metmin"],wg_train_data["wg"])
plt.scatter(wg_train_data["metmin"],wg_fitted_simp_nonlin_model2, c = "red")
wg_pred_simp_nonlin_model2 = wg_simp_nonlin_model2.predict(wg_test_data)
MAPE(wg_test_data["wg"],wg_pred_simp_nonlin_model2) # 16.9% Mean APE, 5.4% Median APE

#####3 Log transformations
## Linear - Linear: wg = a*metmin + C
plt.scatter(wgclean["metmin"], wgclean["wg"]) # non linear relationship
## Linear - Log: wg = a*log(metmin) + C
plt.scatter(wgclean["metmin_log"], wgclean["wg"]) # non linear
## Log - Linear: log(wg) = a*metmin + C
plt.scatter(wgclean["metmin"], wgclean["wg_log"]) # linear
## Log - Log: log(wg) = a*log(metmin) + C
plt.scatter(wgclean["metmin_log"], wgclean["wg_log"]) # Non linear

##### Log - Linear
wg_log_lin_model = ols("wg_log ~ metmin", data = wg_train_data).fit()
wg_log_lin_model.params
# log(wg) = -0.0011*metmin + 4.88
# Taking antilog: wg = exp(-0.0011*metmin + 4.88)
wg_log_lin_model.rsquared # 0.982; improved from 0.974 for 2nd order model
wg_fitted_simp_loglin_model2_log = wg_log_lin_model.predict(wg_train_data)
wg_fitted_simp_loglin_model2_antilog = np.exp(wg_fitted_simp_loglin_model2_log)

MAPE(wg_train_data["wg"],wg_fitted_simp_loglin_model2_antilog) # 6.6% Mean APE, 5.06% Median APE
plt.scatter(wg_train_data["metmin"],wg_train_data["wg"])
plt.scatter(wg_train_data["metmin"],wg_fitted_simp_loglin_model2_antilog, c = "red")

wg_pred_simp_loglin_model = wg_log_lin_model.predict(wg_test_data)
wg_pred_simp_loglin_model_antilog = np.exp(wg_pred_simp_loglin_model)

MAPE(wg_test_data["wg"],wg_pred_simp_loglin_model_antilog) # 6.5% Mean APE, 5.7% Median APE

####### Q: How do I know whether the coefficients are significant?
## Ans: Check the p values. They have to be less than 0.05
wg_simp_nonlin_model2.summary()
wg_log_lin_model.summary()


################# Multiple Regression

cement_data = pd.read_csv("data/cement.csv")

## Step 1
# DV: y (heat evolved)
# IDV: x1, x2, x3, x4 (composition of key ingredients)

## Step 2

cement_data.plot.scatter("x1","y")
cement_data.plot.scatter("x2","y")
cement_data.plot.scatter("x3","y")
cement_data.plot.scatter("x4","y")

for i in cement_data.columns[:4]:
    cement_data.plot.scatter(i,"y")

sns.pairplot(cement_data) # provides scatter plot for all combination


cement_data["x1"].corr(cement_data["y"]) # 0.7307; decent positive
cement_data["x2"].corr(cement_data["y"]) # 0.816; strong positive
cement_data["x3"].corr(cement_data["y"]) # -0.53; weak negative
cement_data["x4"].corr(cement_data["y"]) # -0.82; strong negative

cement_corr = cement_data.corr() #provides correlation of all combinations
# x1 and x3 are strongly correlated
# x2 and x4 are strongly correlated
# Only one variable has to be selected out of the 2 to remove redundancy

## Step 3
cement_multi_lin = ols("y ~ x1 + x2 + x3 + x4", data = cement_data).fit()
cement_multi_lin.params
# y = 1.55*x1 + 0.51*x2 + 0.1*x3 - 0.14*x4

## Step 4
cement_multi_lin.summary() # Adj R2: 0.974
# p values are high due to multi collinearity

## Check the Adj R2 for each of the following model and pick the model

## x1, x2
cement_x1x2 = ols("y ~ x1 + x2", data = cement_data).fit()
cement_x1x2.summary() # Adj R2: 0.974, p values close to 0

## x1, x4
cement_x1x4 = ols("y ~ x1 + x4", data = cement_data).fit()
cement_x1x4.summary() # Adj R2: 0.967, p values close to 0

## x2, x3
cement_x2x3 = ols("y ~ x2 + x3", data = cement_data).fit()
cement_x2x3.summary() # Adj R2: 0.816, p values close to 0

## x3, x4
cement_x3x4 = ols("y ~ x3 + x4", data = cement_data).fit()
cement_x3x4.summary() # Adj R2: 0.922, p values close to 0


##### Treating Nominal Categorical variables #########################
cat_gender_encoded = pd.get_dummies(catsdata["Gender"])
cats2 = pd.concat([catsdata,cat_gender_encoded], axis = 1)

cats_multi_lin_model = ols("Hwt ~ Bwt + M + F", data = cats2).fit()
cats_multi_lin_model.params

## Categorical variables typically have repeated values
catsdata.nunique()

############# Assignment (Availability prediction for AWS spot instance) ###

availability = pd.read_csv("data/availability.csv")
availability.nunique()
availability["Bid_sq"] = availability["Bid"]**2

## Step 1
# DV: availability
# IDVs: Bid price (continious), Spot Price (Ordinal categorical)

## Step 2
availability.plot.scatter("Bid","Availability")
availability.plot.scatter("Spotprice","Availability")
sns.lmplot("Bid","Availability", data = availability, hue = "Spotprice", fit_reg = False)
availability.corr()

## Step 3, Step 4
avail_train_data, avail_test_data = train_test_split(availability,
                                test_size = 0.3, random_state = 1234)

##### Simple Linear
avail_simp_lin_model = ols("Availability ~ Bid", data = avail_train_data).fit()
avail_simp_lin_model.summary()
# Availability = 49.9*Bid - 0.6215
# R-squared = 0.407 ; bad model
fittd_avail_tr_simple_lin = avail_simp_lin_model.predict(avail_train_data)
plt.scatter(avail_train_data["Bid"],avail_train_data["Availability"])
plt.scatter(avail_train_data["Bid"],fittd_avail_tr_simple_lin, c = "red")
pred_avail_te_simple_lin = avail_simp_lin_model.predict(avail_test_data)
MAPE(avail_test_data["Availability"],pred_avail_te_simple_lin)
## 341% Mean APE; 26.3% Median APE

###### Simple Non Linear
avail_simp_nonlin_model = ols("Availability ~ Bid + Bid_sq", 
                              data = avail_train_data).fit()
avail_simp_nonlin_model.summary()
# Adj Rsq: 0.630; decent model
fittd_avail_tr_simple_nonlin = avail_simp_nonlin_model.predict(avail_train_data)
plt.scatter(avail_train_data["Bid"],avail_train_data["Availability"])
plt.scatter(avail_train_data["Bid"],fittd_avail_tr_simple_lin, c = "red")
plt.scatter(avail_train_data["Bid"],fittd_avail_tr_simple_nonlin, c = "green")
pred_avail_te_simple_nonlin = avail_simp_nonlin_model.predict(avail_test_data)
MAPE(avail_test_data["Availability"],pred_avail_te_simple_nonlin)
## 157% Mean APE; 19.16% Median APE

######  Multiple Non Linear
avail_multi_nonlin_model = ols("Availability ~ Bid + Bid_sq + Spotprice", 
                              data = avail_train_data).fit()
avail_multi_nonlin_model.summary()
# Adj R2: 0.791

pred_avail_te_multi_nonlin = avail_multi_nonlin_model.predict(avail_test_data)
MAPE(avail_test_data["Availability"],pred_avail_te_multi_nonlin)
## 184% Mean APE; 16.79% Median APE






