
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.model_selection import train_test_split, cross_val_score # latest version of sklearn
from sklearn.cross_validation import train_test_split, cross_val_score # older version of sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc


## Step 0: Business understanding, data preparation, data cleaning
## Step 1: Know the DVs and IDVs
   # IDVs have to be scaled for algorithms like KNN
## Step 2: Exploratory analysis
   # Groupby aggregate
   # Boxplot
   # Scatter Plot
   # Class imbalance
     # To handle class imbalance, 
       # down sample over represented class - random sampling
       # up sample under represented class - SMOTE
## Step 3: Building Model
   # Train Test Split
   # Build model on training data
     # K Nearest Neighbors
     # Decision tree
     # Random Forest
     # Boosting algorithms
     # Logistic Regression
## Step 4: Model Evaluation    
   # Confusion Matrix
   # Accuracy, True Positive Rate, False Positive Rate, ROC, AUC
   # Model fine tuning - Hyperparameter tuning
   # Cross validation
     # k fold cross validation; 5 fold cross validation
## Step 5: Go live and start predicting


irisdata = pd.read_csv("data/iris.csv")

## Step 1:
# DV: Species (setosa, versicolor, virginica)
# IDV: S.L, S.W, P.L, P.W
# IDVs need not be scaled as they are all measured in cms

## Step 2:
iris_class_summary = irisdata.groupby("Species").agg(np.mean)
for i in iris_class_summary.columns:
    plt.figure()
    iris_class_summary[i].plot.bar()

iris_class_summary2 = irisdata.groupby("Species").agg([min, max])

for i in irisdata.columns[:4]:
    irisdata.boxplot(column = i, by = "Species")

sns.lmplot("Petal.Length","Petal.Width", data = irisdata,
           hue = "Species", fit_reg = False)
sns.lmplot("Sepal.Length","Sepal.Width", data = irisdata,
           hue = "Species", fit_reg = False)

irisdata["Species"].value_counts()

## Step 3:
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(
        irisdata.iloc[:,:4], irisdata["Species"], test_size = 0.3, 
        random_state = 1234)

# Building the model on training data

## KNN
iris_knn1 = KNeighborsClassifier(n_neighbors = 1).fit(X_iris_train,y_iris_train)
iris_knn2 = KNeighborsClassifier(n_neighbors = 2).fit(X_iris_train,y_iris_train)

## Decision Tree
iris_dtree1 = DecisionTreeClassifier(
        max_depth = 2, random_state = 1234).fit(
                X_iris_train,y_iris_train)
iris_dtree1.feature_importances_
iris_dtree2 = DecisionTreeClassifier(criterion = "entropy",
        max_depth = 2, random_state = 1234).fit(
                X_iris_train,y_iris_train)
iris_dtree2.feature_importances_
iris_dtree3 = DecisionTreeClassifier(
        max_depth = 3, random_state = 1234).fit(
                X_iris_train,y_iris_train)
iris_dtree3.feature_importances_

##  Random Forest
iris_rf = RandomForestClassifier(n_estimators = 5, max_depth = 2).fit(
        X_iris_train,y_iris_train)
iris_rf.feature_importances_

## Gradient Boosting
iris_gbm = GradientBoostingClassifier(n_estimators = 20).fit(
        X_iris_train,y_iris_train)
iris_gbm.feature_importances_

## Step 4:

## KNN
iris_pred_tr = iris_knn1.predict(X_iris_train)
pd.crosstab(y_iris_train, iris_pred_tr) # 100% accuracy on training data with k = 1

iris_pred_tr2 = iris_knn2.predict(X_iris_train)
pd.crosstab(y_iris_train, iris_pred_tr2)
102/105 # 97% accuracy on training data with k = 2

iris_pred_te = iris_knn1.predict(X_iris_test)
pd.crosstab(y_iris_test, iris_pred_te)
44/45 # 97.7% accuracy on test data with k = 1

iris_pred_te2 = iris_knn2.predict(X_iris_test)
pd.crosstab(y_iris_test, iris_pred_te2)
44/45 # 97.7% accuracy on test data with k = 2
accuracy_score(y_iris_test, iris_pred_te2) # inbuilt function

## Decision Tree
iris_pred_dtree1 = iris_dtree1.predict(X_iris_test)
pd.crosstab(y_iris_test, iris_pred_dtree1)
accuracy_score(y_iris_test, iris_pred_dtree1) #95.55

iris_pred_dtree2 = iris_dtree2.predict(X_iris_test)
pd.crosstab(y_iris_test, iris_pred_dtree2)
accuracy_score(y_iris_test, iris_pred_dtree2) #95.55

iris_pred_dtree3 = iris_dtree3.predict(X_iris_test)
pd.crosstab(y_iris_test, iris_pred_dtree3)
accuracy_score(y_iris_test, iris_pred_dtree3) #97.77

## Random Forest
iris_pred_rf = iris_rf.predict(X_iris_test)
pd.crosstab(y_iris_test, iris_pred_rf)
accuracy_score(y_iris_test, iris_pred_rf) # 95.5

## Gradient Boosting
iris_pred_gbm = iris_gbm.predict(X_iris_test)
pd.crosstab(y_iris_test, iris_pred_gbm)
accuracy_score(y_iris_test, iris_pred_gbm) # 95.5

## Cross validation
np.mean(cross_val_score(KNeighborsClassifier(n_neighbors = 1),  # algorithm
                irisdata.iloc[:,:4], # IDV
                irisdata["Species"], # DV
                cv = 5) # number of folds
                    ) #96% cross validated accuracy

## Parameter tuning
for k in range(1,11):
    print("K = ", k,
          "Accuracy = ",
          np.mean(cross_val_score(KNeighborsClassifier(n_neighbors = k),  # algorithm
                irisdata.iloc[:,:4], # IDV
                irisdata["Species"], # DV
                cv = 5) # number of folds
                    ))
# k = 6 is the optimal parameter with a cross validated accuracy of 98%

## Decision Tree
for depthi in range(1,11):
    print("Max depth = ", depthi,
          "Accuracy = ",
          np.mean(cross_val_score(DecisionTreeClassifier(
                  max_depth = depthi, random_state = 1234),  # algorithm
                irisdata.iloc[:,:4], # IDV
                irisdata["Species"], # DV
                cv = 5) # number of folds
                    ))
# Max depth = 4 is optimal
    
## Random Forest
for n_est in range(1,11):
    print("Number of estimators = ", n_est,
          "Accuracy = ",
          np.mean(cross_val_score(RandomForestClassifier(
                  n_estimators = n_est, random_state = 1234),  # algorithm
                irisdata.iloc[:,:4], # IDV
                irisdata["Species"], # DV
                cv = 5) # number of folds
                    ))
# number of estimators = 5 looks to be the optimal parameter
    
## Gradient Boosting
for n_est in range(1,20):
    print("Number of estimators = ", n_est,
          "Accuracy = ",
          np.mean(cross_val_score(GradientBoostingClassifier(
                  n_estimators = n_est, random_state = 1234),  # algorithm
                irisdata.iloc[:,:4], # IDV
                irisdata["Species"], # DV
                cv = 5) # number of folds
                    ))
# number of boosting stages = 4 looks to be the optimal
    
##################### Wine Data set ############################################
    
winedata = pd.read_csv("data/wine.data", header = None)
winedata.columns = ["Wine_Class", "Alcohol","Malic_acid","Ash","Alcalinity_of_ash",
                       "Magnesium","Total_phenols","Flavanoids","Nonflavanoid_phenols",
                       "Proanthocyanins","Color_intensity","Hue","OD280_OD315",
                       "Proline"]

## Step 1:
# DV: Wine Class (1, 2, 3)
# IDV: 13 attributes
  # IDVs need scaling while using KNN

################## Diabetes Data ##########################################

diabetes_data = pd.read_csv("data/diabetes_data.csv")

## Step 1:
# DV: Class (1 - diabetic, 0 - non diabetic)
# IDVs: 8 (diagnosis results)

## Step 2:
diabetes_class_properties = diabetes_data.groupby("Class").agg([min, np.mean, max])

for i in diabetes_data.columns[:8]:
    diabetes_data.boxplot(column = i, by = "Class")

sns.lmplot("BMI","Age", data = diabetes_data,
           fit_reg = False, hue = "Class")

diabetes_data["Class"].value_counts() # 262 non diabetic, 130 diabetic
262/394 # 66:33 There is some class imbalance

## Step 3 & 4:
for n_est in range(1,21):
    print("Number of estimators = ", n_est,
          "Accuracy = ",
          np.mean(cross_val_score(RandomForestClassifier(
                  n_estimators = n_est, random_state = 1234),  # algorithm
                diabetes_data.iloc[:,:8], # IDV
                diabetes_data["Class"], # DV
                cv = 5) # number of folds
                    ))
# number of estimators = 9 is optimal which gives a cross validated accuracy of 79%
    
X_diab_train, X_diab_test, y_diab_train, y_diab_test = train_test_split(
        diabetes_data.iloc[:,:8], diabetes_data["Class"], test_size = 0.3, 
        random_state = 1234)
y_diab_train.value_counts() # 177 non diabetic and 97 diabetic


diab_rf = RandomForestClassifier(n_estimators = 9, random_state = 1234).fit(
        X_diab_train, y_diab_train)
# Feature Importance
pd.Series(diab_rf.feature_importances_,
          index = diabetes_data.columns[:8]).sort_values(ascending = False)

pred_class_rf = diab_rf.predict(X_diab_test)
y_diab_test.value_counts() # 85 non diabetic, 33 diabetic patients
pd.crosstab(y_diab_test, pred_class_rf)
# Out of 85 non diabetic patients, 15 got incorrectly classified
15/85 # False Positive Rate: 17.6%
# Out of 33 diabetic patients, 23 got correctly classified
23/33 # True Positive Rate: 69.69%
accuracy_score(y_diab_test, pred_class_rf) # 78.8% accuracy

# Random Sampling
arr = np.array([1,4,7,2,5,9,10,88,76,54,34])
# random sampling of 5 values from arr without replacement
np.random.choice(arr,5,replace = False) 
# random sampling of 5 values from arr with replacement (same value can be picked multiple times)
np.random.choice(arr,5,replace = True) 

################# Logistic Regression ####################################
diab_logit = LogisticRegression().fit(X_diab_train, y_diab_train)
pd.Series(diab_logit.coef_[0], index = diabetes_data.columns[:8])
# log odds of being diabetic = 0.02*Plasma Glucose Concentration + .... + 0.0057*Age

pred_prob = diab_logit.predict_proba(X_diab_test) ## predicted probability
pred_prob_diab = pred_prob[:,1] # extracting probability of Class 1
pred_class_diab = np.zeros(len(y_diab_test))
pred_class_diab[pred_prob_diab >= 0.5] = 1 # cutoff of 0.5
pd.crosstab(y_diab_test,pred_class_diab)
# Out of 85 non diabetic patients, 11 got incorrectly classified
11/85 # False Positive Rate: 12.9%
# Out of 33 diabetic patients, 20 got correctly classified
20/33 # True Positive Rate: 60.6%
accuracy_score(y_diab_test,pred_class_diab) # 79.6%

pred_class_diab = diab_logit.predict(X_diab_test) # by default assumes 0.5 cutoff and predicts class
pd.crosstab(y_diab_test,pred_class_diab) # same result as above

######## TPR vs FPR Tradeoff 
  # If TPR increases FPR will also increase
## Increase True Positive Rate: By reducing the cutoff
pred_class_diab2 = np.zeros(len(y_diab_test))
pred_class_diab2[pred_prob_diab >= 0.3] = 1 # cutoff of 0.3
pd.crosstab(y_diab_test,pred_class_diab2)
# Out of 85 non diabetic patients, 31 got incorrectly classified
31/85 # False Positive Rate: 36.4%
# Out of 33 diabetic patients, 29 got correctly classified
29/33 # True Positive Rate: 87.8%
accuracy_score(y_diab_test,pred_class_diab2) # 70.6%

## Decrease False Positive Rate: By increasing the cutoff
pred_class_diab3 = np.zeros(len(y_diab_test))
pred_class_diab3[pred_prob_diab >= 0.7] = 1 # cutoff of 0.3
pd.crosstab(y_diab_test,pred_class_diab3)
# Out of 85 non diabetic patients, 3 got incorrectly classified
3/85 # False Positive Rate: 0.3%
# Out of 33 diabetic patients, 8 got correctly classified
8/33 # True Positive Rate: 24.2%
accuracy_score(y_diab_test,pred_class_diab3) # 76.2%

## ROC Curve
diab_fpr, diab_tpr, diab_thresholds = roc_curve(y_diab_test,pred_prob_diab)
plt.plot(diab_fpr,diab_tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

## Area Under the Curve
# will be between 0.5 and 1
  # 0.5: Bad (Random) model
  # 1: Ideal model
auc(diab_fpr,diab_tpr) # 0.83










