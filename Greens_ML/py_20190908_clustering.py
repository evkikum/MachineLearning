import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import seaborn as sns
import os

os.chdir(r"/home/evkikum/Desktop/Data Science/Python/GreenInstitute_Course")

## Step 0: Data Preparation, Business context understanding
## Step 1: Scale the data if variables are of diff measurement units
## Step 2: Exploratory analysis
## Step 3: Building cluster - kMeans
  # Choose optimal K by combining business inputs and elbow curve analysis
## Step 4: Profile the clusters

############### Euclidean Distance and Importance of Scaling ############
"""
Given is the age and income of 5 individuals (P1,P2,P3,P4 and P5)

Can you find a pair for each individual among the other 4 based
on the similarities in age and income?

Who is similar to P1?
"""
age = [32,45,28,60,55]
income = [25000,55000,35000,18000,42000]
plt.scatter(age,income)

d12 = ((32 - 45)**2 + (25000 - 55000)**2)**0.5 # 30000
d13 = ((32 - 28)**2 + (25000 - 35000)**2)**0.5 # 10000
d14 = ((32 - 60)**2 + (25000 - 18000)**2)**0.5 # 7000
d15 = ((32 - 55)**2 + (25000 - 42000)**2)**0.5 # 17000
# d14 is less which is counter intuitive
  # income is dominating age due to it's scale

## How to bring age and income to comparable scale? 
  ## Divide income by 10000
age = [32,45,28,60,55]
income = [25,55,35,18,42]
d12 = ((32 - 45)**2 + (25 - 55)**2)**0.5 # 30000
d13 = ((32 - 28)**2 + (25 - 35)**2)**0.5 # 10000
d14 = ((32 - 60)**2 + (25 - 18)**2)**0.5 # 7000
d15 = ((32 - 55)**2 + (25 - 42)**2)**0.5 # 17000
# d13 is less which makes sense

## if there are several variables, such division or multiplication is not feasible

## Scaling to 0 mean and unit variance is a very common approach
  ## (x - mu)/sigma; mu: mean; sigma: standard deviation
age_scaled = scale(age)
income_scaled = scale(income)
plt.scatter(age_scaled,income_scaled)

##############################################

irisdata = pd.read_csv("Data/iris.csv")
newiris = irisdata.iloc[:,:4]  # Extracting S.L, S.W, P.L and P.W
# Can we split the iris flowers into groups based on their homogenity in S.L, S.W, P.L and P.W

## Step 1: Scaling not needed for iris as all measurements in cm

## Step 2: 

newiris.plot.scatter("Sepal.Length","Sepal.Width")
newiris.plot.scatter("Petal.Length","Petal.Width")
newiris.plot.scatter("Petal.Length","Sepal.Length")
newiris.plot.scatter("Petal.Width","Sepal.Width")
# 4c2 combinations of plots are possible

sns.pairplot(newiris) # plots a matrix of scatter plots
iris_stats = irisdata.describe()

## Step 3:
iris_with_clu_label = newiris.copy()

irisclust2 = KMeans(n_clusters = 2, random_state = 1234).fit(newiris)
iris_with_clu_label["Label_k2"] = irisclust2.labels_
irisclust2.inertia_ # 152.3 within cluster distance

irisclust3 = KMeans(n_clusters = 3, random_state = 1234).fit(newiris)
iris_with_clu_label["Label_k3"] = irisclust3.labels_
irisclust3.inertia_ # 78.85 within cluster distance

## CHOOSING OPTIMAL K (ELBOW CURVE)
within_clust_dist = pd.Series(0.0, index = range(1,11))
for k in range(1,11):
    iris_anyK = KMeans(n_clusters = k, random_state = 1234).fit(newiris)
    within_clust_dist[k] = iris_anyK.inertia_
plt.plot(within_clust_dist)  
# Elbow point is around k = 2 and 3

## Step 4:
irisclust2_profile = iris_with_clu_label.groupby("Label_k2").agg(np.mean)
sns.lmplot("Petal.Length","Petal.Width", data = iris_with_clu_label,
           hue = "Label_k2", fit_reg = False)
sns.lmplot("Petal.Length","Sepal.Length", data = iris_with_clu_label,
           hue = "Label_k2", fit_reg = False)
sns.lmplot("Sepal.Length","Sepal.Width", data = iris_with_clu_label,
           hue = "Label_k2", fit_reg = False)
# Cluster 0: Less P.L, P.W, S.L; High S.W
# Cluster 1: High P.L, P.W, S.L; Less S.W

irisclust3_profile = iris_with_clu_label.groupby("Label_k3").agg(np.mean)
sns.lmplot("Petal.Length","Petal.Width", data = iris_with_clu_label,
           hue = "Label_k3", fit_reg = False)
sns.lmplot("Petal.Length","Sepal.Length", data = iris_with_clu_label,
           hue = "Label_k3", fit_reg = False)
# Cluster 0: Less P.L, P.W, S.L; High S.W
# Cluster 1: High P.L, P.W; Middling S.L; Less S.W
# Cluster 2: High P.L, P.W, S.L; Middling S.W
sns.lmplot("Sepal.Length","Sepal.Width", data = iris_with_clu_label,
           hue = "Label_k3", fit_reg = False)

###################### Wine dataset ########################################

## Step 0: Data Preparation, Business context understanding
  # https://archive.ics.uci.edu/ml/datasets/Wine
  # Even though the extension .data, it is still a csv file
  # Note that the data doesn't have header. It has to be added separately
  # 13 attributes are present from 2nd column. 1st column can be ignored for now
winedata = pd.read_csv("Data/wine.data", header = None)
winedata.columns = ["Wine_Class", "Alcohol","Malic_acid","Ash","Alcalinity_of_ash",
                       "Magnesium","Total_phenols","Flavanoids","Nonflavanoid_phenols",
                       "Proanthocyanins","Color_intensity","Hue","OD280_OD315",
                       "Proline"]
newwine = winedata.iloc[:,1:]

wine_stats_summary = newwine.describe()
 
## Step 1: Scale the data if variables are of diff measurement units
  # this data has to be scaled
wine_scaled = scale(newwine)
type(wine_scaled) # numpy matrix
wine_scaled = pd.DataFrame(wine_scaled,
                           columns = newwine.columns)
wine_scaled_stats_summary = wine_scaled.describe()
# all variables have become zero mean, unit variance

## Step 2: Exploratory analysis
  # 13 attributes. You will see the challenge in interpreting several scatter plots
newwine.plot.scatter("Flavanoids", "Proline")
wine_scaled.plot.scatter("Flavanoids", "Proline")
# Relationship remains the same whether on original or raw

sns.pairplot(newwine)

## Step 3: Building cluster - kMeans
  # Choose optimal K by combining business inputs and elbow curve analysis
within_clust_dist = pd.Series(0.0, index = range(1,11))
for k in range(1,11):
    wine_anyK = KMeans(n_clusters = k, random_state = 1234).fit(wine_scaled)
    within_clust_dist[k] = wine_anyK.inertia_
plt.plot(within_clust_dist)  
# Elbow point at k = 3
wineclust = KMeans(n_clusters = 3, random_state = 1234).fit(wine_scaled)

wine_with_label = newwine.copy()
wine_with_label["Cluster_Label"] = wineclust.labels_
# Labels can be attached back to original scale. 
 # Scaling was done only to handle the limitation of distance measurement

## Step 4: Profile the clusters
 
wine_with_label["Cluster_Label"].value_counts()
# 62 wine in Cluster 0
# 51 wine in Cluster 1
# 65 wine in Cluster 2
wine_with_label["Cluster_Label"].value_counts().plot.pie()

wine_clu_profile = wine_with_label.groupby("Cluster_Label").agg(np.mean)

# Cluster 0: High in alcohol, magnesium, Proline; Less in malic acid, alcalinity of ash
# Cluster 1: High in alcohol, malic acid, Color intensity; Less in phenols, Flavanoids
# Cluster 2: Less in alcohol, malic acid, magnesium

sns.lmplot("Flavanoids","Proline", data = wine_with_label,
           hue = "Cluster_Label", fit_reg = False)









