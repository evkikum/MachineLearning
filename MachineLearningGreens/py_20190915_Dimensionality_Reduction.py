
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import seaborn as sns
from statsmodels.formula.api import ols # https://www.statsmodels.org/stable/index.html



## Step 0: Data preparation, business understanding
## Step 1: Scale the data if needed
## Step 2: Exploratory Analysis: Correlation analysis
   # If there are strong correlations between the variables, then higher compression can be expected
   # If there are only weak correlations btw variables, reduction will be less
## Step 3: Build PCA transformation and project to new dimension space
## Step 4: Analyze the variance captured by each new dimension and reduce accordingly
## Step 5: Feed the new dimensions to a downstream process
   # Visualization
   # Clustering
   # Regression
## Step 6: Factor Analysis: How to trace back to original variables

irisdata = pd.read_csv("data/iris.csv")
newiris = irisdata.iloc[:,:4]  # Extracting S.L, S.W, P.L and P.W

#################33 VARIABLE SELECTION ###############################
# From the 4 variables, can you select 2 representative variables?
# Petal Length is a representative of S.L, P.W
# Sepal Width do not correlate with others and hence carries unique info
iriscorr = newiris.corr()

############### PCA #######################################
## Step 1: Scaling not needed as all measurements in cms

## Step 2:
 # There is strong correlation  btw P.L, S.W and S.W. So good compression can be expected

## Step 3
irispca = PCA(n_components = 4).fit(newiris) 
iris_projected = pd.DataFrame(irispca.transform(newiris))
iris_projected.corr() # correlation between the variables will be 0

## Step 4
irispca.explained_variance_ratio_
# Dim 1 captures 92.4% of variance
# Dim 2 captures 5.3% of variance
# Dim 3 captures 1.7% of variance
# Dim 4 captures 0.5% of variance

sum(irispca.explained_variance_ratio_) #all dimensions will be needed to capture 100% variance
np.cumsum(irispca.explained_variance_ratio_)
# First 2 dimensions captures 97.7% of variance
irispca2 = PCA(n_components = 2).fit(newiris) 
iris_projected = pd.DataFrame(irispca2.transform(newiris),
                              columns = ["Dim1","Dim2"])

## Step 5: 

# Visualization
iris_projected.plot.scatter("Dim1","Dim2")

# Clustering
iris_pca_kmeans = KMeans(n_clusters = 3, random_state = 1234).fit(iris_projected)
iris_projected["Label"] = iris_pca_kmeans.labels_
sns.lmplot("Dim1","Dim2", data = iris_projected, 
           fit_reg = False, hue = "Label")
# Cluster 0: Less in Dim 1
# Cluster 1: High in Dim 1 and High in Dim 2
# Cluster 2: High in Dim 1 and Less in Dim 2

## Step 6
irispca2.components_ # K x D matrix
irispca2.components_.T# D x K matrix
# Dim1 = 0.36*S.L - 0.08*S.W + 0.85*P.L + 0.35*P.W
# Dim 1 is dominated by P.L

# Dim2 = 0.65*S.L + 0.73*S.W - 0.17*P.L - 0.07*P.W
# Dim 2 is dominated by S.W followed by S>L

# Cluster 0: Less in P.L
# Cluster 1: High in P.L and High in S.W, S.L
# Cluster 2: High in P.L and Less in S.W, S.L


###############
mtcars = pd.read_csv("data/mtcars.csv")
mtcars_idv = mtcars.iloc[:,1:]
# Can you reduce the 10 IDVs into a smaller set?

mtcars_corr = mtcars_idv.corr()

### PCA
mtcars_idv_scaled = scale(mtcars_idv)
mtcars_pca = PCA(n_components = 10).fit(mtcars_idv_scaled)
mtcars_pca.explained_variance_ratio_
np.cumsum(mtcars_pca.explained_variance_ratio_)
# 3 dimensions are needed to capture 90% of variance
mtcars_pca = PCA(n_components = 3).fit(mtcars_idv_scaled)
mtcars_projected = pd.DataFrame(mtcars_pca.transform(mtcars_idv_scaled),
                    columns = ["Dim1","Dim2","Dim3"])
mtcars_projected["mpg"] = mtcars["mpg"]

### Regression
mtcars_projected.plot.scatter("Dim1","mpg")
mtcars_projected.plot.scatter("Dim2","mpg")
mtcars_projected.plot.scatter("Dim3","mpg")
mtcars_proj_corr = mtcars_projected.corr()

mtcars_pca_regression = ols("mpg ~ Dim1 + Dim2 + Dim3", 
                            data = mtcars_projected).fit()
mtcars_pca_regression.summary()
# Dim 2 is not statistically significant. Removing it from model
mtcars_pca_regression = ols("mpg ~ Dim1 + Dim3", 
                            data = mtcars_projected).fit()
mtcars_pca_regression.summary()
# Adj R2: 0.843
# mpg = 2.24*Dim1 - 1.27*Dim3 + 20.09

###################### wine data set ############################

## Step 0: Data preparation, business understanding
winedata = pd.read_csv("data/wine.data", header = None)
winedata.columns = ["Wine_Class", "Alcohol","Malic_acid","Ash","Alcalinity_of_ash",
                       "Magnesium","Total_phenols","Flavanoids","Nonflavanoid_phenols",
                       "Proanthocyanins","Color_intensity","Hue","OD280_OD315",
                       "Proline"]
newwine = winedata.iloc[:,1:]
# Can we reduce from 13 attributes?

## Step 1: Scale the data if needed
wine_scaled = pd.DataFrame(scale(newwine),
                           columns = newwine.columns)

## Step 2: Exploratory Analysis: Correlation analysis
   # If there are strong correlations between the variables, then higher compression can be expected
   # If there are only weak correlations btw variables, reduction will be less
wine_raw_corr = newwine.corr()
wine_scaled_corr = wine_scaled.corr()
# Correlation relationship do not change because of scaling
# Not many highly correlated variables. 
# PCA may not be able to compress this to a very small set of variables

## Step 3: Build PCA transformation and project to new dimension space
winepca = PCA().fit(wine_scaled)

## Step 4: Analyze the variance captured by each new dimension and reduce accordingly
winepca.explained_variance_ratio_
# Dim 1 captures 36.19% of variance
# Dim 2 captures 19.2% of variance
np.cumsum(winepca.explained_variance_ratio_)
# At least 5 new dimensions needed to capture 80% of the variance
winepca = PCA(n_components = 5).fit(wine_scaled)
wine_transformed = pd.DataFrame(winepca.transform(wine_scaled),
                columns = ["Dim" + str(i) for i in range(1,6)])

## Step 5: Feed the new dimensions to a downstream process
# Visualization
sns.pairplot(wine_transformed)
wine_transformed.plot.scatter("Dim1","Dim2")

# Clustering
wine_pca_clust = KMeans(n_clusters = 3, random_state = 1234).fit(wine_transformed)
wine_transformed_with_labels = wine_transformed.copy()
wine_transformed_with_labels["Label"] = wine_pca_clust.labels_
sns.lmplot("Dim1","Dim2", data = wine_transformed_with_labels,
           hue = "Label", fit_reg = False)
wine_pca_clust_profiling = wine_transformed_with_labels.groupby("Label").agg(np.mean)
# Cluster 0: Less in Dim 1; Less in Dim 2
# Cluster 1: High in Dim 1; Less in Dim 2
# Cluster 2: Middling in Dim 1; High in Dim 2

## Step 6: Factor Analysis: How to trace back to original variables

wine_factor_loadings = pd.DataFrame(winepca.components_.T,
                            index = newwine.columns,
                            columns = wine_transformed.columns)
# Dim 1 = 0.14*Alcohol - 0.2*Malic - ....... + 0.26*Proline
# Dim 2 = -0.48*Alcohol - 0.22*Malic Acid + ...... - 0.36*Proline

# Flavanoids, Total Phenols and OD280 dominates Dim 1
# Color intensity, Alcohol, Proline dominates Dim 2 (negative coefficients)

# Cluster 0: Less in Flavanoids, Total Phenols and OD280; High in Color intensity, Alcohol, Proline 
# Cluster 1: High in Flavanoids, Total Phenols and OD280; High in Color intensity, Alcohol, Proline 
# Cluster 2: Middling in Flavanoids, Total Phenols and OD280; Less in Color intensity, Alcohol, Proline 




