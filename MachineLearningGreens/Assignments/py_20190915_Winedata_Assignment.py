# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 17:02:13 2019

@author: evkikum
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.decomposition import PCA
import os

os.chdir(r"C:\Users\evkikum\Documents\Python Scripts\GreenInstitute_Course")

winedata = pd.read_csv("data/wine.data")
winedata = winedata.iloc[:,1:]
winedata.columns = ["Alcohol","Malic_acid","Ash","Alcalinity_of_ash","Magnesium","Total_phenols","Flavanoids","Nonflavanoid_phenols","Proanthocyanins","Color_intensity","Hue","OD280_OD315_of_diluted_wines","Proline"]

winedata_stats = winedata.describe()
winedata_scale = scale(winedata)


sns.pairplot(winedata)

##Correlation
winedata_corr = winedata.corr()


##LETS CHOOSE OPTIMAL K (ELBOW CURVE)
within_clust_dist = pd.Series(0.0 , index = range(1,15))
for i in range(1,15):
        winedata_anyK = KMeans(n_clusters=i, random_state=1234).fit(winedata_scale)
        within_clust_dist[i] = winedata_anyK.inertia_
    
plt.plot(within_clust_dist)  ## ELBOW CURVE IS AROUND 3

wineclust3 = KMeans(n_clusters=3, random_state=1234).fit(winedata_scale)
winedata["Label_k"] = wineclust3.labels_
winedata["Label_k"].value_counts().plot.pie()

winedata_profile = winedata.groupby("Label_k").agg(np.mean)


## CLUSTER 0 : 
## CLUSTER 1 :
## CLUSTER 2 :
    
    
##LETS USE PCA APPROACH

winedata = pd.read_csv("data/wine.data")
winedata = winedata.iloc[:,1:]
winedata.columns = ["Alcohol","Malic_acid","Ash","Alcalinity_of_ash","Magnesium","Total_phenols","Flavanoids","Nonflavanoid_phenols","Proanthocyanins","Color_intensity","Hue","OD280_OD315_of_diluted_wines","Proline"]
winedata_scale = scale(winedata)


## NOT MANY ARE HIGHLY CORRELATED
winepca = PCA().fit(winedata_scale)
windata_projected = pd.DataFrame(winepca.transform(winedata_scale))
winepca.explained_variance_ratio_
np.cumsum(winepca.explained_variance_ratio_)

winepca = PCA(n_components= 5).fit(winedata_scale)
windata_projected = pd.DataFrame(winepca.transform(winedata_scale), columns = ["Dim1", "Dim2", "Dim3", "Dim4", "Dim5"])
winepca.explained_variance_ratio_
np.cumsum(winepca.explained_variance_ratio_)


within_clust_dist = pd.Series(0.0, index = range(1,15))

for i in range(1,15):
    winedata_anyK = KMeans(n_clusters = i, random_state = 1234).fit(windata_projected)
    within_clust_dist[i] = winedata_anyK.inertia_

plt.plot(within_clust_dist)

wineclust3 = KMeans(n_clusters=3, random_state=1234).fit(windata_projected)
windata_projected["Label"] = wineclust3.labels_
## winedata["Label"] = windata_projected["Label"]

sns.pairplot(windata_projected)


##Clustering
wineclust3_profile = windata_projected.groupby("Label").agg(np.mean)

wine_factor_loadings = pd.DataFrame(winepca.components_.T, index = winedata.columns, columns = win)


## Cluster 0 : 
## Cluster 1 :
## Cluster 2 :

