#%%    # Import libraries
import os
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

# Working directory
os.chdir("D:\Trainings\python")
exec(open(os.path.abspath('CommonUtils.py')).read())

# Some standard settings
plt.rcParams['figure.figsize'] = (13, 9) #(16.0, 12.0)
plt.style.use('ggplot')


#Set PANDAS to show all columns in DataFrame
pd.set_option('display.max_columns', None)
#Set PANDAS to show all rows in DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('precision', 2)

# Constants
strResponse = 'SPECIES'

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value)
#%% Read data afresh
train = pd.read_csv("./data/Iris.csv")
train.columns = map(str.upper, train.columns)
train.dtypes

# Cluster does not need response
train.drop(strResponse, axis=1, inplace = True)

# scalling and centering required
train = ScaleAndCenter_NumericOnly(train)
train.describe()

# Number of cluster from observation
nCluster = 3

# Run with above count of cluster
# Explanation of linkage
#ward minimizes the variance of the clusters being merged.
#Ward: Tends to look for spherical clusters, very cohesive inside and extremely
#differentiated from other groups. Another nice characteristic is that the method tends
#to find clusters of similar size. It works only with the Euclidean distance.

#average uses the average of the distances of each observation of the two sets.
#Average: Links clusters using their centroids and ignoring their boundaries. The method
#creates larger groups than the complete method. In addition, the clusters can be different
# sizes and shapes, contrary to the Ward’s solutions. Consequently, this average,
# multipurpose, approach sees successful use in the field of biological sciences.

#complete or maximum linkage uses the maximum distances between all observations of the two sets
#Complete: Links clusters using their furthest observations, that is, their most dissimilar
# data points. Consequently, clusters created using this method tend to be comprised of
# highly similar observations, making the resulting groups quite compact.

#Cosine (cosine): A good choice when there are too many variables and you worry that some
#variable may not be significant. Cosine distance reduces noise by taking the shape of the
# variables, more than their values, into account. It tends to associate observations that
# have the same maximum and minimum variables, regardless of their effective value.

model = AgglomerativeClustering(n_clusters = nCluster, linkage = 'ward') # ('average', 'complete', 'ward')
model.fit(train)
listClusters = model.labels_ + 1

# View the cluster on 2-D by pca
train_pca = GetPCA(train, n_components=2, bScale = False, fileImageToSave = "./images/pca.png")

# Add cluster column
mapping = {1:'black' ,2 :'red', 3:'green'}
train_pca['COLOR'] = listClusters
train_pca['COLOR'] = train_pca['COLOR'].replace(mapping)

# Plot the cluster on 2-D
fig, ax = plt.subplots()
ax.scatter(train_pca.PC1, train_pca.PC2, color = train_pca.COLOR)

# Add label
for i, txt in enumerate(listClusters):
    ax.annotate(txt, (train_pca.iloc[i,0],train_pca.iloc[i,1]))

plt.title('The cluster on 2-D')
plt.tight_layout()  # To avoid overlap of subtitles
plt.show()

#View Dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage

linkage_matrix = linkage(train, 'ward')

# calculate full dendrogram
plt.figure() # figsize=(25, 10)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('index')
plt.ylabel('distance')
dendrogram(linkage_matrix, leaf_rotation=90.,  leaf_font_size=8., truncate_mode='lastp',
           p = 12, show_contracted=True)
plt.show()

#Class work: Get quality of Clustering using normalized_mutual_info_score
#Class work: Difference between cluster assignment by KMeans and Agglomerative