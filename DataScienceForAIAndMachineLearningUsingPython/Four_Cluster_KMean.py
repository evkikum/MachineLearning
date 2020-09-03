#%%    # Import libraries
import os
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

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

# Store actual to compare normalized_mutual_info_score
mapping = {'setosa' : 0, 'versicolor' : 1, 'virginica' : 2}
actual = train[strResponse].replace(mapping).astype(np.integer)

# Kmean does not need response
train.drop(strResponse, axis=1, inplace = True)

# scalling and centering required
train = ScaleAndCenter_NumericOnly(train)
train.describe()

#How to get right count of cluster: Elbow method
rangeKClusters = range(1,10)

# Calculate distances bewteen clusters
lsitMeandist=[]
for k in rangeKClusters:
    # Build model
    model=KMeans(n_clusters=k)
    model.fit(train)
    # Predict for existing data points
    clusassign=model.predict(train)
    # Take -> euclidean disctance of each data point from center -> select minimum distance
    # Take sum of those distances
    lsitMeandist.append(sum(np.min(cdist(train, model.cluster_centers_, 'euclidean'), axis=1))/ train.shape[0])
    # end of for k in rangeKClusters:

# Plot the K in Elbow image
plt.plot(rangeKClusters, lsitMeandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')
plt.tight_layout()  # To avoid overlap of subtitles
plt.show()

# Number of cluster from observation
nCluster = 3

# Run Kmeans with above count of cluster
model = KMeans(n_clusters = nCluster, precompute_distances = True, random_state = seed_value)
model.fit(train)
listClusters = model.predict(train)

#Class work: Both Species and Cluster relations same. How to identify relations
# Hint: Add in main train data frame and do the cross tab with Species and Cluster

# How good the Cluster with already identified species
normalized_mutual_info_score(actual, listClusters) # 0.74

#%% View the cluster on 2-D by PCA
from sklearn.decomposition import PCA

n_components = 2
# create instance of PCA object
pca = PCA(n_components=n_components)

# Fit the model with X and apply the dimensionality reduction on X.
train_pca = pca.fit_transform(train)

#Cumulative Variance explains
cumVarExplained = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
cumVarExplained # array([99.76, 99.97])

# Make DF for further use
train_pca = pd.DataFrame(train_pca, columns= paste(["PC"] * n_components, np.arange(1, n_components+1, 1), sep=''))  #   # ('string\n' * 4)[:-1]
train_pca.head(2)

#%% View cluster in 2 D
# Increase by one so that we can read apprpritely
listClusters = listClusters + 1

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

#%% Get attribute of Cluster

# Add cluster id as one column
train['CLUSTERID'] = listClusters
train.head()
df = getClusterContentWithMaxPercentageAttr(train, count_cluster = nCluster, col_cluster = 'CLUSTERID', FeaturesForCluster = train.columns.values, HeaderPrefix = "Cluster",TopNFactorCount = 5, listCoreExcluded = train.columns.values) #
df
#%% K-mode and K Prototypes clustering
