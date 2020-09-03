#%%    # Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

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

# KNN does not need response
train.drop(strResponse, axis=1, inplace = True)

# To view get the data on 2-D by pca
train = GetPCA(train, n_components=2, bScale = True, fileImageToSave = "./images/pca.png")

# Do KNN
nb = NearestNeighbors(n_neighbors=3, algorithm='auto') # 3-4 other algorithums are present
nb.fit(train)

# get Neighbours of sample data with row index 0,10,20,30,100
ar_count_neigh = np.array([0,10,20,30,100], dtype=np.int64)
distances, indices = nb.kneighbors(train.iloc[ar_count_neigh,:])

#Define color for each row of test data
colors = ['black', 'red', 'green', 'blue', 'pink']

# Plot the cluster on 2-D
fig, ax = plt.subplots()
ax.scatter(train.PC1, train.PC2, color = 'white')

# Add label for test data
for i in ar_count_neigh: # i =0
    ax.annotate(str(i), (train.loc[i,'PC1'],train.loc[i,'PC2']))

# Add neighbours points with different color
for i in range(len(ar_count_neigh)): # i = 0; train.loc[indices[i], ['PC1','PC2']]; test.iloc
    ax.scatter(train.loc[indices[i], 'PC1'], train.loc[indices[i], 'PC2'], color = colors[i])

plt.title('The KNN on 2-D')
plt.tight_layout()  # To avoid overlap of subtitles
plt.show()
