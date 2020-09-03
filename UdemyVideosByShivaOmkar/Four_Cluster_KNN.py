#%%    # Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc; gc.enable()
from sklearn.neighbors import NearestNeighbors

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value)
#%% Temporary data
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# Class work: See the data in scatter plot

# Do KNN
nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto') # 3-4 other algorithums are present
nbrs.fit(X)

# get Neighbours
distances, indices = nbrs.kneighbors(X)

# Convert to DF for easy access for plotting
df = pd.DataFrame(X)
df.columns = ['C1', 'C2']
df = df.reset_index()

# Plot the knn on 2-D
fig, ax = plt.subplots()
ax.scatter(df.C1, df.C2)

# Add label
for i, txt in enumerate(df.index):
    txt_display = str(txt) + ', knn pair: ' + str(indices[i])
    ax.annotate(txt_display, (df.loc[i,'C1'],df.loc[i,'C2']))

plt.title('The KNN on 2-D')
plt.tight_layout()  # To avoid overlap of subtitles
plt.show()

del(X, df, distances, i, indices); gc.collect()
#Class work: Do the above for Iris data set using file 'Four_Cluster_KNN_Iris.py'