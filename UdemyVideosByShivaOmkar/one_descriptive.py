#%%    # Import libraries
import os
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

# Working directory
os.chdir("/home/evkikum/Desktop/Data Science/Python/Udemy_Shiv")
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
catColumns = ['ORIGIN']; strResponse = 'MPG'

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value)
#%%  Read data, Descriptive Analysis
train = pd.read_csv("./data/mpg.csv")
train.columns = map(str.upper, train.columns)

# First view
train.info()
train.dtypes
train.head(2)
train.index # get row index # train.index[0] # 1st row label # train.index.tolist() # get as a list
train.shape

# Change data types
#catColumns = list(set(train.columns).intersection(catColumns));
train[catColumns] = train[catColumns].apply(lambda x: x.astype('category'))

# View summary
print(train.describe(include = 'all'))

# Description: This provides data dictioery for given data table.
dataDic = GetDataDictionary(train)
dataDic.to_csv("train_DataDictionary.csv", index=False)

# Descriptive Analysis
# Need seaborn
Desc_Numeric_Single(train, strResponse= strResponse, folderImageDescriptive = "./Images/")

# Do not Need seaborn
Desc_Numeric_Double(train, strResponse= strResponse, folderImageDescriptive = "./Images/")
Desc_Categorical_Single(train, strResponse= strResponse, folderImageDescriptive = "./Images/")
Desc_Categorical_Double(train, strResponse= strResponse, folderImageDescriptive = "./Images/")

# Need seaborn
Desc_Numeric_AllatOnce(train, strResponse= strResponse, folderImageDescriptive = "./Images/", folderOutput = "./Images/")

#%% PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#  Read data afresh
train = pd.read_csv("./data/mpg.csv")
train.columns = map(str.upper, train.columns)

# Drop categorical columns
train.drop(catColumns, axis=1, inplace=True)

# List of independent variables
#listAllPredictiveFeatures = list(set(train.columns) - set([strResponse]))
listAllPredictiveFeatures = np.setdiff1d(train.columns,strResponse)

n_components = 2
# create instance of PCA object
pca = PCA(n_components=n_components)

# Fit the model with X and apply the dimensionality reduction on X.
train_pca = pca.fit_transform(train[listAllPredictiveFeatures])

#Cumulative Variance explains
cumVarExplained = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
cumVarExplained # array([99.76, 99.97])

#Plot the cumulative explained variance as a function of the number of components
plt.subplots(figsize=(13, 9))
plt.plot(range(1,n_components+1,1), cumVarExplained, 'bo-')  #
plt.ylabel('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.xticks(np.arange(1, n_components+1, 1))
plt.title("PCA: Number of Features vs Variance (%)")
plt.tight_layout()  # To avoid overlap of subtitles
plt.show()

# Prepapre for return
train_pca = pd.DataFrame(train_pca, columns= paste(["PC"] * n_components, np.arange(1, n_components+1, 1), sep=''))  #   # ('string\n' * 4)[:-1]
train_pca.head(5)
#%% Outliers by IsolationForest
from sklearn.ensemble import IsolationForest

#  Read data afresh
train = pd.read_csv("./data/mpg.csv")
train.columns = map(str.upper, train.columns)

# Drop categorical columns
train.drop(catColumns, axis=1, inplace=True)

# List of independent variables
#listAllPredictiveFeatures = list(set(train.columns) - set([strResponse]))
listAllPredictiveFeatures = np.setdiff1d(train.columns,strResponse)

# Get fraction of outliers
outliers_fraction = 0.02 # 5%

# define outlier detection
clf = IsolationForest(max_samples=train.shape[0], contamination=outliers_fraction)

# Fit the model
clf.fit(train[listAllPredictiveFeatures])  # fit the data and tag outliers
scores_pred = clf.decision_function(train[listAllPredictiveFeatures]) # Average outlier score

# get count of outliers
outliers = scores_pred <= np.percentile(scores_pred,(1-outliers_fraction))
len(outliers)
np.unique(outliers, return_counts=True)
#HW: See the outlier rows

# Prepare for the outliers
train_pca['color'] = outliers
mapping = {False:'black' ,True :'red'}
train_pca['color'] = train_pca['color'].replace(mapping)
#train_pca['color'] = train_pca['color'].to_string()
train_pca.dtypes
train_pca.head()

# View the outliers
plt.title("Outliers by IsolationForest")
plt.scatter(train_pca['PC1'], train_pca['PC2'], c=train_pca['color'])
plt.axis('tight')
plt.show()
