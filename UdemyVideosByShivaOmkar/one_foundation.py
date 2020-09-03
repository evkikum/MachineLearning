#%%    # Import libraries
import os
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
#from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
#import seaborn as sns

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

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value)

#%%  Panda data frame explorations
# Read data
train = pd.read_csv("./data/mpg.csv")
train.head()
train.columns = map(str.upper, train.columns)
train.head()

#Infer few Constants from above
catColumns = ['ORIGIN']; strResponse = 'MPG'

# First view
train.dtypes
train.shape
train.index # get row index # train.index[0] # 1st row label # train.index.tolist() # get as a list
train.info()

# Change data types
#catColumns = list(set(train.columns).intersection(catColumns));
train[catColumns] = train[catColumns].apply(lambda x: x.astype('category'))

# View summary
print(train.describe(include = 'all'))

# Few basic statistics
cr = train.corr() # pairwise correlation cols
cr
train.kurt() # kurtosis over cols (def)

# Data Extraction. Note: label slices are inclusive, integer slices exclusive
train.loc[0,'MPG'] # by labels
train.iloc[0,0]
train.iloc[0, :] # 0th row and all column
train.iloc[:, 0] # 0th column and all row
train.at[0,'MPG']
#train.ix[0,'MPG'] #Depricated. mixed label and integer position indexing

#Column level
train.MPG.head()
train['MPG'].head()

# Few more operations at data frame level
train.count()
train.min()

# Few more operations at column level
train['MPG'].idxmin() # get the index number where minimum is present
train['MPG'].where(train['MPG']>15)
train['MPG'].where(train['MPG']>15,other=0)
train[1:3] # 2 rows excluding row number 3. label slices are inclusive, integer slices exclusive.

# Creation of new column
train['temp_col'] = train.MPG / train.CYLINDERS
train.head()
train = train.drop('temp_col', axis=1)

train['temp_col'] = train['MPG'] / train['CYLINDERS']
train.drop('temp_col', axis=1, inplace=True)

# to get array of index where criteria mets
a = np.where(train['MPG'] > 15) # a is tuple(immutable, sequences like lists, parentheses lists use square brackets)
type(a), type(a[0])
len(a[0])

# Get index as one column so that it will help in merge
train = train.reset_index()
train.head()
train.drop('index', axis=1, inplace=True)

# strings operation
train['ORIGIN'] = train['ORIGIN'].str.lower() # upper, contains, startswith, endswith, replace('old', 'new'), extract('(pattern)')
train.head()

#%% Merge and concate various ways
df_hr = pd.DataFrame({'NAME':['A','B'], 'AGE': [35, 46]})
df_sal = pd.DataFrame({'NAME':['A','B'], 'SALARY': [1000, 2000]})

# Merge column wise
pd.merge(df_hr, df_sal, on='NAME') # cbind

#Simple concatenation is often the best
df_hr_2 = pd.DataFrame({'NAME':['C','D'], 'AGE': [25, 40]})
pd.concat([df_hr, df_hr_2],axis=0) #top/bottom

df_proj = pd.DataFrame({'PROJ':['R','Python']})
pd.concat([df_hr,df_proj],axis=1)#left/right

#%% Various Tables
train.head(2)

train_summary = train.groupby('ORIGIN').size()
train_summary

train_summary = train.groupby('ORIGIN')['MPG'].agg(['mean','count'])
train_summary

train_summary = train.groupby('ORIGIN')['MPG'].agg(['sum','mean','count'])
train_summary
#%% Basic plots
numericColumn = 'HORSEPOWER'

# Scatter
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
train.plot.scatter(x=numericColumn, y=strResponse, ax = ax)
plt.ylabel(strResponse, size=10)
plt.xlabel(numericColumn, size=10)
plt.title(numericColumn + " vs " + strResponse + " (Response variable)", size=10)
plt.tight_layout()  # To avoid overlap of subtitles
plt.show()

# Histogram
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
train[numericColumn].plot.hist(bins=10, color='blue')  # alpha=0.5
plt.ylabel('Count', size=10)
plt.xlabel(numericColumn, size=10)
plt.title("Distribution of " + numericColumn, size=10)
plt.tight_layout()  # To avoid overlap of subtitles
plt.show()

# Box Plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
bp = train[numericColumn].plot.box(sym='r+', showfliers=True, return_type='dict')
plt.setp(bp['fliers'], color='Tomato', marker='*')
plt.ylabel('Count', size=10)
plt.title("Distribution of " + numericColumn, size=10)
plt.tight_layout()  # To avoid overlap of subtitles
plt.show()

#Class work: Box Plot for 'HORSEPOWER' and 'WEIGHT' together

#Class work: For all features in one view. Run it and write the explanation
train.hist()
plt.show()

#Class work: For all features in one view. Run it and write the explanation
scatter_matrix(train)
plt.show()
#%% Class work: Open one_foundation_numpy.py and Practice