#from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neural_network import MLPClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF

#%%    # Import libraries
import os
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
#from sklearn.metrics import confusion_matrix
from pandas_ml import ConfusionMatrix
#from pandas_confusion import ConfusionMatrix

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
catColumns = ['SPECIES']; strResponse = 'SPECIES'

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value)
#%% Random forest
from sklearn.ensemble import RandomForestClassifier

# Read data afresh
train = pd.read_csv("./data/Iris.csv")
train.columns = map(str.upper, train.columns)
print(train.dtypes)

# Change data types
#catColumns = list(set(train.columns).intersection(catColumns));
catColumns = np.intersect1d(train.columns, catColumns)
train[catColumns] = train[catColumns].apply(lambda x: x.astype('category'))

#Segragate 85% and 15%
train ,test = train_test_split(train,test_size=0.15)

# Getting lists of IV
#listAllPredictiveFeatures = list(set(train.columns) - set([strResponse]))
listAllPredictiveFeatures = np.setdiff1d(train.columns,strResponse)

# Default training param for RF
param = {'n_estimators': 10, # The number of trees in the forest.
         'min_samples_leaf': 5, # The minimum number of samples required to be at a leaf node (external node)
         'min_samples_split': 10, # The minimum number of samples required to split an internal node (will have further splits)
         'max_depth': None, 'bootstrap': True, 'max_features': "auto", # The number of features to consider when looking for the best split
          'verbose': True} # , 'warm_start' : True

#Build model on training data
classifier = RandomForestClassifier(**param)
classifier = classifier.fit(train[listAllPredictiveFeatures],train[strResponse])

# Self predict
predictions = classifier.predict(train[listAllPredictiveFeatures])

# Compute confusion matrix
confusion_matrix = ConfusionMatrix(train[strResponse].tolist(), predictions.tolist())
confusion_matrix

# Predict on test data
predictions = classifier.predict(test[listAllPredictiveFeatures])
confusion_matrix = ConfusionMatrix(test[strResponse].tolist(), predictions.tolist())
confusion_matrix

# View confusion matrix
confusion_matrix.plot()
plt.show()

# normalized confusion matrix
confusion_matrix.plot(normalized=True)
plt.show()

#Statistics are also available as follows
confusion_matrix.print_stats()
cms = confusion_matrix.stats()
print("Overall Accuracy is ", round(cms['overall']['Accuracy'], 2),", Kappa is ", round(cms['overall']['Kappa'], 2))

df = cms['class'].reset_index()
df[df['index'].str.contains('Precision')]
df[df['index'].str.contains('Sensitivity')]
df[df['index'].str.contains('Specificity')] # 101 line
# http://pandas-ml.readthedocs.io/en/latest/conf_mat.html#confusion-matrix-and-class-statistics

# How to predict probabilities of each class
pred_prob = classifier.predict_proba(test[listAllPredictiveFeatures])
pred_class_index = np.argmax(pred_prob, axis=1)

# Get categories lebal so that aove index can be conveted to actual lebal
pred_class_label = np.array(train[strResponse].cat.categories)
pred_class = pred_class_label[pred_class_index]
pred_class
#%%CatBoost: A machine learning library to handle categorical (CAT) data automatically
# https://tech.yandex.com/catboost/
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

# Read data afresh
train = pd.read_csv("./data/Iris.csv")
train.columns = map(str.upper, train.columns)
train.dtypes

# Change data types
#catColumns = list(set(train.columns).intersection(catColumns));
#train[catColumns] = train[catColumns].apply(lambda x: x.astype('category'))

# Replace string with int. It is required by algorithm
train['SPECIES'].unique()
mapping = {'setosa' : 0, 'versicolor' : 1, 'virginica' : 2}
train[strResponse] = train[strResponse].replace(mapping)
train[strResponse] = train[strResponse].astype(np.integer)
train.dtypes

#Segragate 85% and 15%
train ,test = train_test_split(train,test_size=0.15)

# Getting lists of IV
listAllPredictiveFeatures = np.setdiff1d(train.columns,strResponse)

#The model need index of category features
train[listAllPredictiveFeatures].dtypes

# Through object identification
categorical_features_indices = list(np.where(train[listAllPredictiveFeatures].dtypes == np.object)[0])

## In generic: with assuption that non numeric is category
#categorical_features_indices_f = np.where(train[listAllPredictiveFeatures].dtypes != np.float64)[0]
#categorical_features_indices_i = np.where(train[listAllPredictiveFeatures].dtypes != np.int64)[0]
#categorical_features_indices = list(set(categorical_features_indices_f).intersection(categorical_features_indices_i))

# Basic sanity test
if len(categorical_features_indices) == 0:
    categorical_features_indices = None

#building model
model=CatBoostClassifier(iterations=50, depth=3, learning_rate=0.1, loss_function='MultiClass', random_seed = seed_value) # , loss_function='Logloss' CrossEntropy
model.fit(train[listAllPredictiveFeatures], train[strResponse],cat_features=categorical_features_indices) # ,plot=True

# Predict on test data
predictions = model.predict(test[listAllPredictiveFeatures])
confusion_matrix = ConfusionMatrix(test[strResponse].tolist(), np.ravel(predictions))
confusion_matrix

# View confusion matrix
confusion_matrix.plot()
plt.show()

# normalized confusion matrix
confusion_matrix.plot(normalized=True)
plt.show()

#Statistics are also available as follows
confusion_matrix.print_stats()
cms = confusion_matrix.stats()
print("Overall Accuracy is ", round(cms['overall']['Accuracy'], 2),", Kappa is ", round(cms['overall']['Kappa'], 2))

df = cms['class'].reset_index()
df[df['index'].str.contains('Precision')]
df[df['index'].str.contains('Sensitivity')]
df[df['index'].str.contains('Specificity')]

##Class work: confusion matrix after reverse mapping

# How to predict probabilities of each class
pred_prob = model.predict_proba(test[listAllPredictiveFeatures])
pred_class_indexes = np.argmax(pred_prob, axis=1)

# Get categories lebal so that aove index can be conveted to actual lebal
mapping_rev = {v:k for k,v in mapping.items()}
pred_class = [mapping_rev[pred_class_index] for pred_class_index in pred_class_indexes]
pred_class

#%% One class SVM
from sklearn import svm
# http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, classification_report

# Read data afresh
train = pd.read_csv("./data/Iris.csv")
train.columns = map(str.upper, train.columns)
print(train.dtypes)

#SVM Need the scled data and hence scalling and centering required
train = ScaleAndCenter_NumericOnly(train, strResponse)
train.head(2)
train.describe(include = 'all')

#OneClassSVM returns +1 or -1 to indicate whether the data is an "inlier" or "outlier"
#respectively. To make comparison easier replacing a matching +1 or -1 value.
mapping = {'setosa' : 1, 'versicolor' : -1, 'virginica' : -1}
train[strResponse] = train[strResponse].replace(mapping)
train.describe(include = 'all')

#Segragate
test = train[train[strResponse] == -1]; train = train[train[strResponse] == 1]
test.reset_index(drop = True, inplace = True) # # Get index as one column
test.head(2)
train.shape, test.shape

# Getting lists of IV
listAllPredictiveFeatures = np.setdiff1d(train.columns,strResponse)

#Build model on training data
#nu is "An upper bound on the fraction of training errors. Basically this means the
#proportion of outliers we expect in our data. This is an important factor to consider
#when assessing algorithms.
#kernel is the kernel type - a non-linear function to project the hyperspace to higher
#dimension. The default is rbf (RBF - radial basis function).
#gamma is a parameter of the RBF kernel type and controls the influence of individual
#training samples - this effects the "smoothness" of the model. A low value improves
# the smoothness and "generalizability" of the model, while a high value reduces it
# but makes the model "tighter-fitted" to the training data. Some experimentation is
# often required to find the best value.
classifier = svm.OneClassSVM(nu=0.05, kernel='rbf', gamma=min(0.05, 1/train.shape[1]))
classifier = classifier.fit(train[listAllPredictiveFeatures])

# Self predict
predictions = classifier.predict(train[listAllPredictiveFeatures])

# Compute confusion matrix
confusion_matrix(train[strResponse], predictions)
classification_report(train[strResponse], predictions)

#Statistics are also available as follows
print("Overall Accuracy is ", round(accuracy_score(train[strResponse], predictions), 2),", Kappa is ", round(abs(cohen_kappa_score(train[strResponse], predictions)), 2))

# Predict on test data
predictions = classifier.predict(test[listAllPredictiveFeatures])

# if all are predicted correctly (-1) then CM will not be genegated and hence add dummy
if len(np.unique(predictions)) == 1:
    predictions[0] = 1
    test[strResponse][1] = 1

confusion_matrix(test[strResponse], predictions)
classification_report(test[strResponse], predictions)

#Statistics are also available as follows
print("Overall Accuracy is ", round(accuracy_score(test[strResponse], predictions), 2),", Kappa is ", round(abs(cohen_kappa_score(test[strResponse], predictions)), 2))

#%% Logistic Regression using RF: Theory on PPT
#from sklearn.linear_model import LogisticRegression # Can use instead of RF but it is basic soln
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Read data afresh
train = pd.read_csv("./data/Iris.csv")
train.columns = map(str.upper, train.columns)
print(train.dtypes)

# Since Logistic Regression need binary outcome and hence prepraing data for accordingly
train = train[train[strResponse].isin(['setosa','versicolor'])]

# Convert 'setosa' to 0 and 'versicolor' to 1
mapping = {'setosa' : 0, 'versicolor' : 1}
train[strResponse] = train[strResponse].replace(mapping)
train[strResponse] = train[strResponse].astype(np.integer)
train.dtypes

## Calculate cut off probability with response equal to 1. Will be required later
cutoff = train[train[strResponse] == 1].shape[0]/train.shape[0]
cutoff

# Change data types
catColumns = np.intersect1d(train.columns, catColumns)
train[catColumns] = train[catColumns].apply(lambda x: x.astype('category'))

#Segragate 85% and 15%
train ,test = train_test_split(train,test_size=0.15)

# Getting lists of IV
#listAllPredictiveFeatures = list(set(train.columns) - set([strResponse]))
listAllPredictiveFeatures = np.setdiff1d(train.columns,strResponse)

# Default training param for RF
param = {'n_estimators': 10, # The number of trees in the forest.
         'min_samples_leaf': 5, # The minimum number of samples required to be at a leaf node (external node)
         'min_samples_split': 10, # The minimum number of samples required to split an internal node (will have further splits)
         'max_depth': None, 'bootstrap': True, 'max_features': "auto", # The number of features to consider when looking for the best split
          'verbose': True} # , 'warm_start' : True

#Build model on training data
classifier = RandomForestClassifier(**param)
classifier = classifier.fit(train[listAllPredictiveFeatures],train[strResponse])

# Self predict
predictions = classifier.predict(train[listAllPredictiveFeatures])

# Compute confusion matrix
confusion_matrix = ConfusionMatrix(train[strResponse].tolist(), predictions.tolist())
confusion_matrix

# Predict on test data
predictions = classifier.predict(test[listAllPredictiveFeatures])
confusion_matrix = ConfusionMatrix(test[strResponse].tolist(), predictions.tolist())
confusion_matrix

# View confusion matrix
confusion_matrix.plot()
plt.show()

# normalized confusion matrix
confusion_matrix.plot(normalized=True)
plt.show()

# How to predict probabilities of each class
pred_prob = classifier.predict_proba(test[listAllPredictiveFeatures])
pred_class = np.argmax(pred_prob, axis=1)
pred_class

