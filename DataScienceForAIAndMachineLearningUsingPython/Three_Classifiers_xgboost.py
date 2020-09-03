#%% xgboost
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix

# Constants
catColumns = ['SPECIES']; strResponse = 'SPECIES'

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value)

# Working directory
os.chdir("D:\Trainings\python")
exec(open(os.path.abspath('CommonUtils.py')).read())

# Read data afresh
train = pd.read_csv("./data/Iris.csv")
train.columns = map(str.upper, train.columns)
train.dtypes

# Replace string with int. It is required by algorithm
train['SPECIES'].unique()
mapping = {'setosa' : 0, 'versicolor' : 1, 'virginica' : 2}
train[strResponse] = train[strResponse].replace(mapping)
train[strResponse] = train[strResponse].astype(np.integer)
train.dtypes

# Change data types
#catColumns = list(set(train.columns).intersection(catColumns));
#train[catColumns] = train[catColumns].apply(lambda x: x.astype('category'))

#Segragate 85% and 15%
train ,test = train_test_split(train,test_size=0.15)

# Getting lists of IV
listAllPredictiveFeatures = np.setdiff1d(train.columns,strResponse)

# Get data in form DMatrix as required by xgboost
d_train = xgb.DMatrix(train[listAllPredictiveFeatures], label=train[strResponse])

# https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
params = {'eval_metric' : 'merror', 'eta': 0.3, 'min_child_weight': 1,
          'colsample_bytree': 1, 'subsample': 1, 'max_depth': 6, 'nthread' : 4,
          'booster' : 'gbtree', 'objective' : "multi:softmax", 'num_class' : 3} # multi:softprob

#Build model on training data
classifier = xgb.train(dtrain = d_train, params = params)

#Self Prediction
pred = classifier.predict(d_train)

# Predict on test data
d_test = xgb.DMatrix(test[listAllPredictiveFeatures])
predictions = classifier.predict(d_test)
confusion_matrix = ConfusionMatrix(test[strResponse].tolist(), predictions)
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
