# Libraries
import os
os.chdir("D:\Trainings\python")

import numpy as np

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value)
#%% Index, Slice and Reshape NumPy Arrays for Machine Learning in Python

#Machine learning data is represented as arrays.
#In Python, data is almost universally represented as NumPy arrays.
#If you are new to Python, you may be confused by some of the pythonic ways of accessing data, such as negative
#indexing and array slicing.

#How to convert your list data to NumPy arrays.
#How to access data using Pythonic indexing and slicing.
#How to resize your data to meet the expectations of some machine learning APIs.

#%%From List to Arrays

# list of data
data = [11, 22, 33, 44, 55]

# array of data
data = np.array(data)
print(data)
print(type(data))
data.shape

# list of data
data = [[11, 22], [33, 44], [55, 66]]

# array of data
data = np.array(data)
print(data)
print(type(data))
data.shape

# simple indexing

# define array
data = np.array([11, 22, 33, 44, 55])
# index data
print(data[0])
print(data[4])

#you can use negative indexes to retrieve values offset from the end of the array.
#For example, the index -1 refers to the last item in the array. The index -2 returns the second last item all the
# way back to -5 for the first item in the current example.
print(data[-1])
print(data[-5])

#Indexing two-dimensional data is similar to indexing one-dimensional data, except that a comma is used to separate
# the index for each dimension.
data = np.array([[11, 22], [33, 44], [55, 66]])
data[0,0]

#all items in the first row, we could leave the second dimension index empty, for example:
print(data[0,])

#%%Array Slicing
#Slicing is specified using the colon operator ‘:’ with a ‘from‘ and ‘to‘ index before and after the column
#respectively. The slice extends from the ‘from’ index and ends one item before the ‘to’ index.

# define array
data = np.array([11, 22, 33, 44, 55])

# Access all data
print(data[:])

#The first item of the array can be sliced by specifying a slice that starts at index 0 and ends at index 1 (one
#item before the ‘to’ index).
print(data[0:1])

#We can also use negative indexes in slices. For example, we can slice the last two items in the list by starting
#the slice at -2 (the second last item) and not specifying a ‘to’ index; that takes the slice to the end of the
#dimension.
print(data[-2:])
print(data[-2]) # Note the difference

#Two-Dimensional Slicing
#It is common to split your loaded data into input variables (X) and the output variable (y).
#We can do this by slicing all rows and all columns up to, but before the last column, then separately indexing the
#last column.

#For the input features, we can select all rows and all columns except the last one by specifying ‘:’ for in the
#rows index, and :-1 in the columns index.
data = np.array([[11, 22, 33], [44, 55, 66], 	[77, 88, 99]])

# separate data
X, y = data[:, :-1], data[:, -1]
print(X) # 2D array
print(y) # y is a 1D array.

#Split Train and Test Rows
split = 2
train,test = data[:split,:],data[split:,:]
print(train)
print(test)

#%% Array Reshaping
#For example, some libraries, such as scikit-learn, may require that a one-dimensional array of output variables
# (y) be shaped as a two-dimensional array with one column and outcomes for each column.
#Some algorithms, like the Long Short-Term Memory recurrent neural network in Keras, require input to be specified
# as a three-dimensional array comprised of samples, timesteps, and features.

# array of data
print('Rows: %d' % data.shape[0])
print('Cols: %d' % data.shape[1])

#Reshape 1D to 2D Array
#NumPy provides the reshape() function on the NumPy array object that can be used to reshape the data.
#The reshape() function takes a single argument that specifies the new shape of the array. In the case of reshaping
#a one-dimensional array into a two-dimensional array with one column, the tuple would be the shape of the array as
# the first dimension (data.shape[0]) and 1 for the second dimension.

data = np.array([11, 22, 33, 44, 55])
print(data.shape)
# reshape
data = data.reshape((data.shape[0], 1))
print(data.shape)

#Reshape 2D to 3D Array
data = np.array([[11, 22], [33, 44], [55, 66]])
print(data.shape)
# reshape
data = data.reshape((data.shape[0], data.shape[1], 1))
print(data.shape)

#%% Class work: Take the mean of neighbors
data = np.array([11, 22, 33, 44, 55])
#mean_neigh = ( data containing till 44 +  data containing from 22) / 2.0

#%% append, Vertical Stack, Horizontal Stack
data1 = np.array([11, 22])
data2 = np.array([33, 44])

np.append(data1,data2)
np.hstack((data1,data2))
np.vstack((data1,data2))

#%% Memory operation (Deep copy vs Shallow copy)

mat = np.array([[11,12,13],[21,22,23],[31,32,33]])
print("Original matrix")
print(mat)

# Simple Indexing
mat_slice = mat[:2,:2]
#mat_slice = np.array(mat[:2,:2]) # # Notice the np.array method

print ("\nSliced matrix")
print(mat_slice)
print ("\nChange the sliced matrix")
mat_slice[0,0] = 1000
print (mat_slice)
print("\nBut the original matrix? It got changed too!")
print(mat)

# Now uncomment mat_slice = np.array(mat[:2,:2]) and run again. Did you see the difference

#%% Nesting of IF-ELSE
data = np.array([11, 22, 33, 44, 55])
np.where(data > 22, 'greater than 22','less or equal to 22')
np.where(data > 22, True, False)
