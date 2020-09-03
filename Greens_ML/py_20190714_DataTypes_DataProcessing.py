## ()
  # calling functions
  # Creating tuples
## []
  # Slicing/Extracting
  # creating list
## {}
  # Creating dictionary in {Key:Value} format
  # Creating set inside {}


# int, float, str, bool, complex are the atomic data types in Python

############## Integer ##########################################
# assign any number without decimal point is treated as integer
a = 2
type(a)
b = -10
c = 12352385454644634635724243242535525352534553 # no short int or long int
#g = 01 # not valid integr

################## Float #######################################
d = 13.45677
type(d)

# integer to float and vice versa
d2 = int(d) # convert to the nearest small integer
d3 = round(15.6)
d4 = round(15,4)

a2 = float(a)

############### String ##################################
# anything within quotes is a string
s1 = "nature"
type(s1)
s2 = 'water' # even single quotes can be used
s3 = "999" # even number within quotes is a string
s33 = int(s3)
#s11 = int(s1) # throws error
s4 = "01"
s44 = int(s4) # works!
s10 = "obama"
s11 = "osama"
s12 = s10 + s11 # +  of 2 strings is concatenation
len(s10) # retuns the number of characters in a string
s6 = "a" # there is no character data type

######### Boolean ##########################################
# Either True or False
f = True
type(f)
g = False
#gg = false # throws error as python is case sensitive
h = 5 > 6
m = 6 == 6 # == for equality check
n = "hello" == "hi"
"hello" != "hi" # inequality check
h and m
h or m
not(h)

###################### Complex #############################
o = 5 + 10j
type(o)
o.real
o.imag
p = 5 - 10j
type(p)
p.real
p.imag
o * p # (a + bi) * (a - bi) = a^2 + abi - abi -bi^2 = a^2 + b^2

################ tuple #####################################
t1 = (1,5,4,2,9,8)
type(t1)
t2 = 1,5,4,3,2,6,7 # this also a tuple but not a common approach
len(t1)
t2 = ("hello","nature","forest","water")
t3 = (1,5.6,"water",False) # mix of data types is possible

t1[0]
t1[3]
t1[len(t1) - 1] # last element
t2[len(t2) - 1]
t1[-1] # last element
t2[-1]
t1[1:4] # slicing range of positions; last position excluded
t1[0:4] # from 0th till 3rd position
t1[:4] # same as above
t1[2:6] # 2nd till last position
t1[2:] # same as above
#t1[10] # index out of range error
(t1[2],t1[5]) # slicing disjoint positions

## tuples are immutable
# once created, values cannot be edited
#t1[2] = 100

################ LIST ################################
l1 = [1,5,4,2,9,8]
type(l1)
l1[2] = 100 # List is mutable
l3 = list(t3) # list can also have mix of data types

l1.append(99) # appending
l1.insert(2,88)
l2 = [5,7,3,4,65,7,7,8,4,4,4,4,4,4]
#l1.append(l2) # this will append l2 as a list and not the elements
l1.extend(l2) # extend l1 with values of l2
l1 = l1 + l2 # + will also concatenate but has to be explicitly assigned

## LIST SORTING
l1.sort()
l1.sort(reverse = True) # sort in descending order
l3.reverse() # reverse the order of values

# Get positon corresponding to a value
l2[3]
l2.index(65) # position corresponding to a value
l2.index(7) # only the first occurence position is returned
del l2[3] # deleting value in a position
l2.remove(65)  # deleting a value
l2.remove(7) # only first occurence is deleted
l2.count(7)
l2.count(4)

## LIST SEARCHING
100 in l2
5 in l2
100 not in l2

## LIST SLICING
l1[0]
l1[-1]
l1[2:10] # slicing range of positions with step size
l1[2:10:2] # slicing range of positions with step size

## LIST OF LIST
l10 = [1,2,3]
l11 = [5,6]
ll10 = [l10,l11]
ll10[0][1]

## RANGE: generate sequence of numbers
lseq1 = list(range(1,10)) # 1, 2,.... 9
lse2 = list(range(1,100,2)) # 1,3,5,7,....,99
lseq3 = list(range(10)) # 0,1,2,.....,9
rr = range(10)
type(rr)

## REPEAT VALUES
lrep1 = [5]*10
lrep2 = [1,2,3]*20
lrep3 = ["hello"]*5

### ADDITIONAL REFERENCES
### LIST ASSIGNMENT
#Create following lists

#a) [1,2,3,….,19,20]
a = list(range(1,21))

#b) [20,19,…,2,1]
b = list(range(20,0,-1)) # option 1

# Option 2
b = a.copy() # making a copy of a
b.reverse()

#c) [1,2,3,….19,20,19,18,….,2,1]
c = a + b[1:] # option 1
c = a[:-1] + b # option 2
c = list(range(1,21)) + list(range(19,0,-1)) # option 3

#d) [4,6,3] and assign it to variable tmp
tmp = [4,6,3]

#e) [4,6,3,4,6,3,…..,4,6,3] where there are 10 occurences of [4,6,3]
e = tmp*10

#f) [4,6,3,4,6,3,….,4,6,3,4] where there 10 occurences of [4,6,3] followed by 4

f = e + [4] # Option 1
f = e + [tmp[0]]  # Option 1A

# Option 2
f = e.copy()
f.append(4)

#g) [4,4,….,4,6,6,….,6,3,3,….,3] where there are 10 occurences of 4, 
    # 20 occurences of 6 and 30 occurences of 3
g = [4]*10 + [6]*20 + [3]*30 # Option 1
g = tmp[0]*10 + tmp[1]*20 + tmp[2]*30 # Option 2

#Slice the following from list “f”
# 0th element
f[0]
# last but 3 till last element
f[-4:]

# downsample by 2 (skip alternative samples)
f_downs = f[::2]


############ DICTIONARY ################################
# {Key:Value} pair
math_score_list = [95,67,88,45,84]
math_score_list[3]

math_score_dict = {"Ram": 95, 
                   "Raj": 67, 
                   "Ravi": 88, 
                   "Roshini": 45, 
                   "Ranjith":84}
# have a look at the order in Variable explorer
# Till Python 3.5 dictionary do not follow the order in which it is created

# Keys have to be unique
# Duplicates will override original
math_score_dict2 = {"Ram": 95, 
                   "Raj": 67, 
                   "Ravi": 88, 
                   "Roshini": 45, 
                   "Ram": 79,
                   "Ranjith":84}

type(math_score_dict)
math_score_dict["Ravi"]
math_score_dict["Ranjith"]
math_score_dict["Rohit"] = 87 # include a key value pair in dictionary
del math_score_dict["Rohit"] # delete a key value pair
#math_score_dict[0] # dictionary can only be sliced using Key
math_score_dict.keys()
math_score_dict.values()

# Q: How to extract keys corresponding to a value? 
  # Not directly supported by dictionary. Need to do a work around using list
idx_corr = list(math_score_dict.values()).index(84)
list(math_score_dict.keys())[idx_corr]

states = {
    'Oregon':'OR',
    'Florida':'FL',
    'California':'CA',
    'New York': 'NY',
    'Michigan':'MI'}

cities = {
    'CA':'San Fransico',
    'MI': 'Detroit',
    'NY':'Manhattan'}
# what is the name of the city in state "MI"
cities["MI"]
# Add a city 'Orlando' to 'FL'
cities["FL"] = "Orlando"
# what is the city name in 'New York' State
cities[states["New York"]]

## Key can be of any immutable data type
## Value can be of any data type

some_dict = {"A": 5.6,
             3: "Hello",
             ("Hi",6): [6,7,8,9],
             5: {"Ram": 56, "Raj": 65}}
some_dict[("Hi",6)]
# below lines will throw error as key cannot be a list
#some_dict2 = {"A": 5.6,
#             3: "Hello",
#             ["Hi",6]: [6,7,8,9],
#             5: {"Ram": 56, "Raj": 65}}

emp_details = {"R101": {"Emp Name": "Karthik","Dept": "ABC"},
               "R102":  {"Emp Name": "Ramu", "Dept": "CBA"}}


################### set #######################################
# useful for set operations
s1 = {1,24,5,6,2,4,5,4,3,5,6,4,3,2,3,4,5,6,3}
type(s1)
print(s1) # only unique values are saved

setA = {1,2,3,4,5}
setB = {4,5,6,7}
setA & setB # intersection
setA | setB # union
setA - setB # values in A which are not present in B
setA ^ setB # values exclusive in A and B

########################## date and time ########################
import datetime

###### date
dtod = datetime.date.today()
type(dtod) # datetime.date
dtod.day
dtod.month
dtod.weekday() # Monday starts with 0

####### datetime
tnow = datetime.datetime.now()
print(tnow)
type(tnow) # datetime.datetime
tnow.day
tnow.minute

## Standard date format in Python yyyy/mm/dd format

##### STRPTIME: Convert custom format to a standard format
dt1 = datetime.datetime.strptime("15/8/2018","%d/%m/%Y")
dt2 = datetime.datetime.strptime("15-8-2018","%d-%m-%Y")
dt1 = datetime.datetime.strptime("August 15, 2018","%B %d, %Y")
dt1.weekday()

######### STRFTIME: Convert standard format to a custom format
datetime.datetime.strftime(dtod, "%d/%m/%Y")
datetime.datetime.strftime(dtod, "%B %d, %Y")

### Q: How to get current time of a different timezone?
import pytz
py_time_zone_list = pytz.all_timezones
tz = pytz.timezone("Europe/London")
datetime.datetime.now(tz)

############### mathematical calculations ####################################

math_score_list = [95,67,88,45,84]
sum(math_score_list)/len(math_score_list) # average

# What is the median?
math_score_list2 = math_score_list.copy()
math_score_list2.sort()
math_score_list2[int(len(math_score_list2)/2)]

math_score_list3 = [95,67,88,45,84,58]
math_score_list4 = math_score_list3.copy()
math_score_list4.sort() # find the 2 middle points and calculayte average of them

################## numpy #######################
##### numpy comes pre-installed in anaconda

## numpy has a lot of mathematical functions
import numpy as np
np.mean(math_score_list)
np.median(math_score_list)
np.std(math_score_list)

############ numpy array ################################
# all values should be of same data type
math_score_array = np.array(math_score_list)
type(math_score_array)
l100 = [1,5.6,"hello", True] # list can have mix of data types
arr100 = np.array(l100) # all values converted to astring
arr100[1] + 10
arr111 = np.array([1,5.6,True,False])

## numpy number generators
arr_odd = np.arange(1,100,2)
type(arr_odd)
## THE BELOW RANDOM.RAND(100) WILL PRODUCE 100 RANDOM VALUES
arr2 = np.random.rand(100)
np.mean(arr2)
np.median(arr2)
## Median is robust to outliers when compared with mean
arr22 = np.append(arr2,10000000000)
np.mean(arr22)
np.median(arr22)

# fixing the seed results in same random logic done across running instances and across computer too
np.random.seed(543) # fixing the randomness helps reproducing results
## THE BELOW RANDOM.RAND(100) WILL PRODUCE 10 RANDOM VALUES
np.random.rand(10)
##THE BELOW WILL PRODUCE 20 NUMBERS STARTING FROM 1 TILL 100.
np.random.randint(1,100,20)
# generate income of 1000 employees following normal distribution 
 # average income being 45000 and standard deviation being 5000
rand_income = np.random.normal(45000,5000,1000)
np.mean(rand_income)

np.random.seed(67775)
np.random.normal(45000,5000,10)

#### ARRAY SLICING
## SLICING LIKE  A LIST
math_score_array[0]
math_score_array[2]
math_score_array[-1]
math_score_array[1:4]

#### VECTORIZED OPERATION
l10 = [2,5,7]
l11 = [7,6,2]
#l12 = [9,11,9] # element wise sum needs a for loop in case of list
l10 + l11 # concatenation
 
arr10 = np.array(l10)
arr11 = np.array(l11)
arr10 + arr11 # element wise sum

arr12 = np.array([1,2,3])
arr13 = np.array([7,8])
#arr12 + arr13 # throws error as the lengths are not equal
math_score_updated = math_score_array + 2
l10[0] > 5
l10[1] > 5
l10[2] > 5
#l10 > 5  # throws error
arr10 > 5 # possible in a numpy array

## BOOLEAN SLICING
arr20 = np.array([True,False,False,True,True])
arr21 = np.array([5,6,7,4,3])
arr21[arr20]
math_score_array[math_score_array > 70]

## Practice
math_score_array = np.array([95,67,88,45,84])
eng_score_array = np.array([78,67,45,39,67])
gender_array = np.array(["M","F","F","M","M"])

# extract maths score above average maths score
avg_maths = np.mean(math_score_array)
cond = math_score_array > avg_maths
math_abv_avg = math_score_array[cond]

# all in 1 line
math_abv_avg = math_score_array[math_score_array > np.mean(math_score_array)]

# extract maths score of male students
math_male_bool = gender_array=='M'
male_math_score = math_score_array[math_male_bool]

# average maths score of male students
np.mean(male_math_score)
np.mean(math_score_array[gender_array=='M']) # in 1 line

# Extract maths score of male students who have scored above 70

# Option 1
male_math_score[male_math_score > 70] 

# Option 2
cond1 = gender_array == "M"
cond2 = math_score_array > 70
math_score_array[cond1 & cond2]

###Bit wise Logical Operators
arr1 = np.array([True,False,True])
arr2 = np.array([True,True,False])
True and False
True or False
#arr1 and arr2 # and can be used only to compare scalars not vectors(arrays)
arr1 & arr2 # bit-wise (element wise) and
arr1 | arr2 # bit-wise OR operator'
~arr1 # vectorized NOT

# Extract maths score of students who are above average in english?
np.mean(math_score_array[eng_score_array > np.mean(eng_score_array)])

# Average english score of male students who are above average in maths
cond1 = gender_array == "M"
cond2 = math_score_array > np.mean(math_score_array)
np.mean(eng_score_array[cond1 & cond2])

##### Assignment sent through email
#Create two numpy arrays with following values
xVec = np.array([42,85,84,23,11,55,14,96,13,30])
yVec = np.array([13,8,85,71, 1,7,55, 2,34,24])
len(xVec)
len(yVec)

#a. Subset xVec with values greater than 60
xVec[xVec > 60]

#b. Subset yVec with values less than mean of yVec
mean_yvec = np.mean(yVec)
cond = yVec < mean_yvec
yVec[cond]
yVec[yVec < np.mean(yVec)] # above 3 lines in 1 line

#c. How many odd numbers in xVec?
len(xVec[xVec % 2 == 1]) # Option 1: SLicing and calculation length
sum(xVec % 2 == 1) # Option 2: sum of a boolean array gives count of Trues

#d. Subset values in yVec which are between minimum and maximum values of xVec (yes, xVec)
yVec[(yVec > min(xVec)) & (yVec < max(xVec))]


# Q: Get the index of 2nd occurence of 8
a222 = np.array([6,7,3,4,8,8])
np.where(a222 == 8)[0][1]

############ numpy matrix #########################################
arr10 = np.random.randint(1,100,12)
type(arr10) # numpy n dimensional array

## Creating matrix by reshaping an array
mat10 = np.reshape(arr10, [3,4])
mat11 = np.reshape(arr10, [4,3])
mat12 = np.reshape(arr10, [2,3,2])
#mat13 = np.reshape(arr10, [3,3])# throws error as the lengths mismatch
mat01 = mat10.T # Transpose

## Creating matrix by stacking arrays
math_score_array = np.array([95,67,88,45,84])
eng_score_array = np.array([78,67,45,39,67])

score_mat = np.column_stack([math_score_array,eng_score_array])
score_mat2 = np.row_stack([math_score_array,eng_score_array])

# Converting multi dimensional array to 1 dimensional
np.reshape(score_mat,10)

# Q: How to iniate a matrix of required shape?
np.zeros([3,4])

### MATRIX SLICING
# matrix[row_pos,col_pos]
score_mat[0,0] # 0th row, 0th col
score_mat[2,1] # 2nd row, 1st col
score_mat[:,0] # all rows, 1st column
score_mat[1:4,:] # 1st till 3rd row
score_mat[[1,3],:] # 1st row, 3rd row

# extracting data of students who scored greater than 70 in maths
score_mat[score_mat[:,0] > 70,:]

################# pandas #########################################
import pandas as pd

######### pandas Series ########################################
# has the advantages of both numpy array and dictionary
math_score_array = np.array([95,67,88,45,84])
math_score_array[1]
math_score_array[3]

math_score_series = pd.Series([95,67,88,45,84])
type(math_score_series)
math_score_series = pd.Series([95,67,88,45,84],
                index = ["R101","R104","R110","R210","R150"])
math_score_series[1] # slicing using position
math_score_series[1:4]
math_score_series["R104"] # slicing using index
math_score_series["R104":"R210"]

eng_score_array = np.array([78,67,45,39,67])
eng_score_series = pd.Series(eng_score_array,
                    index = ["R101","R104","R110","R210","R150"])
math_score_series + eng_score_series #element wise operation
eng_score_series[math_score_series > 70] # boolean slicing

# Mathematical functions
math_score_series.mean() # pandas mean function
np.mean(math_score_series) # numpy function can be applied on a series
math_score_series.median()

# Creating series from dictionary
cities = {
    'CA':'San Fransico',
    'MI': 'Detroit',
    'NY':'Manhattan'}
cities_series = pd.Series(cities)

## element wise operations between 2 series happens based on index
s1 = pd.Series([1,2,3], index = ["R1","R2","R3"])
s2 = pd.Series([4,5,6], index = ["R1","R2","R4"])
s3 = pd.Series([7,8], index = ["R1","R2"])
s1 + s2
s1 + s3

#### Series with integer index
## Be watchful as index will override position
math_score_series4 = pd.Series([95,67,88,45,84],
                    index = range(1001,1006))
#math_score_series4[0] # throws error as index is an integer
math_score_series4[1003]
math_score_series4[1:4] # slicing range of position works
math_score_series4[0:1] # gives 0th position; more of a hack

################ pandas Dataframe #####################################
# similar to an SQL table or an Excel table

### Creating dataframe from dictionary of list
# every key value pair becomes a column where key is column header
df = pd.DataFrame({
        "A": [1,2,3],
        "B": [4,5,6]})
# below throws error as the lenghts are not equal
#df2 = pd.DataFrame({
#        "A": [1,2,3],
#        "B": [4,5]})

########## Creating data frame from dictionary of series
df3 = pd.DataFrame({
        "A": pd.Series([1,2,3], index = ["m","n","o"]),
        "B": pd.Series([4,5,6], index = ["m","n","o"])})

# matching happens using index. mismatches filled with nulls
df4 = pd.DataFrame({
        "A": pd.Series([1,2,3], index = ["m","n","o"]),
        "B": pd.Series([4,5,6], index = ["n","o","p"])})

"""
create a data frame df_emp_details with data of 3 employees
with unique names "Ram","Raj","Ravi" as index.
Create 2 columns Age and Income and assign any integer
"""

# Option 1
df_emp_details = pd.DataFrame({
        "Age": pd.Series([25,32,45], index = ["Ram","Raj","Ravi"]),
        "Income": pd.Series([27000,29000,39000], index = ["Ram","Raj","Ravi"])})

# Option 2: Giving index in common
df_emp_details = pd.DataFrame({
        "Age": [25,32,45],
        "Income": [27000,29000,39000]},
        index = ["Ram","Raj","Ravi"])

# Option 3: From numpy matrix
emp_mat = np.column_stack([[25,32,45],[27000,29000,39000]])
df_emp_details = pd.DataFrame(emp_mat,
                              index = ["Ram","Raj","Ravi"],
                              columns = ["Age","Income"])

#### DATAFRAME PROPERTIES
df_emp_details.shape # returns the number of rows and columns as tuple
df_emp_details.shape[0] # number of rows
df_emp_details.shape[1] # number of columns
df_emp_details.index
df_emp_details.columns
df_emp_details.dtypes
df_emp_details.head(2) # first 2 rows
df_emp_details.tail(2) # last 2 rows

# replacing all the column names
df_emp_details.columns = ["Age1","Income1"]

## replace specific columns using rename function
#df_emp_details.columns = "Age2" throws error as the lengths mismatch
df_emp_details = df_emp_details.rename(columns = {"Age1": "Age2"})


##### DATAFRAME SLICING

## Slicing columns
df_age = df_emp_details["Age2"] # slicing one column
type(df_age) # series
cols_needed = ["Age2","Income1"]
df_age_inc = df_emp_details[cols_needed] # slicing list of columns
type(df_age_inc) # dataframe

#### .LOC ; slicing using index
df_emp_details.loc["Raj",:]
df_emp_details.loc[["Ram","Ravi"],:]
df_emp_details.loc[:,"Income1"] # same as df_emp_details["Income1"]

###### .ILOC; slicing using position
df_emp_details.iloc[1,:]
df_emp_details.iloc[1,0]
df_emp_details.iloc[:,1]
df_emp_details.iloc[1:3,:]

### CONDITIONAL SLICING (.LOC)
## data of employees with income greater than 28000
# SQL: Select * from df_emp_details where income > 28000
cond = df_emp_details["Income1"] > 28000
df_emp_details[cond] # all columns can be obtained this way
df_emp_details.loc[cond,:] # same as above

## age of employees with income greater than 28000
#SQL: Select age from df_emp_details where income > 28000
df_emp_details.loc[cond,"Age2"]

### DATAFRAME SORTING
df_emp_details.sort_values(by = "Age2")
df_emp2 = df_emp_details.sort_values(by = ["Age2","Income1"], 
                                     ascending = False)

############## Assignment 
math_score_array = np.array([95,67,88,45,84])
eng_score_array = np.array([78,67,45,39,67])
gender_array = np.array(["M","M","F","M","F"])

# Create a data frame (score_df) with above 3 arrays as columns
# Add "R1001","R1002",...."R1005" as row indexes
# Add "Maths","English","Gender" as column indexes

## Option 1: From dictionary of arrays
score_df = pd.DataFrame({
        "Maths": math_score_array,
        "English": eng_score_array,
        "Gender": gender_array},
            index = ["R1001","R1002","R1003","R1004","R1005"])
score_df = score_df[["Maths","English","Gender"]] # rearranging the columns
score_df.dtypes # string will be saved as object data type in dataframe

## Option 2: From numpy matrix
score_mat = np.column_stack([math_score_array,eng_score_array,gender_array]) # string matrix
score_df = pd.DataFrame(score_mat,
                        index = ["R1001","R1002","R1003","R1004","R1005"],
                        columns = ["Maths","English","Gender"])
score_df.dtypes # all columns are object
score_df["Maths"] = score_df["Maths"].astype(int)
score_df["English"] = score_df["English"].astype(int)
score_df.dtypes

## DATAFRAME SLICING

# Slice the following
# Maths column
score_df["Maths"] # option 1
score_df.loc[:,"Maths"] # option 2
type(score_df["Maths"]) # series

# Maths and English Column
cols_needed = ["Maths","English"]
score_df[cols_needed]
score_df.loc[:,["Maths","English"]]
#score_df["Maths","English"] throws error as columns needed is not passed as list

# "Maths" column of "R1001"
score_df.loc["R1001","Maths"]

# "Maths" and English column values of "R1001" and "R1003"
score_df.loc[["R1001","R1003"],["Maths","English"]]

# All rows, 2nd column
score_df.iloc[:,2]

# 0th and 3rd row, 0th and 1st column
score_df.iloc[[0,3],[0,1]]

# data frame of Male students alone
# SQL: Select * from score_df where Gender = "M"
cond = score_df["Gender"] == "M"
score_df[cond]
score_df.loc[cond,:] # same as above

# english and maths score of Male students
# SQL: Select Engish, Maths from score_df where Gender = "M"
score_df.loc[cond,["English","Maths"]]

# all columns of students who score above 70 in Maths
# Select * from score_df where Maths > 70
cond = score_df["Maths"] > 70
score_df[cond]

# average maths core of students who got above 60 in English
score_df.loc[score_df["English"] > 60,"Maths"].mean()

# average english score of students who are above average in maths
# SQL: Select average(engligh) from score_df where maths > average(maths)
mean_math = score_df["Maths"].mean()
cond = score_df["Maths"] > mean_math
score_df.loc[cond, "English"].mean()
# last 3 lines in 1 line
score_df.loc[score_df["Maths"] > score_df["Maths"].mean(),"English"].mean()

# all columns of male students who scores above 60 in maths
cond1 = score_df["Maths"] > 60
cond2 = score_df["Gender"] == "M"
score_df[cond1 & cond2]

"""
slice english and gender column of either 
female students or students with maths score above 60
"""
cond1 = score_df["Maths"] > 60
cond2 = score_df["Gender"] == "F"
score_df.loc[cond1 | cond2,["English","Gender"]]

###### Pandas additional references

###### Pandas assignment (sent over email)

# Option 1: Dataframe from dictionary of list/arrays
df = pd.DataFrame({
        "A": np.random.randint(10,101,25),
        "B": [5]*25,
        "C": range(25,0,-1),
        "D": range(0,50,2)},
    index = range(1001,1026))

# Option 2: Dataframe from a numpy matrix
df = pd.DataFrame(np.column_stack([np.random.randint(10,101,25),
                 [5]*25,
                 range(25,0,-1),
                 range(0,50,2)]),
            index = range(1001,1026),
            columns = ["A","B","C","D"])
df.dtypes
                
#1. Slice column ‘A’ from df and save it as a series ‘s’
s = df["A"]
type(s)

#2. Slice column ‘A’ and column ‘C’ and save it as df2
df2 = df[["A","C"]]
type(df2)

#3. Slice 0th and 2nd column using column number and save it as df3
df3 = df.iloc[:,[0,2]]

#4. Slice from 0 till 5th position in series ‘s’
s[0:6] # option 1
s[:6] # option 2
s.iloc[0:6] # option 3

#5. Slice all columns from rows 3 till 19 and save it as df4
df4 = df.iloc[3:20,:]

#6. Create df5 which has subset of data from df 
 # where column A values are above median of column A. 
 # Note: slice entire columns based on condition on column A
# SQL: Select * from df where A > median(A)
cond = df["A"] > df["A"].median()
df5 = df[cond] # option 1
df5 = df[df["A"] > df["A"].median()] # option 2
df5 = df.loc[df["A"] > df["A"].median(),:] # option 3
















