import numpy as np
################ if #########################################################
"""
C style
if (condition){
  do some instructions if condition satisfy
}
"""

"""
Python - indendation based language
if (condition):
    do some instructions
next set of instructions
"""
a = 2
#    b = 3 throws error as there was unnecessary indendation

if (6 > 7):
    print("Condition satisfied") # will NOT be executed as condition failed
print("Something") # will be executed as it is outside if

if (6 < 7):
    print("Condition satisfied")


## Nested if
if (6 > 7):
    print("Condition1 satisfied")
    if (6 < 7):
        print("Condition2 satisfied")

### if else
        # if and else will be at the same indendation level
if (6 > 7):
    print("Condition satisfied")
else:
    print("Condition not satisified")

### else if (elif)
x = -10
if (x < 0):
    print("Negative number")
elif (x > 100):
    print("Large number")
elif (x > 0) and (x < 100):
    print("Valid range")
else:
    print("invalid")

#####################3 for loop ###################################
"""
C style
for (int i = 0; i < 10; i++){
  do repetitive operation with the iterating variable i
}
"""

"""
Python
for i in list/array/series/dictionary:
    do repetitive operation
"""
l1 = [5,6,3,2,9,8]
for i in l1: # looping through values in a list
    print("Square of ", i, "is ", i**2)

# for (int i = 0; i < len(l1); i++)
for i in range(0,len(l1)): # looping through sequence of numbers 0,1,2,3,4
    print("Square of ", l1[i], "is ", l1[i]**2)

for i in enumerate(l1): # loop through a (pos,value) tuple
    print(i)

#l1_sq = [25,36,9,4,81,64] # expected result list
l1_sq = []
for i in l1:
    l1_sq.append(i**2)
print(l1_sq)

# above approach of appending in a loop involves dynamic memory allocation
  # which is computationaly intensive
  # preallocating memory is generally faster to do wherever possible
l1_sq = [0]*len(l1) # dummy list
pos = 0 # maintaining another iterating variable for assigning
for i in l1:  # looping through the values   
    l1_sq[pos] = i**2
    pos = pos + 1
print(l1_sq)

l1_sq = [0]*len(l1) # dummy list
for i in range(len(l1)): # looping through positions using sequence
    l1_sq[i] = l1[i]**2
print(l1_sq)

#### LIST COMPREHENSION: Compact for loops
#[operation(i) for i in values if(condition)]
l1_sq = [i**2 for i in l1]
print(l1_sq)

### NUMPY VECTORIZED
l1_np = np.array(l1)
l1_sq = l1_np**2
print(l1_sq)

#### extract odd numbers from l1
#l1_odd = [5,3,9] # expected result
l1_odd = []
for i in l1: # looping through values
    if (i % 2 == 1):
        l1_odd.append(i)
print(l1_odd)

l1_odd=[]
for i in range(0,len(l1)): # looping through position
    if(l1[i] % 2 == 1):
        l1_odd.append(l1[i])
print(l1_odd)

## LIST COMPREHENSION
l1_odd = [i for i in l1 if (i % 2 == 1)]
print(l1_odd)

## NUMPY VECTORIZED
l1_odd = l1_np[l1_np % 2 == 1]
print(l1_odd)

#############################
math_score_list = [95,67,88,45,84]
#math_abv_70 = [95,88,84] expected result

# for loop
math_abv_70 = []
for i in math_score_list:
    if (i > 70):
       math_abv_70.append(i)
print(math_abv_70)

# list comprehension
math_abv_70 = [i for i in math_score_list if (i > 70)]
print(math_abv_70)

# numpy vectorized
math_score_array = np.array(math_score_list)
math_abv_70 = math_score_array[math_score_array > 70]

# looping through characters in a string
for i in "hello":
    print(i)

### Looping through dictionary
math_score_dict = {"Ram": 95, 
                   "Raj": 67, 
                   "Ravi": 88, 
                   "Roshini": 45, 
                   "Ranjith":84}
for i in math_score_dict:
    print(i)
    print(math_score_dict[i])
    
###### Nested for loop
list_of_list = [[1,2,3],[4,5,6],[10,11,12]]
for i in range(len(list_of_list)):
    li = list_of_list[i]
    for j in range(len(li)):
        print(list_of_list[i][j]**2)

#### Assignment on For loop (sent via email)
        
#1. Create a list of birth years of 5 friends/family member 
br_yr = [1986, 1989, 1975, 1981, 1978]
# Calculate their age (years alone) as of Dec 31, 2017 using 3 approaches and save it a list.
# Regular for loops

## Option 1
age = []
for i in br_yr:
    age.append(2017 - i)
print(age)

# Option 2
age = [0]*len(br_yr)
for i in range(len(br_yr)):
    age[i] = 2017 - br_yr[i]
print(age)
    
# List comprehension
age = [(2017 - i) for i in br_yr]
print(age)

# Vectorized operation using numpy array
br_yr_np = np.array(br_yr)
age = 2017 - br_yr_np
print(age)

#2. Create a string “this is a python exercise which is neither too easy nor too hard to be solved in the given amount of time”. 
 # Split the string to list of individual words 
 # [Hint: split command. Don’t search in classwork]. 
 # Remove words like ‘is’, ‘a’ and ‘the’ programmatically using 3 approaches
some_str = "this is a python exercise which is neither too easy nor too hard to be solved in the given amount of time"
str_list = some_str.split()

# Regular for loops

## Option 1
str_stopwords_removed = []
for i in str_list:
    if (i != "is") and (i != "the") and (i != "a"):
        str_stopwords_removed.append(i)

## Option 2: Nested for loop
stop_words = ["is","a","the"]
str_stopwords_removed = []
for i in str_list: # looping through the words
    is_stopword = 0 # defaulting that it is not a stop word
    for j in stop_words: # looping through the stop words and comparing
        if (i == j):
            is_stopword = 1
            break
    if (is_stopword == 0):
        str_stopwords_removed.append(i)    

## Option 3
str_stopwords_removed = []
for i in str_list:
    if (i not in stop_words):
        str_stopwords_removed.append(i)

# List comprehension
# Option 1
str_stopwords_removed = [i for i in str_list if (i != "is") and (i != "the") and (i != "a")]

# Option 2
str_stopwords_removed = [i for i in str_list if i not in stop_words]

# Vectorized operation using numpy array
str_np = np.array(str_list)
# Option 1: Multiple Conditions
str_stopwords_removed = str_np[(str_np != "is") & 
                                 (str_np != "the") & 
                                 (str_np != "a")]

## Option 2: Vectorized "IN"

"water" == "forest"
"water" != "forest"
"water" in ["water","forest","ocean"]
"animals" in ["water","forest","ocean"]
"animals" not in ["water","forest","ocean"]
np.in1d(["water","animals"],["water","ocean","forest"]) # vectorized in
~np.in1d(["water","animals"],["water","ocean","forest"]) # vectorized not in

arr1 = np.array([1,2,3,4,5])
arr2 = np.array([3,5,7])
#arr1[np.isin(arr1,arr2)] # values of arr1 which are present in arr2
arr1[np.in1d(arr1,arr2)] # alternative implementation of isin
#arr1[~np.isin(arr1,arr2)] # values of arr1 which are NOT present in arr2

stop_words = ["is","a","the"]
str_stopwords_removed = str_np[~np.in1d(str_np,stop_words)]



###################### while loop #############################################
"""
C style
while (condition){
  repeat operations till condition fails
}
"""

x = 4
while (x < 20):
    print(x**2)
    x = x + 1

"""
Following code will result in an infinite loop
x = 4
while (x < 20):
    print(x**2)
    x = x - 1
"""

################# break ######################################33
# used for terminating loops 

### Optimization algorithm do not have a pre-defined end
x = 4
max_iter = 100
iter_count = 0
while (x < 20):
    print(x**2)      
    #### An optimization logic which could potentially not end
    x = x - 1
    #####    
    ### Breaking condition
    if (iter_count > max_iter):
        print("Maximum iteration reached. Breaking")
        break
        print("Breaking complete") # will NOT be executed
    iter_count = iter_count + 1

### detecting whether a number is prime or not
# i = 2 till n-1
    
n = 1101
is_prime = 1 # default value that it is a prime number
for i in range(2, n): 
    if (n % i) == 0: 
       is_prime = 0 # flipping prime flag to 0
       break
if (is_prime == 0):
    print(n, "is not a prime number. Divisible by ", i) 
elif (is_prime == 1):
    print(n, "is a prime number")

######## for else ###################################
n = 18
if n > 1:
    for i in range(2,n):
        if (n % i == 0):
            print ("Not Prime")
            break
    else:
        print("Prime")

############## Continue ####################################
### Skipping iterations

for i in range(1,21):   
    if (i > 4 and i < 15):
        continue
    print("Square of", i, "is", i**2)





