import warnings
import numpy as np
import pandas as pd
## There are a lot of inbuilt functions
print("Hello","World")
sum([1,2,3,5])

## Some functions are part of packages
np.median([3,7,6,5])

############# user defined functions ######################
"""
def fn_name(input arguments):
    some calculation
    return(ouput)
"""

# Function definition should be done before calling it
def print_with_exclamation(ip_word):
    print(ip_word + "!!!!!")
    
print_with_exclamation("Hello") # calling a function

def calc_sum(x,y):
    z = x + y
    return(z)
    
op1 = calc_sum(5,10)
op2 = calc_sum(60,70)
#calc_sum(5)  throws error as function expect 2 mandatory inputs
print("Hello","World") # sep is defaulted to " "
print("Hello","World", sep = "|")

## Function with default arguments
def calc_sum2(x, y = 0): # x is mandatory and y is optional
    z = x + y
    return(z)
calc_sum2(5)

"""
Below function throws error as argument with default values should 
come in the end. Mandatory inputs should come in the beginning
def calc_sum3(x = 0, y):
    z = x + y
    return(z)
"""

def calc_sum3(x = 100, y = 200):
    z = x + y
    return(z)
calc_sum3(5,10)
calc_sum3(5)
calc_sum3()

### Create a function as follows
def calculate(x = 0, y = 0, option = "sum"):
    if (option == "sum"):
        z = x + y
    elif (option == "diff"):
        z = x - y
    elif (option == "mult"):
        z = x*y
    elif (option == "div"):
        z = x/y
    else:
#        print("Incorrect option. Returning None")
        warnings.warn("Incorrect option. Returning None")
        z = None
    return(z)

##### Positional matching
op1 = calculate(5,3,"sum") # 8
op2 = calculate(5,3,"diff") # 2
op4 = calculate(5,3,"mult") # 15
op6 = calculate(6,3,"div") #2
op7 = calculate(6,7,"junk")

##### Argument Matching
calculate(y = 5, x = 10, option = "div")
calculate(option = "diff", y = 5, x = 10)

########## Scope of a function #################################
## Variables get created and dies within a function
def some_fn(v1):
    v2 = v1 + 5
    return(v2)

res1 = some_fn(10)
""" 
following lines throw error as scope of v1 and v2 is inside the function
print(v1)
print(v2)
"""

v2 = 50
re2 = some_fn(5)
print(v2) # v2 inside the function is different from v2 outside the function

########## Returning multiple values from function
## Multiple outputs can be wrapped as a list/tuple
def calc_sum_diff(x,y):
    s = x + y
    d = x - y
#    return(s,d)# this returning a tuple
    return([s,d])
res10 = calc_sum_diff(5,10)
print(res10)    
# the returned values can be captured in multiple variables too
res11, res12 = calc_sum_diff(5,10)
print(res11)
print(res12)
#res11, res12, res13 = calc_sum_diff(5,10) # throws error as only 2 outputs are returned whereas 3 are expected

### Write a function which returns MAPE
   # mean and median absolute percentage error

def MAPE(actual, predicted):      
    ### Absolute percent error calculation using numpy vectorized
    actual_np = np.array(actual)
    predicted_np = np.array(predicted)
    ape = abs(actual_np - predicted_np)*100/actual_np    
    """
    Absolute percent error calculation using for loop
    ape = [0.0]*len(actual)
    for i in range(len(actual)):
        ape[i] = abs(actual[i] - predicted[i])*100/actual[i]
        
    """    
    mean_ape = np.mean(ape)
    median_ape = np.median(ape)
    return([mean_ape,median_ape])
    
predicted_score_india = [300,280,290,310,250]
actual_score_india = [340,290,260,315,290]
mean_abs_pe, median_abs_pe = MAPE(actual_score_india,predicted_score_india)

predicted_seats_bjp_mp_up_raj = [20, 50, 30] 
actual_seats_bjp_mp_up_raj = [18, 60, 25] 
MAPE(actual_seats_bjp_mp_up_raj,predicted_seats_bjp_mp_up_raj)


#################### Lambda functions ########################################
## Compact functions
## Another style of defining functions
"""
fn_name = lambda ip_arguments: output_arguments
"""
def calc_sum(x,y):
    z = x + y
    return(z)
calc_sum(5,10)

calc_sum_l = lambda x,y: x + y
calc_sum_l(5,10)    

# lambda function with default input arguments
calc_sum_l2 = lambda x = 0, y = 0: x + y
calc_sum_l2(6,7)
calc_sum_l2(6)

# lambda function with more than 1 output
calc_sum_diff_l = lambda x,y: [x + y, x - y]
s,d = calc_sum_diff_l(5,10)

def mean_median_diff(a):
    r = np.mean(a) - np.nanmedian(a)
    return(r)
mean_median_diff([1,4,5,7,8,5,10])

mean_median_diff_l = lambda a: np.mean(a) - np.median(a)
mean_median_diff_l([1,4,5,7,8,5,10])

########### Functional Programming
## Passing function as an input to another function
def some_calc(a,f1):
    return(f1(a))
l1 = [1,5,6,8,3,4]
some_calc(l1,sum)
some_calc(l1,min)
some_calc(l1,max)
# even a user defined function (udf) can be passed as input to another udf
some_calc(l1,mean_median_diff) 

airquality = pd.read_csv("data/airquality.csv")

## Passing udf to apply, aggregate
airquality.mean()
airquality.apply(np.mean, axis = 0) # any function can be fed as apply
airquality.apply(mean_median_diff, axis = 0)
airquality.groupby("Month")["Temp"].agg(mean_median_diff)

############### Assignment ################################
"""
1.	Write a function which accepts a list/array/series 
as input and returns the 
difference between mean and median
"""

def mean_median_diff(a):
    b = np.mean(a) - np.median(a)
    return(b)

mean_median_diff2 = lambda a: np.mean(a) - np.median(a)

"""
2.	Write a function (max_var2_corresponding) which accepts 
a data frame (df) as input along with 2 column names 
(var1, var2) in the data frame. 
Calculate the maximum value in var1 column of df. 
Return the value of var2 corresponding to maximum value of var1
"""
def max_var2_corresponding(df,var1,var2):
    return(df.loc[df[var1].idxmax(),var2])
    
#a.	Test Case 1:
#Create a dataframe score_df using following set of commands
math_score_array = np.array([95,67,88,45,84])
eng_score_array = np.array([78,67,45,39,67])
gender_array = np.array(["M","M","F","M","F"])
score_df = pd.DataFrame({
        'Maths':math_score_array,
        'English':eng_score_array,
        'Gender':gender_array})
score_df.index = ["R1001","R1002","R1003","R1004","R1005"]
#Call the function developed by you with following statements. Expected outcome is provided after #
max_var2_corresponding(score_df,"Maths","English") #78
max_var2_corresponding(score_df,"English","Gender") #M

#b.	Test Case 2:
#Create a dataframe emp_details using following set of commands
emp_details_dict = {
    'Age': [25,32,28],
    'Income': [1000,1600,1400]
    }
emp_details = pd.DataFrame(emp_details_dict)
emp_details.index = ['Ram','Raj','Ravi']
#Call the function developed by you with following statements. Expected outcome is provided after #
max_var2_corresponding(emp_details,"Income","Age") #32

