import pandas as pd
import os

#Escape characters": \n, \t, 
print("hello")
print("hel\nlo") # new line
print("he\tllo") # tab
# providing r before string doesn't intepret escape characters
print(r"hel\nlo") #\n is treated as another character in string

acs2013 = pd.read_csv(r"C:\Karthik\Learning\Python\Green\data\ACS_13_5YR_S1903.csv")
acs2013 = pd.read_csv("C:\\Karthik\\Learning\\Python\\Green\\data\\ACS_13_5YR_S1903.csv")
acs2013 = pd.read_csv("C:/Karthik/Learning/Python/Green/data/ACS_13_5YR_S1903.csv")

acs2008 = pd.read_csv("C:/Karthik/Learning/Python/Green/data/ACS_08_3YR_S1903.csv")

## It is recommended to use relative path than absolute path
## Typically it is done by changind working directory to the project folder

os.getcwd()
os.chdir("C:\\Karthik\\Learning\\Python\\Green")
os.getcwd()

acs2013 = pd.read_csv("data/ACS_13_5YR_S1903.csv")
acs2008 = pd.read_csv("data/ACS_08_3YR_S1903.csv")

## Pandas assigns the data type for each column by itself
## It is a good practice to check the same after reading ecternal files
acs2013.dtypes
# GEO.id2 is FIPS code which is supposed to be a string of length 2
# However pandas has converted it to an integer which is incorrect
#a = 01 invalid integer
a = "01"

acs2013 = pd.read_csv("data/ACS_13_5YR_S1903.csv",
                      dtype = {"GEO.id2": str})

acs2008 = pd.read_csv("data/ACS_08_3YR_S1903.csv",
                      dtype = {"GEO.id2": str})

## Try following questions for ACS 2013 data
# Q1. slice the first 7 columns and save as acs_2013_s
acs_2013_s = acs2013.iloc[:,:7]

# Q2. rename the column names as follows
acs_2013_s.columns = ["ID","FIPS","State",
                    "Total Household", "Total Household MOE",
                    "Income","Income MOE"]

# Q3. calculate average income of US
acs_2013_s["Income"].mean()

# Q4. what is the maximum income and which state is that?

# SQL: Select State from acs_2013_s where Income = max(Income)
# Option 1
cond = acs_2013_s["Income"] == acs_2013_s["Income"].max()
acs_2013_s.loc[cond,"State"]

# Option 2
acs_2013_income_sorted = acs_2013_s.sort_values("Income", ascending = False)
acs_2013_income_sorted.head(1)["State"]

# Option 3
acs_2013_s["Income"].max()
acs_2013_s["Income"].idxmax() # index corresponding to maximum value
acs_2013_income_sorted.loc[acs_2013_s["Income"].idxmax(), "State"]


# Q5. what is the minimum income and which state is that?
# Option 1
acs_2013_s.loc[acs_2013_s["Income"] == acs_2013_s["Income"].min(),"State"]

# Option 2
acs_2013_income_sorted.loc[acs_2013_s["Income"].idxmin(), "State"]


# Q6. get the list of states which are above average in household income
# SQL: Select state from acs_2013_s where Income > average(Income)
cond = acs_2013_s["Income"] > acs_2013_s["Income"].mean()
acs_2013_s.loc[cond, "State"]


# Q7. get the income of texas state
# Select Income from acs_2013_s where State = Texas
acs_2013_s.loc[acs_2013_s["State"] == "Texas","Income"]

# Q8. what is the state which has the 2nd highest income
acs_2013_income_sorted.iloc[1,2] # Option 1
acs_2013_income_sorted.reset_index().loc[1,"State"] # Option 2

############### FILE MERGING ################################################
 # Similar to JOIN operation in SQL or LOOKUP operations in Excels
 # Merging happens on key(s)
# Employee personal details (Emp ID, Name, Phone, EMail ID, #no of dependents, address)
# Employee org details (Emp ID, Dept ID, Dept Name, Designation, Reporting manager EMp ID)
# Employee comp details (Emp ID, Basic, HRA, PF)
 
# Transaction table (Product ID, Store ID, Unit sales, Price per unit, Coupon)
# Product master (Product ID, Product name, Manufacturer name, Brand, Pack size)
# Store master (Store ID, Store address, Lat, Long)

# Merge data from 2008 and 2013 survey. 
 # Extract FIPS and Income column for both the data
 # Rename columns as "FIPS","Income

income_2008 = acs2008.iloc[:,[1,5]]
income_2008.columns = ["FIPS","Income"]
income_2013 = acs2013.iloc[:,[1,5]]
income_2013.columns = ["FIPS","Income"]

### If the name of reference column is same in both tables, use "On"
merged_income = pd.merge(income_2008,income_2013, on = "FIPS")
# If the left and right table has same columns, they will be appended with 
  # x for left table, y for right table by default
merged_income = pd.merge(income_2008,income_2013, on = "FIPS",
                         suffixes = ("_2008","_2013"))

### If the reference key column names are different, left_on and right_on to be given
income_2008.columns = ["FIPS_2008","Income_2008"]
income_2013.columns = ["FIPS_2013","Income_2013"]
merged_income = pd.merge(income_2008,income_2013, 
                         left_on = "FIPS_2008", right_on = "FIPS_2013")

## Which state had the highest percentage increase in income from 2008 to 2013
merged_income["Percent_Change"] = (merged_income["Income_2013"] - 
                                   merged_income["Income_2008"])*100/merged_income["Income_2008"]
merged_income.loc[merged_income["Percent_Change"].idxmax(),"FIPS_2008"]

#### There is always possibility for records in primary key not completely matching
    # between left and right tables

income_2008_2 = income_2008.drop(1,axis = 0)  # FIPS 02 is removed
income_2013_2 = income_2013.drop(2,axis = 0) # FIPS 04 is removed
# axis = 0 for removing a row, axis  = 1 for removing a column

### OUTER JOIN: All entries retained. Filled with nulls wherever there is mismatch
merged_income_outer = pd.merge(income_2008_2,income_2013_2,
                               how = "outer",
                               left_on = "FIPS_2008",right_on = "FIPS_2013")

### INNER JOIN: Only matching entried retained
merged_income_inner = pd.merge(income_2008_2,income_2013_2,
                               how = "inner",
                               left_on = "FIPS_2008",right_on = "FIPS_2013")

### LEFT JOIN: All entries of left table retained. Mismatching entries in right gets null
merged_income_left = pd.merge(income_2008_2,income_2013_2,
                               how = "left",
                               left_on = "FIPS_2008",right_on = "FIPS_2013")

### RIGHT JOIN: All entries of right table retained. Mismatching entries in left gets null
merged_income_right = pd.merge(income_2008_2,income_2013_2,
                               how = "right",
                               left_on = "FIPS_2008",right_on = "FIPS_2013")

#################### FILE WRITING ##################################
merged_income.to_csv("output/merged_income_20190804.csv")

# By default, index is saved as column. You can set it to False

merged_income.to_csv("output/merged_income_20190804.csv", index = False)



