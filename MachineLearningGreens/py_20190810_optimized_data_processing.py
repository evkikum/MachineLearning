import pandas as pd
import numpy as np

# Record of 8 students
np.random.seed(10)
df = pd.DataFrame({
        'A':['female','male','female','male','female','male','female','female'],
        'B':['secA','secB','secB','secA','secA','secB','secB','secA'],
        'Math':np.random.randint(20,100,8),
        'Eng':np.random.randint(20,100,8),
        'Sci':np.random.randint(20,100,8),
        'Soc':np.random.randint(20,100,8),
        'Tam':np.random.randint(20,100,8)})

airquality = pd.read_csv("data/airquality.csv")
################3 apply #######################################################
### repetitively applies a function across rows or columns
  # axis = 0 for column wise operation
  # axis= 1 for row wise operation

### Column wise operation
df["Math"].mean()
df["Soc"].mean()
for i in df.columns[2:]:
    print(df[i].mean())

df_needed = df.iloc[:,2:]
avg_each_subject = df_needed.apply(np.mean,axis = 0)
median_each_subject = df_needed.apply(np.median,axis = 0)
min_each_subject = df_needed.apply(min,axis = 0)

# Following alternative approach will internally call the apply function
 # only certain standard functions are implemented
 # whereas using apply, any function can be used
df_needed.mean() 
df_needed.median()

## Row wise operation
for i in df.index:
    print(df.iloc[i,2:].mean())
avg_each_student = df_needed.apply(np.mean, axis = 1)

#13.Calculate average values of Ozone, Solar, Wind and Temperature 
aq_needed = airquality.iloc[:,:4]
aq_avg_multicols = aq_needed.apply(np.mean, axis = 0)
# last 2 lines in 1 line
aq_avg_multicols = airquality.iloc[:,:4].apply(np.mean, axis = 0) 
aq_needed.mean() # alternative

################# groupby #################################################
## splits the data based on a categorical variable
df_male = df[df["A"] == "male"]
df_female = df[df["A"] == "female"]
# above method is tedious if there are more levels to categorical variables

for i in ["male","female"]:
    print(df[df["A"] == i])
    
# Grouping by gender
df_gb_gender = df.groupby("A")
type(df_gb_gender) # groups of dataframes
df_gb_gender.get_group("male")
df_gb_gender.get_group("female")

# grouping by section
df_gb_sec = df.groupby("B")
df_gb_sec.get_group("secA")
df_gb_sec.get_group("secB")

# grouping by gender and section
df_gb_gender_sec = df.groupby(["A","B"])
df_gb_gender_sec.get_group(("male","secA")) # passing tuple of combination
df_gb_gender_sec.get_group(("female","secA")) # passing tuple of combination
df_gb_gender_sec.get_group(("male","secB")) # passing tuple of combination
df_gb_gender_sec.get_group(("female","secB")) # passing tuple of combination

### split airquality data based on month
aq_gb_month = airquality.groupby("Month")
aq_gb_month.get_group(5)
aq_gb_month.get_group(8)

########### aggregate #########################################################
## repetitively do a function across groups

## Single function on a column across groups
df_gb_gender["Math"].agg(np.mean)

# groupby and aggregation can also be done in 1 line
df.groupby("A")["Math"].agg(np.mean)

## Single function on multiple columns across groups
df_gb_gender[["Math","Sci"]].agg(np.mean)
df_gb_sec["Math"].agg(np.mean)
df_gb_gender_sec[["Math","Sci"]].agg(min)

## Multiple functions on a column across groups
df_gb_gender["Math"].agg([np.mean,np.median])

## Multiple functions on multiple columns across groups
df_gb_gender[["Math","Sci"]].agg([min, max])

## Different function for different columns across groups
df_gb_gender.agg({"Math": np.mean,
                  "Sci": np.median,
                  "Tam": [min,max]})
    
#14.Calculate month-wise average Ozone 
aq_gb_month["Ozone"].agg(np.mean)

#15.Calculate month-wise average Ozone, Solar, Wind and Temperature
cols_needed = airquality.columns[:4]
aq_gb_month[cols_needed].agg(np.mean)










