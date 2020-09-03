import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Tools -> Preferences -> IPython Console -> Graphics -> Change from inline to automatic

wgdata = pd.read_csv("data/wg.csv")
############### EXPLORATORY ANALYSIS AND DATA QUALITY CHECKS #############
#1. number of nans in wg column
wgdata["wg"].isnull().sum()
wgdata["wg"].count() # number of valid entries

#2. number of nans in metmin
wgdata["metmin"].isnull().sum()

#3. how many observations had a null either in wg or in metmin column?
sum(wgdata["wg"].isnull() | wgdata["metmin"].isnull())

#4. extract the observations which don't have null in "wg" as well as "metmin" columns
wgclean = wgdata.dropna(subset = ["wg","metmin"])

#5. how many people have gained weight above average weight gain of the data
sum(wgdata["wg"] > wgdata["wg"].mean()) # 100 people

#6. what is the gender of the person with the highest weight gain?
wgdata.loc[wgdata["wg"] == wgdata["wg"].max(),"Gender"]

#6a. In this data, who gained more weight, male or female?
wgdata.groupby("Gender")["wg"].agg([np.mean, np.median, max])
# In this data, male have generally gained more weight compared to female

#7. get the count of above avg wg people by shift
wg_abv_avg_wg = wgdata[wgdata["wg"] > wgdata["wg"].mean()]
wg_abv_avg_wg.groupby("Shift").size()
sum(wg_abv_avg_wg.groupby("Shift").size()) # 100

#8. percentage of male in each shift
wgdata.groupby("Gender").size() # overall there are less male compared to female
wg_male = wgdata.loc[wgdata["Gender"] == "M",:]

tot_male_by_shift = wg_male.groupby("Shift").size()
tot_people_by_shift = wgdata.groupby("Shift").size()
tot_male_by_shift/tot_people_by_shift

## Alternative understanding: Spread of male across shifts
total_male_count = wg_male.shape[0]
wg_gp_shift =  wg_male.groupby("Shift")
wg_gp_shift["Gender"].count()/ total_male_count

#9. Create a copy of the data frame and impute missing values with mean of the column
wg_imputed = wgdata.copy()
wg_imputed["wg"] = wg_imputed["wg"].fillna(wg_imputed["wg"].mean())
wg_imputed["metmin"] = wg_imputed["metmin"].fillna(wg_imputed["metmin"].mean())

####################### HISTOGRAM ##########################################
## Used for visualizing distribution of an array of values
## Bins an array and plots the count in each bin

# Generating random incomes of an organization with 1000 employees
  # Average income is 30000, standard deviation is 5000
np.random.seed(1234)
income_rand = np.random.normal(30000,5000,1000)

plt.hist(income_rand, color = "red", edgecolor = "black")
plt.xlabel("Income Bins")
plt.ylabel("Count of Employees")
plt.title("Distribution of Income")

# By default, the number of bins is 10
  # the entire range between min and max is split into 10 bins

plt.figure() # creates a new figure
plt.hist(income_rand, bins = 20, color = "red", edgecolor = "black")

plt.figure()
plt.hist(income_rand, bins = range(10000,60000,5000),
         color = "red", edgecolor = "black") # custom binning

## Plot the histogram of weight gain. 
    # What weight gain range has more people?
plt.hist(wgdata["wg"])
# warnings/error shows up because matplotlib cannot handle nulls properly

plt.hist(wgclean["wg"])
plt.xlabel("Weight Gain Bins")
plt.ylabel("Number of people")
plt.title("Weight Gain Distribution")

wgmale = wgclean[wgclean["Gender"] == "M"]
wgfemale = wgclean[wgclean["Gender"] == "F"]
plt.hist(wgfemale["wg"], color = "red")
plt.hist(wgmale["wg"], color = "blue")

############ subplot ###########################
fig, (ax_m, ax_f) = plt.subplots(1, 2, sharex = True, sharey = True)
ax_m.hist(wgfemale["wg"], color = "red")
ax_f.hist(wgmale["wg"], color = "blue")

f, (ax1, ax2) = plt.subplots(2, 1, sharex = True, sharey = True)
ax1.hist(wgfemale["wg"], color = "red")
ax2.hist(wgmale["wg"], color = "blue")

plt.subplots(2, 2)
plt.subplots(3, 1)

#### pandas plot - pandas internally uses matplotlib
wgclean["wg"].plot.hist() # series plot
wgclean["metmin"].plot.hist()

## dataframe plot
wgclean.hist() # all numeric columns gets a histogram plot

pd.DataFrame.hist(wgclean, column = "wg")
pd.DataFrame.hist(wgclean, column = "wg", by = "Gender") # automatically does the subplot
pd.DataFrame.hist(wgclean, column = "wg", by = "Shift")

##################### boxplot #############################################
### another type of visualization for distribution
### Based on percentiles
### Tukey's Boxplot
   ## Proposed as an alternative t0 six sigma
   ## Replaced mean and standard deviation with medians and percentiles
## Very popular metod for outlier detection
plt.hist(income_rand)

np.median(income_rand) # 30,088
Q2_inc = np.percentile(income_rand,50) # same as median; 50% of the population below this value
Q1_inc = np.percentile(income_rand,25) # 25% of population below 26878
Q3_inc = np.percentile(income_rand,75) # 75% of population below 33344
IQR = Q3_inc - Q1_inc # Inter Quartile Range
UWL = Q3_inc + 1.5*IQR # 43042 # Upper Whisker Line
LWL = Q1_inc - 1.5*IQR # 17179 # Lower Whisker Line
# Values outside the whisker lines can be treated as outliers and removed
plt.boxplot(income_rand)

# Draw boxplot for weight gain
plt.boxplot(wgclean["wg"])
# Calculate quartiles, whisker lines and 
  # check whether they match in visualization
wgclean["wg"].describe()
Q1_wg = 8
Q2_wg = 15
Q3_wg = 20
IQR = Q3_wg - Q1_wg # 12
UWL = Q3_wg + 1.5*IQR # 38; Cannot go above maximum
LWL = Q1_wg - 1.5*IQR # -10; Cannot go below minimum
min(wgclean["wg"]) # 2; Lower whisker line capped to minimum

# How to remove outlier?
wg_outlier_removed = wgclean[(wgclean["wg"] <= UWL) & (wgclean["wg"] >= LWL)]
plt.boxplot(wg_outlier_removed["wg"])

## pandas plot
wgclean["wg"].plot.box() # plot on the series

# plot on the dataframe
wgclean.boxplot(column = "wg", by = "Gender")
wgclean.boxplot(column = "wg", by = "Shift")

############### Scatter Plot #########################################
### X - Y Plot
### used for visualizing relationship between 2 variables

### Matplotlib
plt.scatter(wgclean["metmin"], wgclean["wg"], s = 10, c = "red")
plt.xlabel("Activities level (metmin)")
plt.ylabel("Weight Gain (lbs)")
plt.title("Activities vs Weight Gain")
## As activity level increases, weight gain decreases

## Providing different color for male and female
wgclean["Gender_Color"] = "red"
wgclean.loc[wgclean["Gender"] == "M", "Gender_Color"] = "blue"
plt.scatter(wgclean["metmin"], wgclean["wg"], s = 10, 
            c = wgclean["Gender_Color"])

### Pandas
wgclean.plot.scatter("metmin","wg")

############### Bar Plot ######################################
## Comparing simple numbers
genderwise_wg = wgclean.groupby("Gender")["wg"].agg(np.mean)
genderwise_wg.plot.bar() # pandas plot
plt.bar(["F","M"],genderwise_wg)  # matplotlib

############## Pie chart ######################################
## To visualize contribution/share
genderwise_count = wgclean.groupby("Gender").size()
genderwise_count.plot.pie() # pandas
plt.pie(genderwise_count) # matplotlib

###############33 Line Plot ####################################
stockdata = pd.read_csv("data/Stock_Price.csv")
## weekly average closing stock prices of DELL and Intel from Jan 2010
## Data for 76 weeks

## matplotlib
plt.plot(stockdata["DELL"])
plt.plot(stockdata["Intel"])

## pandas
stockdata["DELL"].plot.line()
stockdata.plot.line()

##### Time Series Plots
## If the index of a Series/Data frame is of date time type, 
  # then they behave as time series data
  
# pandas date range functions can be used for generating sequence of dates
pd.date_range(start = "2018-08-01", periods = 30, freq = "D")
pd.date_range(start = "2018-08-01", periods = 12, freq = "W")
pd.date_range(start = "2018-08-01", periods = 4, freq = "M")

stockdata.index = pd.date_range(start = "2010-01-01", periods = 76, freq = "W")
stockdata.plot.line()

stockdata.index = pd.date_range(start = "2010-01-01", periods = 76, freq = "M")
stockdata.plot.line()

#################### Assignment ##############################################

#Airquality
airquality = pd.read_csv("data/airquality.csv")

#1. Get the histogram distribution of Solar.R
plt.hist(airquality["Solar.R"], bins = 5)# matplotlib
plt.hist(airquality["Temp"])
airquality["Solar.R"].plot.hist(bins = range(0,400,50)) # pandas

#2. Get the boxplot distribution of temperature
plt.boxplot(airquality["Temp"]) # matplotlib
airquality["Temp"].plot.box() # pandas
airquality.boxplot(column = "Temp", by = "Month")

airquality["Temp"].describe()

#3. Generate a scatter plot between temperature and solar.R
plt.scatter(airquality["Temp"],airquality["Solar.R"]) # matplotlib
airquality.plot.scatter("Temp","Solar.R") # pandas

#4. Generate a line plot of Solar.R. Create a date column and use 
 # that as x axis in line plot. 
 # Note: Date and time available in data frame. 
 # You may have to refer the documentation of data to know the year.

airquality.index = pd.date_range(start = "1973-05-01", 
                                 periods = 153, freq = "D")
plt.plot(airquality["Solar.R"])

## Note: If there are missing dates in between, above approach will not work
     # Example: Daily stock prices will not have data for weekend
aqclean = airquality.dropna() # removing missing values created missing dates
aqclean.index = pd.to_datetime({"day": aqclean["Day"],
                "month": aqclean["Month"],
                "year": 1973})
plt.plot(aqclean["Solar.R"])
aqclean["Solar.R"].plot.line()
aqclean["Temp"].plot.line()

#Mtcars
mtcars = pd.read_csv("data/mtcars.csv")
#1. Compare the mpg boxplot distribution of automatic vs manual transmission cars
mtcars.boxplot(column = "mpg", by = "am")

#2. Compare the boxplot distribution of mpg of cars by gears and transmission. 
  # One mpg distribution box per gear-am combination
mtcars.boxplot(column = "mpg", by = ["gear","am"])

#3. Generate a scatter plot between mpg and weight of the car
mtcars.plot.scatter("wt","mpg")

