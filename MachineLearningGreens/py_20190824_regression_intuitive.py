import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mtcars = pd.read_csv("data/mtcars.csv")
mtcars.plot.scatter("wt","mpg")

# weight of the car can be estimated during design stage
# however mileage can be known only after manufacturing and road test

# can you predict mileage of a car designed with 4.5 tonne weight?

# can you predict mileage of a car designed with 1 tonne weight?
plt.scatter(mtcars["wt"],mtcars["mpg"])
plt.xlabel("Weight")
plt.ylabel("Mileage")
plt.ylim([0,50])
plt.xlim([0,6])

# Line Equation: y = mX + C
  # m (slope): what is the change in Y for a change in X
  # C (constant/intercept): Where does the line intercept Y axis when X = 0

## Arrive at a relationship with following form
# mpg = m*wt + C
  
new_wt = np.array([1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6])
## IN THE BELOW  numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)[source], start = 1, stop = 6, num = 100 ( THis is the total count)
new_wt = np.linspace(1,6,100) 

predicted_mpg1 = 10*new_wt + 20
plt.scatter(mtcars["wt"],mtcars["mpg"])
plt.scatter(new_wt,predicted_mpg1)

predicted_mpg2 = -10*new_wt + 20
plt.scatter(mtcars["wt"],mtcars["mpg"])
plt.scatter(new_wt,predicted_mpg2)

predicted_mpg3 = -10*new_wt + 45
plt.scatter(mtcars["wt"],mtcars["mpg"])
plt.scatter(new_wt,predicted_mpg3)

predicted_mpg4 = -8*new_wt + 45
predicted_mpg5 = -7*new_wt + 45
predicted_mpg6 = -6*new_wt + 42
plt.scatter(mtcars["wt"],mtcars["mpg"])
plt.scatter(new_wt,predicted_mpg4, c = "red")
plt.scatter(new_wt,predicted_mpg5, c = "green")
plt.scatter(new_wt,predicted_mpg6, c = "purple")

##Q: How to pick the best fitting relatiopnship?

# A1: Evaluate on a test data for which mpg is known and compare with prediction
# A2: Evaluate on the given data and check the goodness of fit

mtcars_model_eval = pd.DataFrame()
mtcars_model_eval["actual_mpg"] = mtcars["mpg"]
mtcars_model_eval["fitted_mpg4"] = -8*mtcars["wt"] + 45
mtcars_model_eval["fitted_mpg5"] = -7*mtcars["wt"] + 45
mtcars_model_eval["fitted_mpg6"] = -6*mtcars["wt"] + 42

mtcars_model_eval["rel4_error"] = mtcars["mpg"] - mtcars_model_eval["fitted_mpg4"]
mtcars_model_eval["rel5_error"] = mtcars["mpg"] - mtcars_model_eval["fitted_mpg5"]
mtcars_model_eval["rel6_error"] = mtcars["mpg"] - mtcars_model_eval["fitted_mpg6"]

## Predict score of India in 3 matches
act_score = [280, 310, 320]
pred1 = [290, 309, 310]
pred2 = [276, 312, 317]
err1 = [-10, 1, 10]
err2 = [-4, -2, 3]
# Just by looking at numbers, err2 looks lesser than err1

## Median Error
np.median(err1) # 1
np.median(err2) # -2

## Mean Error
np.mean(err1) # 0.33
np.mean(err2) # -1

# Above 2 approaches states that err1 is less which is incorrect
# negative errors are cancelling out with positive errors

## Mean absolute error
np.mean(np.abs(err1)) # 7
np.mean(np.abs(err2)) # 3
# err2 is less which makes sense

## Mean Absolute Percent Error
#abs_percent_error = abs(actual - predicted)/actual
#np.mean(abs_percent_error)

def MAPE(actual, predicted):  # pulled from functions session    
    actual_np = np.array(actual)
    predicted_np = np.array(predicted)
    ape = abs(actual_np - predicted_np)*100/actual_np    
    mean_ape = np.mean(ape)
    median_ape = np.median(ape)
    return([mean_ape,median_ape])

MAPE(mtcars_model_eval["actual_mpg"], 
     mtcars_model_eval["fitted_mpg4"]) # 18% Mean APE, 11.4% Median APE
MAPE(mtcars_model_eval["actual_mpg"], 
     mtcars_model_eval["fitted_mpg5"]) # 20% Mean APE, 17.9% Median APE
MAPE(mtcars_model_eval["actual_mpg"], 
     mtcars_model_eval["fitted_mpg6"]) # 19% Mean APE, 15.7% Median APE
## Relationship 1 has the least error







