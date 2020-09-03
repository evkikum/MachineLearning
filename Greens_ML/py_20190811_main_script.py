
## Importing all functions in a package
import numpy as np
np.mean([1,2,3,4,5])
np.median([1,2,3,4,5])

## Importing all functions in a script
# Script should be in the same working directory
import py_20190811_fns1 as fn1
fn1.print_with_excalamtion("Hello")
fn1.print_sum(4,10)

## Importing selected functions from a package
from pandas import Series
s1 = Series([1,2,3], index = ["a","b","c"])

"""
Below lines will throw error as DataFrame function is not imported
df = DataFrame()
df = pd.DataFrame()
"""

## Importing selected functions from a script
from py_20190811_fns2 import calc_sum, calculate
calc_sum(5,10)
calculate(5,10,"diff")
#sum_diff2(5,10) # throws error as that function is not imported
