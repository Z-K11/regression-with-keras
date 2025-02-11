import pandas as pd
import numpy as np
concrete_data =pd.read_csv("./csv_data/concrete_data.csv")
print(concrete_data.head())
print(concrete_data.shape)
print(concrete_data.describe())
print(concrete_data.isnull().sum())
'''.isnull() returns a data frame that is the same shape as conrete_data but has true for any NaN value and false otherwise meaning that 
it is 1 for missing value and 0 for value .sum() adds these values if the sum is 0 we have no missing value if the sum is n then we have 
n missing values '''