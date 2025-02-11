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
conc_data_columns=concrete_data.columns
'''returns the column names of the data set as pandas index object '''
predictors = concrete_data[conc_data_columns[conc_data_columns !='Strength']]
'''by default pandas performed row based filtering see google docs (Deep Learning/page = Pandas Filtering for explanation)'''
target = concrete_data['Strength']
'''selects the 'Strength' column and returns a pandas series  which is a 1d array'''
print(target.head())
print(predictors.head())
predictors_norm = ((predictors-predictors.mean())/predictors.std())
print(predictors_norm.head())
'''normalizing the data '''
number_of_col=predictors_norm.shape[1]
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Input

def regression_model():
    model = Sequential()
    model.add(Input(shape=(number_of_col,)))
    '''number_of_col, tells the shape that a 1d array is being passed'''
    model.add(Dense(50,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(1))
    #compiling model 
    model.compile(optimizer='adam',loss='mean_squared_error')
    return model
model = regression_model()
model.fit(predictors_norm,target,validation_split=0.3,epochs=100,verbose=2)
'''fit method fits the model with the training data first two parameters are our input data validation split 0.3 means that 30 percent
of the data will be reserved for testing and epochs means that we will have 100 iterations verbose means how much info is printed during 
training 0 means no output 1 means progress bar 2 shows one line per epoch with the training and validation loss/accuracy,
 which can be helpful for tracking the model's performance.'''
