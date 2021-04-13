import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

df = pd.read_csv('data/generated_data/stage_2_optimization_training.csv')
df = df.drop(['ID', 'cl','ke','ka','alpha'], axis = 'columns')

y = df.outcomes.values
X = df.drop('outcomes', axis = 'columns').values
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size = 0.6)

param_grid = {'max_depth' : [3,4,5,10,None],
              'max_features': [1,2,3,4,5,None]}

rf_gscv = GridSearchCV(RandomForestRegressor(n_estimators = 1000), 
                    param_grid = param_grid, 
                    cv = 5, 
                    scoring = 'neg_mean_squared_error',
                    verbose = 2).fit(Xtrain, ytrain)

with open('data/rf_gscv.txt', 'wb') as file:
    pickle.dump(rf_gscv, file)