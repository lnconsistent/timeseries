import pandas as pd
import numpy
import matplotlib.pyplot as plt
import scipy as sp
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import *
from data_process import *

a = get_day_data('C:\\Users\\olive\\Documents\\CO2\\')
p, d, q = 1, 1, 1

def get_train_test(df):
    mse_scores = []
    # get train_test_split
    tscv = TimeSeriesSplit(n_splits=int((len(df)-3)/3))
    for train_index, test_index in tscv.split(df):
        train, test = df.iloc[train_index], df.iloc[test_index]

        model = ARIMA(train['temperature'], order=(p, d, q))
        arima_model = model.fit()

        # Predict using the test data
        predictions = arima_model.predict(start=len(train), end=len(train) + len(test) - 1)

        # Evaluate your predictions against the actual test data
        mse = mean_squared_error(test['temperature'], predictions)
        mse_scores.append(mse)

    # Print MSE scores for each fold
    for i, mse in enumerate(mse_scores, 1):
        print(f"Fold {i} MSE: {mse}")

    # Optionally, you might want to calculate the average MSE across all folds
    average_mse = sum(mse_scores) / len(mse_scores)
    print(f"\nAverage MSE: {average_mse}")