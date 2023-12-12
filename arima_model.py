import pandas as pd
import numpy
import matplotlib.pyplot as plt
import scipy as sp
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_process import arma_acovf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import *
from data_process import *


# p, d, q = 4, 1, 2     # auto-arima for day/week resolution data
# Model:SARIMAX(4, 1, 1)x(1, 0, 1, 12)    # auto-arima for month resolution data

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


data = get_week_data('C:\\Users\\olive\\Documents\\CO2\\')
data = data.set_index('datetime')
signal = data['CO2']

# n_lags = 20
# acf_estimate = acf(signal, nlags=n_lags)
# pacf_estimate = pacf(signal, nlags=n_lags)
# acf_error_estimate = 2/np.sqrt(len(signal))
# pacf_error_estimate = 2/np.sqrt(len(signal))
#
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(40, 20))
# fontsize = 15
# colors = ['#7fcdbb', '#2c7fb8']
#
# ax1.plot(signal, color='grey', label='Data')
# ax1.legend(fontsize=fontsize)
# ax1.set_xlabel(r'Time index', fontsize=fontsize)
# ax1.set_ylabel(r'Signal', fontsize=fontsize)
# ax1.set_title('Data', fontsize=fontsize)
#
# sm.graphics.tsa.plot_acf(signal, ax2, lags=20)
# ax2.legend(fontsize=fontsize)
# ax2.set_xlabel(r'$|h|$', fontsize=fontsize)
# ax2.set_ylabel(r'$\rho(|h|)$', fontsize=fontsize)
# ax2.set_title('ACF', fontsize=fontsize)
#
# sm.graphics.tsa.plot_pacf(signal, ax3, lags=20)
# ax3.legend(fontsize=fontsize)
# ax3.set_xlabel(r'$|h|$', fontsize=fontsize)
# ax3.set_ylabel(r'$\rho(|h|)$', fontsize=fontsize)
# ax3.set_title('PACF', fontsize=fontsize)
#
# plt.show()


import pmdarima as pm

model = pm.auto_arima(signal,
                      m=52,
                      d=None,
                      start_p=0, start_q=0,
                      max_p=100, max_q=100,
                      D=None,
                      start_P=0, start_Q=0,
                      max_P=100, max_Q=100,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)

# print model summary
print(model.summary())