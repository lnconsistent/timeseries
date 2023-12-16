import pmdarima as pm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from data_process import *


data = get_month_data('C:\\Users\\olive\\Documents\\CO2\\')
train_data = data[(data.datetime < '2013-01-01')]

test_data = data[(data.datetime >= '2013-01-01')].set_index('datetime')
test_data.index = pd.DatetimeIndex(test_data.index)

train_data = train_data.set_index('datetime').drop(columns=['CO2_seasonal'])
stationary_signal = train_data['CO2'].diff().dropna()
seasonal_signal = train_data['CO2'].diff().diff(periods=12).dropna()
# signal = train_data['CO2']
signal = seasonal_signal
n_lags = 24

stationary_adf = adfuller(stationary_signal)
seasonal_adf = adfuller(seasonal_signal)
print(stationary_adf)
print(seasonal_adf)
#
# fontsize = 15
# fig = plt.figure()
# plt.plot(data.datetime, data.CO2.values, 'r', linewidth=0.75, label=r'Raw signal')
# plt.title('CO$_2$ emissions over time', fontsize=fontsize)
# plt.xlabel('Date', fontsize=fontsize)
# plt.ylabel('CO$_2$ emissions (ppm)', fontsize=fontsize)
# plt.legend(fontsize=fontsize)
# plt.show()
# fig.savefig('C:\\Users\\olive\\Documents\\CO2\\eda\\data.png')


# acf_estimate = acf(signal, nlags=n_lags)
# pacf_estimate = pacf(signal, nlags=n_lags)
# acf_error_estimate = 2 * np.ones(n_lags) / np.sqrt(len(signal))
# pacf_error_estimate = 2 * np.ones(n_lags) / np.sqrt(len(signal))
#
# # Let's plot our estimate along with the estimator errors.
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=80)
# fig.suptitle('ACF/PACF for seasonally differenced data', fontsize=20)
# fontsize = 15
# colors = ['#7fcdbb', '#2c7fb8']
#
# # Plot comparison between estimated ACF and uncertanties.
# ax1.stem(acf_estimate, linefmt=colors[0], markerfmt=colors[0], basefmt='k', label='Estimated ACF')
# ax1.fill_between(np.arange(1,n_lags+1), acf_error_estimate, -acf_error_estimate, color=colors[1], label='Error in Estimate',
#                  alpha=0.2)
# ax1.set_ylim([-.6, 1.2])
# ax1.set_title('ACF Estimate', fontsize=fontsize)
# ax1.set_xlabel(r'$|h|$', fontsize=fontsize)
# ax1.set_ylabel(r'$\rho(|h|)$', fontsize=fontsize)
# ax1.legend(fontsize=fontsize)
#
# # Plot comparison between our esimated PACF and uncertanties.
# ax2.stem(pacf_estimate, linefmt=colors[0], markerfmt=colors[0], basefmt='k', label='Estimated PACF')
# ax2.fill_between(np.arange(1,n_lags+1), pacf_error_estimate, -acf_error_estimate, color=colors[1], label='Error in Estimate',
#                  alpha=0.2)
# ax2.set_ylim([-.6, 1.2])
# ax2.set_title('PACF Estimate', fontsize=fontsize)
# ax2.set_xlabel(r'$|h|$', fontsize=fontsize)
# ax2.set_ylabel(r'$\mathrm{PACF}(|h|)$', fontsize=fontsize)
# ax2.legend(fontsize=fontsize)
#
# plt.show()
# fig.savefig('C:\\Users\\olive\\Documents\\CO2\\eda\\season_diff_acf_pacf', dpi=fig.dpi)


# model = pm.auto_arima(data.CO2,
#                       m=12,
#                       d=None,
#                       start_p=0, start_q=0,
#                       max_p=10, max_q=10,
#                       D=None,
#                       start_P=0, start_Q=0,
#                       max_P=10, max_Q=10,
#                       trace=True,
#                       error_action='ignore',
#                       suppress_warnings=True,
#                       stepwise=True)
#
# # print model summary
# print(model.summary())
