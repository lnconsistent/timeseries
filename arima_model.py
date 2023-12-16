import pandas as pd
import numpy
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from data_process import *
from sklearn.metrics import mean_squared_error


data = get_month_data('C:\\Users\\olive\\Documents\\CO2\\')
train_data = data[(data.datetime < '2013-01-01')]

test_data = data[(data.datetime >= '2013-01-01')].set_index('datetime')
test_data.index = pd.DatetimeIndex(test_data.index)

# test_data_no_covid = data[(data.datetime >= '2013-01-01') & (data.datetime <= '2020-01-01')].set_index('datetime')
# test_data_no_covid.index = pd.DatetimeIndex(test_data_no_covid.index)

train_data = train_data.set_index('datetime').drop(columns=['CO2_seasonal'])
train_data.index = pd.DatetimeIndex(train_data.index).to_period('M')
fontsize = 15
colors = ['#7fcdbb', '#2c7fb8']

model = ARIMA(train_data, order=(4, 1, 1), seasonal_order=(1, 0, 1, 12))  # seasonal parameters correspond to a seasonal ARMA(12) term
res = model.fit()

with_covid = res.forecast(len(test_data)+5)   # adding 5 because the arima stops 5 months before the end of the sequence for some reason
# without_covid = res.forecast(test_data_no_covid.index[-1])

rmse = mean_squared_error(test_data.CO2.values, with_covid.values[5:], squared=False)
print(rmse)

fig = plt.figure()
plt.plot(test_data.index, test_data.CO2.values, label=r'Ground truth', color='r')
# plt.plot(data.datetime, data.CO2.values, label=r'Ground truth', color='r')
plt.plot(with_covid.index, with_covid.values, label=r'Predictions', color='b')
plt.title('CO$_2$ emissions over time', fontsize=fontsize)
plt.xlabel('Date', fontsize=fontsize)
plt.ylabel('CO$_2$ emissions (ppm)', fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.show()
# fig.savefig('C:\\Users\\olive\\Documents\\CO2\\arima_results\\arima_2003_present_total.png')


'''
Best model:  ARIMA(4,1,1)(1,0,1)[12] intercept
Total fit time: 211.259 seconds
                                     SARIMAX Results                                      
==========================================================================================
Dep. Variable:                                  y   No. Observations:                  782
Model:             SARIMAX(4, 1, 1)x(1, 0, 1, 12)   Log Likelihood                -440.360
Date:                            Wed, 13 Dec 2023   AIC                            898.720
Time:                                    04:08:59   BIC                            940.665
Sample:                                         0   HQIC                           914.852
                                            - 782                                         
Covariance Type:                              opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
intercept      0.0011      0.001      1.566      0.117      -0.000       0.002
ar.L1          0.8087      0.039     20.854      0.000       0.733       0.885
ar.L2         -0.0563      0.034     -1.648      0.099      -0.123       0.011
ar.L3         -0.1853      0.032     -5.815      0.000      -0.248      -0.123
ar.L4         -0.0859      0.032     -2.653      0.008      -0.149      -0.022
ma.L1         -0.8475      0.030    -28.614      0.000      -0.906      -0.789
ar.S.L12       0.9843      0.006    168.238      0.000       0.973       0.996
ma.S.L12      -0.5984      0.026    -23.451      0.000      -0.648      -0.548
sigma2         0.1748      0.006     27.315      0.000       0.162       0.187
===================================================================================
Ljung-Box (L1) (Q):                   0.04   Jarque-Bera (JB):               389.14
Prob(Q):                              0.85   Prob(JB):                         0.00
Heteroskedasticity (H):               0.62   Skew:                            -0.16
Prob(H) (two-sided):                  0.00   Kurtosis:                         6.44
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
'''
