import pandas as pd
import numpy
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from data_process import *
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


data = get_month_data('C:\\Users\\olive\\Documents\\CO2\\')
train_data = data[(data.datetime < '2013-01-01')]

test_data = data[(data.datetime >= '2013-01-01')]

# test_data_no_covid = data[(data.datetime >= '2013-01-01') & (data.datetime <= '2020-01-01')].set_index('datetime')
# test_data_no_covid.index = pd.DatetimeIndex(test_data_no_covid.index)

train_data = train_data.drop(columns=['CO2_seasonal'])

model = LinearRegression()
model.fit(train_data.datetime.values.astype("float64").reshape(-1, 1), train_data.CO2.values.reshape(-1, 1))

preds = model.predict(test_data.datetime.values.astype("float64").reshape(-1, 1))
rmse = mean_squared_error(test_data.CO2.values, preds, squared=False)
print(rmse)