import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.utils import check_random_state
from data_process import *
from sklearn.metrics import mean_squared_error


data = get_month_data('C:\\Users\\olive\\Documents\\CO2\\')
train_data = data[(data.datetime < '2013-01-01')]

test_data = data[(data.datetime >= '2013-01-01')].set_index('datetime')
test_data.index = pd.DatetimeIndex(test_data.index)

train_data = train_data.set_index('datetime').drop(columns=['CO2_seasonal'])
train_data['diff'] = train_data.diff()
train_data = train_data.dropna()
train_data.index = pd.DatetimeIndex(train_data.index).to_period('M')
signal = train_data['diff'].values.reshape(-1, 1)
for i in range(1, 16):
    model = hmm.GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000, tol=0.001, random_state=42).fit(signal)
    state_sequence = model.predict(signal)
    X, Z = model.sample(n_samples=len(test_data)-1, currstate=state_sequence[-1])  # get ~11 years of preds for diff
    # print(f'Transmission Matrix Recovered:\n{model.transmat_.round(3)}\n\n')
    last_data = [train_data['CO2'].values[-1]]
    for difference in X:
        new_pt = last_data[-1]+difference
        last_data.append(float(new_pt))
    rmse = mean_squared_error(test_data.CO2.values, last_data, squared=False)
    print(i, 'rmse: ', rmse)
    # fontsize = 15
    # fig = plt.figure()
    # plt.plot(test_data.index, test_data.CO2.values, label=r'Ground truth', color='r')
    # plt.plot(test_data.index, last_data, label=r'HMM predictions ('+str(i)+' hidden states)', color='b')
    # plt.title('CO$_2$ emissions over time', fontsize=fontsize)
    # plt.xlabel('Date', fontsize=fontsize)
    # plt.ylabel('CO$_2$ emissions (ppm)', fontsize=fontsize)
    # plt.legend(fontsize=fontsize)
    # plt.show()
    # fig.savefig('C:\\Users\\olive\\Documents\\CO2\\hmm_results\\'+'hmm_n_comp_'+str(i)+'_2013_present.png')
