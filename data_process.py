import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

'''
BEFORE LOADING FILES REMOVE ALL NON-DATA VALUES AND COLUMN HEADERS FROM FILE
'''


def get_minute_data(data_dir):
    '''
    % Date and time are in Hawaii Standard Time
    % CO2 concentrations are reported on the Scripps 08A calibration scale
    %
    % CO2: Carbon Dioxide concentration; expressed in parts per million [ppm]
    %
    % Yr,Mn,Dy,Hr,Mi,   CO2
    '''
    ten_minute_path = data_dir+'ten_minute_in_situ_co2_mlo.csv'
    ten_min_header = ['year', 'month', 'day', 'hour', 'minute', 'CO2']
    # Add Datetime object as column
    df = pd.read_csv(ten_minute_path, names=ten_min_header, header=None)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
    # Handle NaN
    df['CO2'] = df['CO2'].replace('NaN', np.nan).astype(float)
    # remove irrelvant data (non-datetime/CO2 columns) and return
    return df[['CO2', 'datetime']]


def get_day_data(data_dir):
    '''
    % Yr    - Year
    % Mn    - Month
    % Dy    - Day
    % CO2   - CO2 baseline value
    % NB    - Number of hourly averages in the baseline
    % scale - Calibration scale code
    % sta   - Sampling station. Occasional samples have been recorded at the Maunakea Observatory (mko)
    %
    % Yr, Mn, Dy,    CO2, NB, scale, sta
    '''
    daily_path = data_dir+'daily_in_situ_co2_mlo.csv'
    daily_header = ['year', 'month', 'day', 'CO2', 'NB', 'scale', 'sta']
    df = pd.read_csv(daily_path, names=daily_header, header=None)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day']])
    # Handle NaN
    df['CO2'] = df['CO2'].replace('NaN', np.nan).astype(float)
    # remove irrelvant data (non-datetime/CO2 columns) and return
    return df[['CO2', 'datetime']]


def get_week_data(data_dir):
    '''
    " The data file below contains 2 columns indicating the date and CO2      "
    " concentrations in micro-mol CO2 per mole (ppm), reported on the 2012   "
    " SIO manometric mole fraction scale.  These weekly values have been     "
    " adjusted to 12:00 hours at middle day of each weekly period as         "
    " indicated by the date in the first column.                             "
    '''
    weekly_path = data_dir+'weekly_in_situ_co2_mlo.csv'
    weekly_header = ['week', 'CO2']
    # Note: datetime column can be created just using the week column
    df = pd.read_csv(weekly_path, names=weekly_header, header=None)
    df['datetime'] = pd.to_datetime(df['week'])
    # return relevant columns
    return df[['CO2', 'datetime']]


def get_month_data(data_dir):
    '''
    " The data file below contains 10 columns.  Columns 1-4 give the dates in several redundant "
    " formats. Column 5 below gives monthly Mauna Loa CO2 concentrations in micro-mol CO2 per   "
    " mole (ppm), reported on the 2012 SIO manometric mole fraction scale.  This is the         "
    " standard version of the data most often sought.  The monthly values have been adjusted    "
    " to 24:00 hours on the 15th of each month.  Column 6 gives the same data after a seasonal  "
    " adjustment to remove the quasi-regular seasonal cycle.  The adjustment involves           "
    " subtracting from the data a 4-harmonic fit with a linear gain factor.  Column 7 is a      "
    " smoothed version of the data generated from a stiff cubic spline function plus 4-harmonic "
    " functions with linear gain.  Column 8 is the same smoothed version with the seasonal      "
    " cycle removed.  Column 9 is identical to Column 5 except that the missing values from     "
    " Column 5 have been filled with values from Column 7.  Column 10 is identical to Column 6  "
    " except missing values have been filled with values from Column 8.  Missing values are     "
    " denoted by -99.99                                                                         "
    " Column 11 is the 3-digit sampling station identifier"
    "                                                                                           "
    " CO2 concentrations are measured on the '12' calibration scale                             "
    "                                                                                           "
      Yr, Mn,    Date,      Date,     CO2,seasonally,        fit,  seasonally,      CO2, seasonally, Sta
    '''
    monthly_path = data_dir+'monthly_in_situ_co2_mlo.csv'
    monthly_header = ['year', 'month', 'date', 'date_alt', 'CO2', 'CO2_seasonal', 'CO2_smoothed', 'CO2_seasonal_smoothed',
                      'CO2_imputed', 'CO2_seasonal_imputed', 'sta']
    # Note: use date_alt to create datetime column
    # Add Datetime object as column
    df = pd.read_csv(monthly_path, names=monthly_header, header=None)
    df['datetime'] = pd.to_datetime(df['date_alt'])
    df['CO2'] = df['CO2'].replace(-99.99, np.nan).astype(float)
    return df[['CO2', 'datetime']]
