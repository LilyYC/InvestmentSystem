# -*- coding: utf-8 -*-
from yahoo_stock import Stock
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from datetime import date
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

asset_name = '000831.SZ'
xitu = Stock(asset_name)
data = xitu.get_historical_price()
data.sort_values(by=['Date'], inplace=True)
data.columns = ['Date', 'Price']

# Feature Extraction
data.index =pd.to_datetime(data['Date'])
data['year'] = data.index.year
data['month'] = data.index.month
data['day'] = data.index.day

# random sampling of 6 raws
data.sample(6,random_state=0)

# EDA
temp = data.groupby([data.index])[asset_name].mean()
temp.plot(figsize=(15,5), title='Adjusted Close Prices for Xitu', fontsize=14)

data.groupby('month')[asset_name].mean().plot.bar()
data.groupby('month')[asset_name].median().plot.bar()
data.groupby('year')[asset_name].mean().plot.bar()
data.groupby('year')[asset_name].median().plot.bar()

# Build models for Time Series Forecasting
# 1. split dataset for train and validation dataset
test = data[date(2020,1,1):]
train = data[:date(2019,12,31)]

# ARIMA model
# Make sure data is stationary
def check_acf(x):
    fig, ax = plt.subplots(3, figsize=(12,6))
    ax[0] = plot_acf(x, ax=ax[0], lags=25)
    ax[1] = plot_pacf(x, ax=ax[1], lags=25)
    ax[2].plot(x)
# Remove trend and seasonality from the data

# 1. Apply Dickey Fuller test to check stationarity of the series
def test_stationarity(timeseries):
    ''' Visualized and display results from Dickey Fuller test
    '''
    # Determine rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    # Plot rolling statistics
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean,color='red', label='Rolling Mean')
    plt.plot(rolstd, color='green', label='Rolling std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)

    print("Results of Dickey Fuller test")
    adft = adfuller(timeseries, autolag='AIC')

    output = pd.Series(adft[0:4], index=['Test Statistics', 'p-value', 'Number of lags used','Number of Observations used'])
    for key, values in adft[4].items():
        output['critical value (%s)'%key] = values
    print(output)

test_stationarity(train[asset_name])

# The statistics shows that the time series is non-stationary 
# as Test Statistics > Critical value
# and the p-value is greater than 5%, we fail to reject the 
# null hypothesis that the time seires is not stationary

# Making it stationary
# We observe from above plot that there is an decreading trend in the data
# So we can apply transformation which penalizes smaller values more than
# large ones, such as exponential transformation

train_exp = np.exp(train[asset_name])
test_exp = np.exp(test[asset_name])
moving_avg = train_exp.rolling(12).mean()
plt.plot(train_exp)
plt.plot(moving_avg,color='red')
plt.show()    

test_stationarity(train_exp)

# try logarithm transformation
train_log = np.log(train[asset_name])
test_log = np.log(test[asset_name])
moving_avg = train_log.rolling(24).mean()
plt.plot(train_log)
plt.plot(moving_avg,color='red')
plt.show()
train_log_moving_avg = train_log-moving_avg
train_log_moving_avg.dropna(inplace=True)
test_stationarity(train_log_moving_avg)

# 2. stabilize the mean of the time series - Use differencing
train_log_diff = train_log_moving_avg - train_log_moving_avg.shift(1)
train_log_diff = train_log_diff.dropna()
test_stationarity(train_log_diff)

#%% Remove seasonality

#%%
model = auto_arima(train_log_diff, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train_log)
forecast = model.predict(n_periods=len(test))
forecast = pd.DataFrame(forecast,index = test_log.index,columns=['Prediction'])
#plot the predictions for validation set
plt.plot(np.exp(train_log), label='Train')
plt.plot(np.exp(test_log), label='Test')
plt.plot(np.exp(forecast), label='Prediction')
plt.title('Wukuang Xitu Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Actual Stock Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()

from math import sqrt
rms = sqrt(mean_squared_error(test_log,forecast))
print("RMSE: ", rms)
