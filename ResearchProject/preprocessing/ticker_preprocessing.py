from typing import Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import math

import yfinance as yf
from datetime import date as date, timedelta
import numpy as np

"""
Preprocessing module to return ticker data that has been requested. Also returns ticker dta preprocessed for keras 
models. 
"""


# Return raw data for ticker between specified dates
def get_ticker_data(ticker: str, date_start: date, date_end: date):

    # Ensure enough data retrieved for split - taking account of weekends
    if date_end - date_start < timedelta(days=10):
        date_start = date_end - timedelta(days=10)

    yf_ticker = yf.Ticker(ticker)
    df = yf_ticker.history(start=date_start, end=date_end, interval="1d")

    return df


# Return preprocessed data for ticker between specified dates
# Reshapes, scales, and splits data into train and test
def get_processed_ticker_data(ticker: str, date_start: date, date_end: date, train_percentage: Optional[float] = 0.8,
                              time_step: Optional[int] = None):

    # Get ticker data
    df = get_ticker_data(ticker, date_start, date_end)

    # Select close column and reshape
    data = df.loc[:, ['Close']]

    # split into train, test, and validation
    train, test = train_test_split(data, train_size=train_percentage, shuffle=False)
    test_dates = test

    # normalize the dataset
    scale = MinMaxScaler(feature_range=(0, 1))
    train = scale.fit_transform(train)
    test = scale.fit_transform(test)

    # Get validation set
    validation = data[-10:len(data)]

    # Ensure time step is set to an appropriate value - at least a 6th of total data size
    if time_step is None or time_step < (0.16 * len(data)):
        days = len(data)
        time_step = math.ceil(days * .1)

    # split into x and y
    train_x, train_y = __split_x_y(train, time_step)
    test_x, test_y = __split_x_y(test, time_step)

    # reshape x, manually assign batch step if necessary
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    if len(test_x) == 1:
        test_x = np.reshape(test_x, (1, test_x.shape[1], 1))
    else:
        test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    return train_x, train_y, test_x, test_y, test_dates, validation, scale, time_step


# Split dataset by time step
def __split_x_y(dataset, timestep):
    x, y = [], []
    for i in range(len(dataset) - timestep):
        x.append(dataset[i:(i + timestep), 0])
        y.append(dataset[i + timestep, 0])
    return np.array(x), np.array(y)
