import yfinance as yf
import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout
from numpy import isnan
import datetime as dt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sentiment_analysis import SentAnVader
from sentiment_analysis.sa_scraper import WebScraperTwitter
from post_processing import RegressionAccuracy

"""
Multi variate prediction model - creates an ensemble model combining sentiment scores and ticker data for predictions. 
"""

np.random.seed(1)  # Fix for reproducible results
train_percentage = 0.8

# Get data for ticker
ticker = 'GOOG'

end_date = dt.date.today()
start_date = end_date - dt.timedelta(days=27)
end = end_date.strftime('%Y-%m-%d')
start = start_date.strftime('%Y-%m-%d')

data = yf.download(ticker, start=start, end=end)

# Select close column and reshape
data = data.loc[:, ['Close']]
print(data)

# normalize the dataset
scale = MinMaxScaler(feature_range=(0, 1))
data_sc = scale.fit_transform(data)

# Sent An search term
search_term = 'Google'


# Calls multivariate predictor
def multivariate_sa_predictor():

    # Gather data and analyse sentiment
    tweets = WebScraperTwitter.scrape_data(search_term)
    sentiments = SentAnVader.get_compound_sentiment(tweets)

    combined = __combine_data(sentiments, data_sc)
    train_x, train_y, test_x, test_y = __prepare_data(combined)

    input_shape = (train_x.shape[1], train_x.shape[2])

    return train_x, train_y, test_x, test_y, input_shape


# Combines data from sentimenta analysis and ticker
def __combine_data(sent_analysis, stocks):
    col_name = 'Close'

    # Correct date format
    stocks.index = stocks.index.date

    # Combine dataframes
    mv_df = pd.concat([sent_analysis, stocks], axis=1)

    # Replace NaN values
    for val, rows in mv_df[col_name].iteritems():
        if isnan(rows):
            mv_df.at[val, 'Close'] = 0

    return mv_df


# Prepares data for use in multivariate predictor
def __prepare_data(comb):
    # Prepare data for model
    pd.set_option('precision', 4)
    np.set_printoptions(suppress=True)
    combined_np = comb.to_numpy()

    # split into train and test
    train_size = int(len(combined_np) * train_percentage)
    train, test = combined_np[0:train_size, :], combined_np[train_size:len(combined_np), :]

    # Split data into x and y
    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]

    # Reshape data for Tensorflow [samples, timesteps, features]
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    print('After reshaping: ')
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    return train_x, train_y, test_x, test_y
