from datetime import datetime

from preprocessing.ticker_preprocessing import get_ticker_data
from traditional_analysis_strategies.analysis_strategies.MATrendFollowing import ema_trend, sma_trend

# Get data for ticker
ticker = 'GOOG'
date_start = datetime.strptime("2021-05-08", '%Y-%m-%d')
date_end = datetime.strptime("2022-05-08", '%Y-%m-%d')

train_percentage = 0.8

# Preprocessing
base_data = get_ticker_data(ticker, date_start, date_end)

final_df, algorithm_return, buy_hold_return, current_signal = ema_trend(base_data)

# ML Results
print("Trend Results")
print("Trend data: ", final_df)
print("Trend Test returns ", algorithm_return)
print("Buy and hold returns: ", buy_hold_return)
print("Current signal: ", current_signal)
