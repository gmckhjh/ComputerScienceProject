from datetime import datetime

from ml_trend_predictions.trend_strategies.RandomForestStrategy import forest_predictor
from ml_trend_predictions.trend_strategies.MLStrategy import ml_trading_strategy
from preprocessing.ticker_preprocessing import get_ticker_data

"""
Client to run machine learning trend predictions. 
"""
# Get data for ticker
ticker = 'GOOG'
date_start = datetime.strptime("2021-05-08", '%Y-%m-%d')
date_end = datetime.strptime("2022-05-08", '%Y-%m-%d')

train_percentage = 0.8

# Preprocessing
base_data = get_ticker_data(ticker, date_start, date_end)

# Call forest
forest_df, forest_test_return, forest_test_buy_hold_return, forest_current_signal = forest_predictor(base_data,
                                                                                                     train_percentage)

# Call ML
ml_df, ml_test_return, ml_test_buy_hold_return, ml_current_signal = ml_trading_strategy(base_data, train_percentage)

# Forest results
print("Forest Results")
print("Forest data: ", forest_df)
print("Forest Test returns ", forest_test_return)
print("Buy and hold returns: ", forest_test_buy_hold_return)
print("Current signal: ", forest_current_signal)

# ML Results
print("ML Results")
print("ML data: ", ml_df)
print("ML Test returns ", ml_test_return)
print("Buy and hold returns: ", ml_test_buy_hold_return)
print("Current signal: ", ml_current_signal)
