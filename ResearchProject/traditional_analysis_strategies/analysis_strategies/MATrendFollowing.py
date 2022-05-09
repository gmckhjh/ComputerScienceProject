import pandas
import pandas as pd
import numpy as np

"""
Moving Average Trend tracking algorithms for slow and exponentially-weighted. Return the results achieved using the 
signals against the buy-and-hold strategy with the most recent signal. 
"""


# Get final data and accuracy results of SMA (Simple Moving Average) trend predictions against buy and hold, including
# the current signal
def sma_trend(df: pandas.DataFrame):
    # Calculate SMA
    df['Slow MA'] = df['Close'].rolling(30).mean()
    df['Fast MA'] = df['Close'].rolling(10).mean()

    final_df, algorithm_return, buy_hold_return, current_signal = __trend_prediction(df)
    return final_df, algorithm_return, buy_hold_return, current_signal


# Get final data and accuracy results of EWMA (Exponentially Weighted Moving Average) trend predictions against buy and
# hold, including the current signal
def ema_trend(df: pandas.DataFrame):
    df['Slow MA'] = df['Close'].ewm(alpha=0.2, adjust=False).mean()
    df['Fast MA'] = df['Close'].ewm(alpha=0.05, adjust=False).mean()

    final_df, algorithm_return, buy_hold_return, current_signal = __trend_prediction(df)
    return final_df, algorithm_return, buy_hold_return, current_signal


# Set buy or sell signals based on when the fast and slow MAs cross. When fast MA is greater than the slow a buy signal
# is set and vice versa.
def __trend_prediction(df: pandas.DataFrame):
    # Get log return prices
    df['Log Return'] = np.log(df['Close']).diff()
    df['Log Return'] = df['Log Return'].shift(-1)

    # Set trading signal
    df['Signal'] = np.where(df['Fast MA'] >= df['Slow MA'], 1, 0)

    # Determine effectiveness against buy and hold strategy
    df['Is Invested'] = df.apply(__assign_is_invested, axis=1)

    # Get stock return percentages
    algorithm_return, buy_hold_return = __compare_buy_hold(df)

    # Get final signal
    current_signal = df['Signal'].iat[-1]

    return df, algorithm_return, buy_hold_return, current_signal


# Compare the algorithm returns with the buy and hold returns
def __compare_buy_hold(df: pd.DataFrame):
    # Calculate the return whenever signals dictate investment
    df['Algo Log Return'] = df['Is Invested'] * df['Log Return']
    algorithm_return = df['Algo Log Return'].sum()

    # Calculate total return for buy and hold strategy
    buy_hold_return = df['Log Return'].sum()

    return algorithm_return, buy_hold_return


is_invested = False


# Determine if would be invested or not based on return values.
def __assign_is_invested(row):
    global is_invested
    if is_invested and row['Signal'] == 0:
        is_invested = False
    if not is_invested and row['Signal'] == 1:
        is_invested = True

    return is_invested
