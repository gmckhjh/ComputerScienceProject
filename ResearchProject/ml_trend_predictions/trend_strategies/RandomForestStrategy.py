import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def forest_predictor(df: pd.DataFrame, train_percentage: float):

    df_returns = pd.DataFrame()
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='any', inplace=True)

    # Get log values for each column
    for name in df.columns:
        df_returns[name] = np.log(df[name]).diff()

    df_returns['Log Return'] = df_returns['Close'].shift(-1)

    # split into train and test
    train_size = int(len(df_returns) * train_percentage)
    train, test = df_returns[1:train_size], df_returns[train_size:-1]

    x_cols = ['Log Return']

    # Split into x and y
    train_x = train[x_cols]
    train_y = train['Close']
    test_x = test[x_cols]
    test_y = test['Close']

    # Create and fit model
    model = RandomForestClassifier(random_state=0)
    train_c = (train_y > 0)
    test_c = (test_y > 0)
    model.fit(train_x, train_c)

    # Predict direction of stock
    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)

    df_returns['Is Invested'] = 0
    df_returns.loc[1:train_size, 'Is Invested'] = train_predict
    df_returns.loc[train_size:-1, 'Is Invested'] = test_predict

    # Get returns
    df_returns['Algo Returns'] = df_returns['Is Invested'] * df_returns['Close']
    test_return = df_returns.iloc[-train_size:-1]['Algo Returns'].sum()
    test_buy_hold_return = test_y.sum()
    current_signal = df_returns['Is Invested'].iloc[-2]

    return df_returns, test_return, test_buy_hold_return, current_signal
