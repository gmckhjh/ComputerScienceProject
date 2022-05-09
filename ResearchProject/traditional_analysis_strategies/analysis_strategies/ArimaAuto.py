from datetime import datetime

import pmdarima as pm

from post_processing import RegressionAccuracy
from preprocessing.ticker_preprocessing import get_ticker_data

# Get data for ticker
ticker = 'GOOG'
date_start = datetime.strptime("2021-05-08", '%Y-%m-%d')
date_end = datetime.strptime("2022-05-08", '%Y-%m-%d')

train_percentage = 0.8

# Preprocessing
base_data = get_ticker_data(ticker, date_start, date_end)

data = base_data.loc[:, ['Close']]

train_data, test_data = data[0:int(len(data) * train_percentage)], data[int(len(data) * train_percentage):]
train_data = train_data['Close'].values
test_data = test_data['Close'].values


# Run the ARIMA model
def run_arima(model_data):
    model = pm.auto_arima(model_data, start_p=1, start_q=1,
                          test='adf',
                          max_p=3, max_q=3,
                          m=1,
                          d=None,
                          seasonal=False,
                          start_P=0,
                          D=0,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)

    print(model.summary())

    # Forecast
    periods = len(test_data)
    model_predictions = model.predict(periods)

    return model_predictions


predictions = run_arima(train_data)

# Calculate accuracy
accuracy = RegressionAccuracy.calc_accuracy(predictions, test_data)
print(accuracy)
