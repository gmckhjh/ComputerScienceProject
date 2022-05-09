from datetime import datetime

import numpy as np
import pandas as pd
from plotly import graph_objs as go

from baseline.NaiveForecast import naive_prediction
from ml_model.models import BiDirectionalFactory, CnnFactory, GruFactory, LstmFactory, RnnFactory
from post_processing import RegressionAccuracy
from preprocessing.ticker_preprocessing import get_processed_ticker_data, get_ticker_data

"""
Client class to run tests and different models for keras to predict stock values. Tests against baseline model and
predicts values without feedback data from test frame in the form of the validation set. 
"""
np.random.seed(1)  # Fix for reproducible results
train_percentage = 0.8

# Get historical data for ticker
ticker = 'GOOG'
date_start = datetime.strptime("2021-05-08", '%Y-%m-%d')
date_end = datetime.strptime("2022-05-08", '%Y-%m-%d')

# Preprocessing
base_data = get_ticker_data(ticker, date_start, date_end)
train_x, train_y, test_x, test_y, test_data, validation, scale, time_step = get_processed_ticker_data(ticker,
                                                                                                      date_start,
                                                                                                      date_end,
                                                                                                      train_percentage,
                                                                                                      0)

# Set input shape and call Keras model
input_shape = (time_step, 1)

# Model Creation call
inputs = 25
input2 = 10
optimiser = 'adam'
loss = 'mse'
cnn_pooling = 'Average'  # allow average or global. Implement when doing main method.
factory = LstmFactory.LstmFactory
self = LstmFactory
model = factory.create_model(self, inputs, input_shape, optimiser, loss)

# Fit model
model.fit(train_x, train_y, epochs=10, batch_size=240, verbose=1)

# Convert values to float from numpy.int32
test_x = test_x.astype(float)
test_y = test_y.astype(float)

# Make test predictions
test_predictions = model.predict(test_x)

# Invert scaling
test_predictions = scale.inverse_transform(test_predictions)
test_y = scale.inverse_transform([test_y])


# Plot the future predictions - feed data same shape as for the model
def future_predictions(last_test):
    # Future Predictions
    prediction_days = 10
    preds = np.array([])

    # Get most recent Close value and prepare for prediction
    last_val = last_test
    last_val = np.reshape(last_val, (last_val.shape[0], last_val.shape[1], 1))
    p = last_val

    # Predict next close price based on previous
    for i in range(prediction_days):
        p = model.predict(p)
        p = np.reshape(p, (1, p.shape[0], p.shape[1]))
        preds = np.append(preds, p[len(p)-1:])

    # Reshape and scale predictions
    preds = np.reshape(preds, (preds.shape[0], 1))
    preds = scale.inverse_transform(preds)

    return preds


# Get Naive baseline model predictions
def get_naive_predictions():
    train_df = pd.DataFrame(train_y)
    train_df = train_df.T
    test_df = pd.DataFrame(test_y)
    test_df = test_df.T

    # Get Naive predictions
    naive_train, naive_test = naive_prediction(train_df, test_df)

    return naive_test


# Plot the prediction data against the original test data
def plot_data(test_df, naive_predictions, test_preds, future_preds):
    # Prepare data for plotting
    test_df = test_df.iloc[-25:]
    test_preds_pd = pd.DataFrame(test_preds)
    test_naive_pd = pd.DataFrame(naive_predictions)
    plot_df = test_df
    plot_df.reset_index(inplace=True)
    predictions_df = pd.DataFrame(future_preds)
    predictions_df.index += 15

    # Combine in one dataframe
    plot_df['Test Predictions'] = test_preds_pd
    plot_df['Naive Predictions'] = test_naive_pd
    plot_df['Validation Predictions'] = predictions_df

    # Plot with plotly
    fig = go.Figure()
    fig.add_trace(go.Line(x=plot_df['Date'], y=plot_df['Close'],
                          name='Base ticker data'))
    fig.add_trace(go.Line(x=plot_df['Date'], y=plot_df['Naive Predictions'],
                          name='Naive prediction against test data'))
    fig.add_trace(go.Line(x=plot_df['Date'], y=plot_df['Test Predictions'],
                          name='Model Prediction against test data'))
    fig.add_trace(go.Line(x=plot_df['Date'], y=plot_df['Validation Predictions'],
                          name='Future validation predictions'))
    fig.layout.update(title_text='Performance of naive model, ml model, and future predictions of ml model ',
                      xaxis_rangeslider_visible=True)

    fig.show()


# Calculate future predictions simulation on validation data
final_test_data = test_y
data_to_add = train_y[len(train_y)-1:]
final_test_data = np.insert(final_test_data, 0, data_to_add)
final_test_data = np.reshape(final_test_data, (1, final_test_data.shape[0]))
predictions = future_predictions(final_test_data)

# Invert train
train_y = scale.inverse_transform([train_y])

# Get predictions
naive_test_results = get_naive_predictions()

# Calculate accuracies
model_metrics = RegressionAccuracy.calc_accuracy(test_predictions[:, 0], test_y[0])
naive_accuracy = RegressionAccuracy.calc_accuracy(naive_test_results.iloc[1:, 0], test_y[0, 1:])
future_metrics = RegressionAccuracy.calc_accuracy(predictions, validation)

plot_data(test_data, naive_test_results, test_predictions, predictions)
