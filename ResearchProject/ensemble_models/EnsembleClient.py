import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

from ensemble_models import MVPredictor, AverageEnsemble
from ml_model.models.CnnFactory import CnnFactory
from ml_model.models.LstmFactory import LstmFactory
from ml_model.models.BiDirectionalFactory import BiDirectionalFactory
from ml_model.models.GruFactory import GruFactory
from ml_model.models.RnnFactory import RnnFactory
from post_processing import RegressionAccuracy

"""
Client for ensemble model combining sentiment compound scores and keras models for predictions. 
"""

# Scaler
scale = MinMaxScaler(feature_range=(0, 1))

np.random.seed(1)  # Fix for reproducible results
train_percentage = 0.8

# Get historical data for ticker
ticker = 'GOOG'
stock_ticker = yf.Ticker(ticker)
data = stock_ticker.history(period='Max')

# Select close column and reshape
data = data.loc[:, ['Close']]

# normalize the dataset
scale = MinMaxScaler(feature_range=(0, 1))
data_sc = scale.fit_transform(data)

# split into train and test
train_size = int(len(data_sc) * train_percentage)
test_size = len(data_sc) - train_size
train, test = data_sc[0:train_size, :], data_sc[train_size:len(data), :]

# Model variables
inputs = 25
input2 = 10
optimiser = 'adam'
loss = 'mse'
cnn_pooling = 'Average'  # allow average or global. Implement when doing main method.

# Model Types
factory1 = LstmFactory
factory2 = CnnFactory


def __split_x_y(dataset, timestep):
    x, y = [], []
    for i in range(len(dataset) - timestep - 1):
        x.append(dataset[i:(i + timestep), 0])
        y.append(dataset[i + timestep, 0])
    return np.array(x), np.array(y)


def create_mv():
    train_x, train_y, test_x, test_y, input_shape = MVPredictor.multivariate_sa_predictor()

    step = 1
    input_shape = (1, step)
    model = factory1.create_model(factory1, inputs, input_shape, optimiser, loss)

    # Fit model
    model.fit(train_x, train_y, epochs=1000, batch_size=240, verbose=1)

    # make predictions
    predictions = model.predict(test_x)

    print(predictions.shape)
    print('0 and 1:', predictions[0], predictions[1])
    print('All predicts: ', predictions)

    # invert predictions
    predictions = scale.inverse_transform(predictions)
    test_y = scale.inverse_transform([test_y])

    # calculate accuracy
    RegressionAccuracy.calc_accuracy(predictions[:, 0], test_y[0])


# def create_average_ensemble(np_output1: np.array, np_output2: np.array):
def create_average_ensemble():

    # split into x and y
    step = 200
    train_x, train_y = __split_x_y(train, step)
    test_x, test_y = __split_x_y(test, step)

    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

    input_shape = (1, step)
    model1 = factory1.create_model(factory1, inputs, input_shape, optimiser, loss)
    model2 = factory2.create_model(factory2, inputs, input_shape, optimiser, loss)

    # Fit models
    model1.fit(train_x, train_y, epochs=1000, batch_size=240, verbose=1)
    model2.fit(train_x, train_y, epochs=1000, batch_size=240, verbose=1)

    # make predictions
    predictions_mod1 = model1.predict(test_x)
    predictions_mod2 = model1.predict(test_x)

    average_predictions = AverageEnsemble.average_outputs(predictions_mod1, predictions_mod2)
    return average_predictions
