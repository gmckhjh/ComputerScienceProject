from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import math
import numpy as np


def calc_accuracy(prediction, original):
    df = {'rmse': __calc_rmse(prediction, original),
          'mape': __calc_mape(prediction, original),
          'min_max': __calc_min_max(prediction, original)
          }

    return df


def __calc_rmse(prediction, original):
    rmse = math.sqrt(mean_squared_error(original, prediction))
    return rmse


def __calc_mape(prediction, original):
    mape = mean_absolute_percentage_error(original, prediction)
    return mape


def __calc_min_max(prediction, original):
    min = np.amin([prediction,
                   original], axis=1)
    max = np.amax([prediction,
                   original], axis=1)
    minmax_error = 1 - np.mean(min / max)  # Min-Max Error
    return minmax_error
