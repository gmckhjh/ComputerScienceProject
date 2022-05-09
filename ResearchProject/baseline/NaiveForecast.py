import pandas as pd

"""
A baseline forecast that follows a random walk to be used as a comparison for the accuracy of other models on a 
dataset (stock over a period of time in this case). A model that doesn't exceed this naive model for accuracy on the 
test split of a dataset can be considered to be worse than randomly predicting stock movements. 
"""


# Predict the naive forecast for a dataset
# Achieved by using the shift function to predict the last value
def naive_prediction(train: pd.DataFrame, test: pd.DataFrame):

    train_predictions = train.shift(1)
    test_predictions = test.shift(1)

    return train_predictions, test_predictions
