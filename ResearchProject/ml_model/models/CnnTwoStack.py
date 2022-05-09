from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, MaxPool1D, Flatten

from ml_model.abstract_models.TwoStackModel import TwoStackModel

"""
Concrete implementation of the SingleModel class - creates 2 stack Convolutional Neural Network model and 
compiles it based on the parameter values
"""


class CnnTwoStack(TwoStackModel):
    def __init__(self, num_inputs1, shape, num_inputs2, optimiser, loss):
        self.create_model(num_inputs1, shape, num_inputs2)
        self.compile_model(self.model, optimiser, loss)

    def create_model(self, num_inputs1, shape, num_inputs2):
        self.model = Sequential()
        self.model.add(Conv1D(filters=num_inputs1, kernel_size=1, strides=4, padding='Valid', use_bias=True,
                              activation='relu', kernel_initializer='VarianceScaling', input_shape=shape))
        self.model.add(MaxPool1D(1))
        self.model.add(Conv1D(filters=num_inputs1, kernel_size=1, strides=4, padding='Valid', use_bias=True,
                              activation='relu', kernel_initializer='VarianceScaling', input_shape=shape))
        self.model.add(MaxPool1D(1))
        self.model.add(Flatten())
        self.model.add(Dense(1))
        return self.model

    def compile_model(self, model, optimiser, loss):
        self.model.compile(
            optimizer=optimiser,
            loss=loss,
            metrics=['accuracy'])