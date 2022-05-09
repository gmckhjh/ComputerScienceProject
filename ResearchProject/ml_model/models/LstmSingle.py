from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

from ml_model.abstract_models.SingleModel import SingleModel

"""
Concrete implementation of the SingleModel class - creates a single layer long short-term memory model and compiles it 
based on the parameter values
"""


class LstmSingle(SingleModel):
    def __init__(self, num_inputs, shape, optimiser, loss):
        self.create_model(num_inputs, shape)
        self.compile_model(self.model, optimiser, loss)

    def create_model(self, num_inputs, shape) -> Sequential:
        self.model = Sequential()
        self.model.add(LSTM(num_inputs, input_shape=shape))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(1))
        return self.model

    def compile_model(self, model, optimiser, loss) -> Sequential:
        self.model.compile(
            optimizer=optimiser,
            loss=loss,
            metrics=['accuracy'])
