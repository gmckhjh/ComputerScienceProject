from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from ml_model.abstract_models.TwoStackModel import TwoStackModel

"""
Concrete implementation of the SingleModel class - creates 2 stack long short-term memory model and compiles it 
based on the parameter values
"""


class LstmTwoStack(TwoStackModel):
    def __init__(self, num_inputs1, shape, num_inputs2, optimiser, loss):
        self.create_model(num_inputs1, shape, num_inputs2)
        self.compile_model(self.model, optimiser, loss)

    def create_model(self, num_inputs1, shape, num_inputs2):
        self.model = Sequential()
        self.model.add(LSTM(num_inputs1, return_sequences=True, input_shape=shape))
        self.model.add(Dropout(0.1))
        self.model.add(LSTM(num_inputs2))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(1))
        return self.model

    def compile_model(self, model, optimiser, loss):
        self.model.compile(
            optimizer=optimiser,
            loss=loss,
            metrics=['accuracy'])
