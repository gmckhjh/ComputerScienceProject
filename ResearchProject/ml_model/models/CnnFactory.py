from ml_model.abstract_models.ModelFactory import ModelFactory
from ml_model.models.CnnSingle import CnnSingle
from ml_model.models.CnnTwoStack import CnnTwoStack
from ml_model.abstract_models import TwoStackModel, SingleModel

"""
Concrete implementation of the ModelFactory class - returns creation of 1 and 2 layer Convolutional Neural Network 
models based on parameter values.
"""


class CnnFactory(ModelFactory):
    def create_model(self, num_inputs, shape, loss, optimiser) -> SingleModel:
        instance = CnnSingle(num_inputs, shape, loss, optimiser)
        return instance.model

    def create_two_stack_model(self, layer_inputs, shape, layer_inputs2, loss, optimiser) -> TwoStackModel:
        instance = CnnTwoStack(layer_inputs, shape, layer_inputs2, loss, optimiser)
        return instance.model
