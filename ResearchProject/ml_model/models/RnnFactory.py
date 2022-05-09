from ml_model.abstract_models.ModelFactory import ModelFactory
from ml_model.models.RnnSingle import RnnSingle
from ml_model.models.RnnTwoStack import RnnTwoStack
from ml_model.abstract_models import TwoStackModel, SingleModel

"""
Concrete implementation of the ModelFactory class - returns creation of 1 and 2 layer Recurrent Neural Network models 
based on parameter values.
"""


class RnnFactory(ModelFactory):
    def create_model(self, num_inputs, shape, loss, optimiser) -> SingleModel:
        instance = RnnSingle(num_inputs, shape, loss, optimiser)
        return instance.model

    def create_two_stack_model(self, layer_inputs, shape, layer_inputs2, loss, optimiser) -> TwoStackModel:
        instance = RnnTwoStack(layer_inputs, shape, layer_inputs2, loss, optimiser)
        return instance.model
