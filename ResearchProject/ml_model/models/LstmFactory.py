from ml_model.abstract_models.ModelFactory import ModelFactory
from ml_model.models.LstmSingle import LstmSingle
from ml_model.models.LstmTwoStack import LstmTwoStack
from ml_model.abstract_models import TwoStackModel, SingleModel

"""
Concrete implementation of the ModelFactory class - returns creation of 1 and 2 layer Long short-term memory models 
based on parameter values.
"""


class LstmFactory(ModelFactory):
    def create_model(self, num_inputs, shape, loss, optimiser) -> SingleModel:
        instance = LstmSingle(num_inputs, shape, loss, optimiser)
        return instance.model

    def create_two_stack_model(self, layer_inputs, shape, layer_inputs2,  loss, optimiser) -> TwoStackModel:
        instance = LstmTwoStack(layer_inputs, shape, layer_inputs2, loss, optimiser)
        return instance.model
