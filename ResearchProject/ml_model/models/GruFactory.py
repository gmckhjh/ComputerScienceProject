from ml_model.abstract_models.ModelFactory import ModelFactory
from ml_model.models.GruSingle import GruSingle
from ml_model.models.GruTwoStack import GruTwoStack
from ml_model.abstract_models import TwoStackModel, SingleModel

"""
Concrete implementation of the ModelFactory class - returns creation of 1 and 2 layer Gated Recurrent Unit models based 
on parameter values.
"""


class GruFactory(ModelFactory):
    def create_model(self, num_inputs, shape, loss, optimiser) -> SingleModel:
        instance = GruSingle(num_inputs, shape, loss, optimiser)
        return instance.model

    def create_two_stack_model(self, layer_inputs, shape, layer_inputs2, loss, optimiser) -> TwoStackModel:
        instance = GruTwoStack(layer_inputs, shape, layer_inputs2, loss, optimiser)
        return instance.model
