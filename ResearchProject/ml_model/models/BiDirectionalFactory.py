from ml_model.abstract_models.ModelFactory import ModelFactory
from ml_model.models.BiDirSingle import BiDirSingle
from ml_model.models.BiDirTwoStack import BiDirTwoStack
from ml_model.abstract_models import TwoStackModel, SingleModel

"""
Concrete implementation of the ModelFactory class - returns creation of 1 and 2 layer BiDirectional models based on 
parameter values.
"""


class BiDirectionalFactory(ModelFactory):
    def create_model(self, num_inputs, shape, loss, optimiser) -> SingleModel:
        instance = BiDirSingle(num_inputs, shape, loss, optimiser)
        return instance.model

    def create_two_stack_model(self, layer_inputs, shape, layer_inputs2, loss, optimiser) -> TwoStackModel:
        instance = BiDirTwoStack(layer_inputs, shape, layer_inputs2, loss, optimiser)
        return instance.model
