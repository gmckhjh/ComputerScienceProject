from abc import ABC, abstractmethod

"""
Abstract Factory pattern model class to serve as template for creating single or double layer models of a selected type
"""


# Class to create keras models
class ModelFactory(ABC):

    # Create a single layer model
    @abstractmethod
    def create_model(self, num_inputs, shape, loss, optimiser):
        pass

    # Create a double layer model
    @abstractmethod
    def create_two_stack_model(self, num_inputs1, shape, num_inputs2, loss, optimiser):
        pass
