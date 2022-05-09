from abc import ABC, abstractmethod

"""
Abstract Two Stack Model class to serve as template ensuring uniformity across keras 2 layer models.
"""


# Abstract two stack model class
class TwoStackModel(ABC):

    @abstractmethod
    def create_model(self, num_inputs1, shape, num_inputs2):
        pass

    @abstractmethod
    def compile_model(self, model, optimiser, loss):
        pass

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        self._model = m
