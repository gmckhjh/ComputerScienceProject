from abc import ABC, abstractmethod

"""
Abstract Single Model class to serve as template ensuring uniformity across keras 1 layer models. 
"""


# Abstract single model class
class SingleModel(ABC):

    @abstractmethod
    def create_model(self, num_inputs, shape):
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
