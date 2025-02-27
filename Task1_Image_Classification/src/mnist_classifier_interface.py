from abc import ABC, abstractmethod
import numpy as np

class MnistClassifierInterface(ABC):
 
    @abstractmethod
    def train(self, X_train, y_train):
        pass
    
    @abstractmethod
    def predict(self, X_test):
        pass