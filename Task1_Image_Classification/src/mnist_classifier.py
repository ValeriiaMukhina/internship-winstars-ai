import numpy as np
from src.mnist_classifier_interface import MnistClassifierInterface
from src.models.rf_classifier import RandomForestMnistClassifier
from src.models.nn_classifier import FeedForwardMnistClassifier
from src.models.cnn_classifier import CNNMnistClassifier



# Main Classifier Wrapper
class MnistClassifier:
    def __init__(self, algorithm):
        if algorithm == "rf":
            self.classifier = RandomForestMnistClassifier()
        elif algorithm == "nn":
            self.classifier = FeedForwardMnistClassifier()
        elif algorithm == "cnn":
            self.classifier = CNNMnistClassifier()
        else:
            raise ValueError("Invalid algorithm choice")
    
    def train(self, X_train, y_train):
        self.classifier.train(X_train, y_train)
    
    def predict(self, X_test):
        return self.classifier.predict(X_test)
