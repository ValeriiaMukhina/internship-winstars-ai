import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.mnist_classifier_interface import MnistClassifierInterface
import os
import joblib

class RandomForestMnistClassifier(MnistClassifierInterface):
  def __init__(self, model_path=None):
      self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
      self.model_path = model_path or "models/random_forest_mnist.joblib"
      
      # Try to load model if exists
      if os.path.exists(self.model_path):
          try:
              self.model = joblib.load(self.model_path)
              print(f"Loaded model from {self.model_path}")
          except Exception as e:
              print(f"Error loading model: {e}")
              print("Using new model instead")
  
  def train(self, X_train, y_train):
      self.model.fit(X_train, y_train)
      
      # Create directory if it doesn't exist
      os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
      
      # Save/overwrite model
      joblib.dump(self.model, self.model_path)
      print(f"Model saved to {self.model_path}")
  
  def predict(self, X_test):
      return self.model.predict(X_test)