import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.mnist_classifier_interface import MnistClassifierInterface
import os

class CNNMnistClassifier(MnistClassifierInterface):
   def __init__(self, model_path=None):
       super().__init__()
       self.feature_extractor = nn.Sequential(
           # Block 1: 32C3 - 32C3 - 32C5S2
           nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
           nn.ReLU(),
           nn.BatchNorm2d(32),
           
           nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
           nn.ReLU(),
           nn.BatchNorm2d(32),
           nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),  
           nn.ReLU(),
           nn.BatchNorm2d(32),
           nn.Dropout(0.4),
           # Block 2: 64C3 - 64C3 - 64C5S2
           nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  
           nn.ReLU(),
           nn.BatchNorm2d(64),
           nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
           nn.ReLU(),
           nn.BatchNorm2d(64),
           nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),  
           nn.ReLU(),
           nn.BatchNorm2d(64),
           nn.Dropout(0.4),
       )
       self.classifier = nn.Sequential(
           nn.Flatten(),
           nn.Linear(64 * 7 * 7, 128),
           nn.ReLU(),
           nn.BatchNorm1d(128),
           nn.Dropout(0.4),
           nn.Linear(128, 10) 
       )
       self.model = nn.Sequential(self.feature_extractor, self.classifier)
       self.criterion = nn.CrossEntropyLoss()
       self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
       self.model_path = model_path or "models/cnn_mnist_model.pt"
       
       # Load model if exists
       if os.path.exists(self.model_path):
           self.model.load_state_dict(torch.load(self.model_path))
           print(f"Loaded model from {self.model_path}")
           
   def train(self, X_train, y_train, epochs=30, batch_size=64):
     
       if not isinstance(X_train, torch.Tensor):
           X_train = torch.tensor(X_train.to_numpy() if hasattr(X_train, "to_numpy") else np.array(X_train), dtype=torch.float32)
       if not isinstance(y_train, torch.Tensor):
           y_train = torch.tensor(y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.array(y_train), dtype=torch.long)
       # Reshape for CNN (batch_size, channels, height, width)
       X_train = X_train.view(-1, 1, 28, 28) / 255.0  # Normalize pixel values
       dataset = TensorDataset(X_train, y_train)
       dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
       self.model.train()
       for epoch in range(epochs):
           for X_batch, y_batch in dataloader:
               self.optimizer.zero_grad()
               outputs = self.model(X_batch)
               loss = self.criterion(outputs, y_batch)
               loss.backward()
               self.optimizer.step()
       
       # Save model after training
       torch.save(self.model.state_dict(), self.model_path)
       print(f"Model saved to {self.model_path}")
       
   def predict(self, X_test):
       if not isinstance(X_test, torch.Tensor):
           X_test = torch.tensor(X_test.to_numpy() if hasattr(X_test, "to_numpy") else np.array(X_test), dtype=torch.float32)
       X_test = X_test.view(-1, 1, 28, 28) / 255.0  # Normalize pixel values
       self.model.eval()
       with torch.no_grad():
           outputs = self.model(X_test)
           return torch.argmax(outputs, dim=1).numpy()  