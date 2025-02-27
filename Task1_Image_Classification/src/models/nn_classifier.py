import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.mnist_classifier_interface import MnistClassifierInterface
import os 


class FeedForwardMnistClassifier(MnistClassifierInterface):
    def __init__(self, input_size=28*28, hidden_units=256, dropout=0.1, num_labels=10, model_path=None):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 12),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(12, 10),
            nn.Softmax(dim=1)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.model_path = model_path or "models/feedforward_mnist_model.pt"
        
        # Load model if exists
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print(f"Loaded model from {self.model_path}")
      
    def train(self, X_train, y_train, epochs=20, batch_size=64):
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train.to_numpy() if hasattr(X_train, "to_numpy") else np.array(X_train), dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.array(y_train), dtype=torch.long)
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        for _ in range(epochs):
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
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            return torch.argmax(outputs, dim=1).numpy()