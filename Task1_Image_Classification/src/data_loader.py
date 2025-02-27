import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def load_mnist_data(test_size=0.2, random_state=42):

    mnist = fetch_openml("mnist_784", version=1)
    X = mnist.data 
    y = mnist.target.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   
    
    return X_train, X_test, y_train, y_test