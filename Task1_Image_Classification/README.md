# MNIST Classifier Project

This project implements different classification models for the MNIST dataset using object-oriented programming principles. The implementation features three different classifiers (Random Forest, Neural Network, and CNN) behind a unified interface.


## Key Features

- **Interface Design Pattern**: All classifiers implement the same interface, providing consistent train and predict methods.
- **Multiple Algorithms**: Three different implementation types:
  - Random Forest: Scikit-learn-based classifier
  - Neural Network: PyTorch-based feed-forward neural network
  - CNN: PyTorch-based convolutional neural network
- **Unified API**: The `MnistClassifier` wrapper class hides implementation details and provides a consistent API.


## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/ValeriiaMukhina/internship-winstars-ai.git
cd Task1_Image_Classification/
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Or run the demo notebook:
```bash
jupyter notebook mnist_demo.ipynb
```

## Usage

```python
from mnist_classifier import MnistClassifier
from utils.data_loader import load_mnist_data

# Load data
X_train, X_test, y_train, y_test = load_mnist_data()

# Create classifier (options: 'rf', 'nn', 'cnn')
classifier = MnistClassifier(algorithm='cnn')

# Train
classifier.train(X_train, y_train)

# Predict
predictions = classifier.predict(X_test)
```

## Implementation Details

### Interface

The `MnistClassifierInterface` defines two abstract methods:
- `train(X_train, y_train)`: Trains the model on provided data
- `predict(X)`: Makes predictions on provided data

### Models

1. **Random Forest**: Uses scikit-learn's RandomForestClassifier to provide a fast, tree-based approach.
2. **Neural Network**: Implements a simple feed-forward network using PyTorch with one hidden layer.
3. **CNN**: 

The CNNs in this kernel follow LeNet5's design with the following improvements:

- Two stacked 3x3 filters replace the single 5x5 filters. These become nonlinear 5x5 convolutions 
- A convolution with stride 2 replaces pooling layers. These become learnable pooling layers. 
- ReLU activation replaces sigmoid. 
- Batch normalization is added 
- Dropout is added More feature maps (channels) are added
Given Architecture Explanation 
- Input size: 28x28 (which corresponds to 784 flattened, but we keep it as a 2D tensor) 
- First block: 32C3-32C3-32C5S2 32 Conv filters (3x3) 32 Conv filters (3x3) 32 Conv filters (5x5, stride=2, same padding) 
- Second block: 64C3-64C3-64C5S2 64 Conv filters (3x3) 64 Conv filters (3x3) 64 Conv filters (5x5, stride=2, same padding) 
- Fully connected layers: 128 - 10 
### Wrapper Class

The `MnistClassifier` class instantiates the appropriate implementation based on the algorithm name provided at initialization. It exposes the same train and predict methods as defined in the interface.

## Performance Comparison

| Algorithm | Training Time | Prediction Time | Accuracy |
|-----------|---------------|-----------------|----------|
| RF        | Medium        | Fast            | ~94%     |
| NN        | Slow          | Medium          | ~95%     |
| CNN       | Slowest       | Medium          | ~99%     |
