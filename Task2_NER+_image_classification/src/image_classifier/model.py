import torch
import torch.nn as nn
from torchvision import models

class VGG16AnimalClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Load pretrained VGG16 model without the classifier part
        # Add SSL certificate workaround for macOS
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features  
        
    
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 256),  # VGG16 outputs 512 feature maps of size 7x7 when input is 224x224
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x