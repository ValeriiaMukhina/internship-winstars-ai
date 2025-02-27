import torch
from torchvision import transforms
from PIL import Image
import json
import os
import argparse

from src.image_classifier.model import VGG16AnimalClassifier

def inference(image_path, model_path='./models/animal_classifier.pth', classes_path=None, classes=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load class names if provided as a path
    if classes is None and classes_path is not None:
        with open(classes_path, 'r') as f:
            classes = json.load(f)
    
    if classes is None:
        classes = [
        "dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "squirrel", "spider"
        ]
    
    model = VGG16AnimalClassifier(num_classes=len(classes)).to(device)
    
    # Load the saved model
    try:
        # First, try to load as a complete model
        checkpoint = torch.load(model_path, map_location=device)
        if hasattr(checkpoint, 'eval'):  # Check if it's a model
            model = checkpoint
            print("Loaded complete model")
        else:
            # If it's a state_dict or a checkpoint dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                # Extract metadata if available
                epoch = checkpoint.get('epoch', None)
                val_acc = checkpoint.get('val_acc', None)
                if epoch is not None and val_acc is not None:
                    print(f"Loaded model from epoch {epoch+1} with validation accuracy: {val_acc:.2f}%")
            else:
                # It's likely just the state_dict
                model.load_state_dict(checkpoint)
                print("Loaded model state dictionary")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Set model to evaluation mode
    model.eval() 
    
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')  # Ensure 3 channels
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities, predicted_idx = torch.max(outputs, 1)
        
    predicted_class = classes[predicted_idx.item()]
    confidence = probabilities.item() * 100
    
    # Get top-3 predictions
    probs, indices = torch.topk(outputs, 3, dim=1)
    top3_classes = [classes[idx] for idx in indices[0].tolist()]
    top3_probs = probs[0].tolist()
    
    results = {
        'class': predicted_class,
        'confidence': confidence,
        'class_index': predicted_idx.item(),
        'top3': [
            {'class': cls, 'confidence': prob * 100} 
            for cls, prob in zip(top3_classes, top3_probs)
        ]
    }
    
    return results