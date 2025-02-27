import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import random

def get_animals10(data_dir='data/animals10/raw-img', batch_size=32):
  
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # Map Italian class names to English
    class_map = {
        'cane': 'dog', 
        'cavallo': 'horse', 
        'elefante': 'elephant', 
        'farfalla': 'butterfly', 
        'gallina': 'chicken', 
        'gatto': 'cat',
        'mucca': 'cow', 
        'pecora': 'sheep', 
        'scoiattolo': 'squirrel', 
        'ragno': 'spider'
    }
    
    # Get indices for each class
    class_indices = {}
    for idx, (_, class_idx) in enumerate(dataset.samples):
        if class_idx not in class_indices:
            class_indices[class_idx] = []
        class_indices[class_idx].append(idx)
    
    # Create stratified train/val/test splits
    train_indices = []
    val_indices = []
    test_indices = []
    
    for class_idx, indices in class_indices.items():
        # Shuffle indices for this class
        random.seed(42)
        random.shuffle(indices)
        
        # Split indices using the same ratios
        n_total = len(indices)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train+n_val])
        test_indices.extend(indices[n_train+n_val:])
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    classes = [class_map[c] for c in dataset.classes]
    return train_loader, val_loader, test_loader, classes