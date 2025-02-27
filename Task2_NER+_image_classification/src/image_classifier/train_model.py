import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import os
import json

from src.image_classifier.model import VGG16AnimalClassifier


def train_model(train_loader, val_loader, classes, num_epochs=10, 
                save_dir='models', model_name='vgg16_animal_classifier'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGG16AnimalClassifier(num_classes=len(classes)).to(device)
    
    # Set up parameters for SGD as specified
    learning_rate = 0.001
    decay_rate = learning_rate / num_epochs
    momentum = 0.9
    
    # SGD optimizer with momentum and weight decay
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, 
                          weight_decay=decay_rate, nesterov=False)
    
    # Binary cross entropy loss for multi-class classification
    criterion = nn.BCELoss()  # Using Binary Cross Entropy as specified
    
    best_val_acc = 0.0
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}.pth")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Convert labels to one-hot encoding for BCE loss
            labels_one_hot = torch.zeros(labels.size(0), len(classes), device=device)
            labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_one_hot)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Convert labels to one-hot for BCE loss
                labels_one_hot = torch.zeros(labels.size(0), len(classes), device=device)
                labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels_one_hot)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100. * val_correct / val_total
        
        # Adjust learning rate according to decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate * (1 / (1 + decay_rate * epoch))
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {val_acc:.2f}%. Saving model...")
            
            # Save the entire model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, save_path)
            
            # Save class mapping
            with open(os.path.join(save_dir, "classes.json"), 'w') as f:
                json.dump(classes, f)
    
    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save final model if different from best model
    if best_val_acc < val_acc:
        final_save_path = os.path.join(save_dir, f"final_{model_name}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, final_save_path)
        print(f"Final model saved to {final_save_path}")
    
    return best_val_acc