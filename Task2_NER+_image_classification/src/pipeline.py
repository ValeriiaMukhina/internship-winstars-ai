import argparse
import json
import os
import torch
from torchvision import transforms
from PIL import Image

from src.ner.inference import run_inference
from src.image_classifier.model import VGG16AnimalClassifier
from src.image_classifier.inference import inference

# Predefined paths for models
DEFAULT_IMG_MODEL_PATH = "models/animal_classifier.pth"
DEFAULT_NER_MODEL_PATH = "models/animal_ner_model"

# Predefined set of animal classes
ANIMAL_CLASSES = [
    "dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "squirrel", "spider"
]

class AnimalDetectionPipeline:
    def __init__(self, img_model_path=None, ner_model_path=None):
        """
        Initialize the pipeline for animal detection
        
        Args:
            img_model_path (str, optional): Path to the image classification model. 
                                           If None, uses the default path.
            ner_model_path (str, optional): Path to the NER model.
                                           If None, uses the default path.
        """
        # Use default paths if not provided
        self.img_model_path = img_model_path if img_model_path else DEFAULT_IMG_MODEL_PATH
        self.ner_model_path = ner_model_path if ner_model_path else DEFAULT_NER_MODEL_PATH
        
        print(f"Using image model: {self.img_model_path}")
        print(f"Using NER model: {self.ner_model_path}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load classes
        self.classes = ANIMAL_CLASSES
        print(f"Using animal classes: {self.classes}")
        
        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def extract_animals_from_text(self, text):
        """
        Extract animal names from text using the NER model
        
        Args:
            text (str): Input text message
            
        Returns:
            list: List of extracted animal names
        """
        try:
            # Use the NER model's run_inference function
            entities_verbose = run_inference(
                text=text,
                model_path=self.ner_model_path,
                output_format="verbose"
            )
            
            # Extract unique animal entities
            animals = entities_verbose.get("unique_entities", [])
            
            # If no animals were found, try looking for our known classes in the text
            if not animals:
                text_lower = text.lower()
                for animal in self.classes:
                    animal_lower = animal.lower()
                    # Check if the animal name appears as a whole word
                    if f" {animal_lower} " in f" {text_lower} ":
                        animals.append(animal_lower)
            
            return animals
        except Exception as e:
            print(f"Error extracting animals from text: {e}")
            # Fall back to a simple word matching approach
            text_lower = text.lower().split()
            return [word for word in text_lower if word in self.classes]
    
    def _classify_image_locally(self, image_path):
        """
        Classify an image using the model directly
        Used as a fallback if the imported classify_image function fails
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            dict: Classification results
        """
        try:
            # Load model
            checkpoint = torch.load(self.img_model_path, map_location=self.device)
            model = VGG16AnimalClassifier(num_classes=len(self.classes)).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities, predicted_idx = torch.max(outputs, 1)
            
            predicted_class = self.classes[predicted_idx.item()]
            confidence = probabilities.item() * 100
            
            # Get top-3 predictions
            probs, indices = torch.topk(outputs, min(3, len(self.classes)), dim=1)
            top3_classes = [self.classes[idx] for idx in indices[0].tolist()]
            top3_probs = probs[0].tolist()
            
            return {
                'class': predicted_class,
                'confidence': confidence,
                'top3': [
                    {'class': cls, 'confidence': prob * 100} 
                    for cls, prob in zip(top3_classes, top3_probs)
                ]
            }
        except Exception as e:
            print(f"Error in local image classification: {e}")
            # Return a default response with low confidence
            return {
                'class': self.classes[0],
                'confidence': 10.0,
                'top3': [{'class': cls, 'confidence': 10.0} for cls in self.classes[:3]]
            }
    
    def process(self, text, image_path):
        """
        Process the text and image, determining if the animal mentioned in the text
        is present in the image
        
        Args:
            text (str): Input text message
            image_path (str): Path to the image
            
        Returns:
            bool: True if the animal mentioned in the text matches the image, False otherwise
        """
        # Extract animal names from the text
        animals_in_text = self.extract_animals_from_text(text)
        
        if not animals_in_text:
            print("No animal was mentioned in the text.")
            return False
        
        # Try to classify the image using the imported function
        try:
            image_result = inference(
                image_path=image_path,
                model_path=self.img_model_path,
                classes=self.classes
            )
        except Exception as e:
            print(f"Error using imported classify_image: {e}")
            print("Falling back to local classification method")
            image_result = self._classify_image_locally(image_path)
        
        # Extract the predicted animal and other info from the classification result
        predicted_animal = image_result['class'].lower()
        
        print(f"Text mentions animals: {animals_in_text}")
        print(f"Image predicted as: {predicted_animal} (Confidence: {image_result['confidence']:.2f}%)")
        
        # Check if any animal mentioned in the text matches the image prediction
        # First check the main prediction with flexible matching
        match_found = False
        for animal in animals_in_text:
         # Strip trailing 's' for plurals
            singular_form = animal[:-1] if animal.endswith('s') else animal
            if predicted_animal == singular_form or predicted_animal + 's' == animal:
                match_found = True
                print(f"Match found! Text mentions '{animal}' and image contains '{predicted_animal}'")
                break
    

            # If no match with the top prediction, check the top-3 predictions
        if not match_found and 'top3' in image_result:
            for pred in image_result['top3'][1:]:  # Skip the first one as we already checked it
                pred_class = pred['class'].lower()
                if pred_class in animals_in_text and pred['confidence'] > 20.0:  # Confidence threshold
                    match_found = True
                    print(f"Match found in top-3 predictions! Text mentions {pred_class} with {pred['confidence']:.2f}% confidence")
                    break
        
        return match_found