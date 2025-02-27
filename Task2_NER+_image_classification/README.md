# Animal Detection Pipeline

This project implements a machine learning pipeline that combines Named Entity Recognition (NER) and Image Classification to verify if an animal mentioned in a text is present in an image.

## Task Description

The pipeline consists of two models responsible for different tasks:
1. **NER Model**: Extracts animal names from text messages
2. **Image Classification Model**: Identifies animals in images

The main goal is to understand what the user is asking (NLP) and check if they are correct (Computer Vision).


## Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the Animals-10 dataset (or use your own dataset)
If you want to train the model for Image Classification part, download dataset Animals10 from 
https://drive.google.com/drive/folders/1BvGcyRhDOA4UyFLfqQhu5pA0vKHq8CAU?usp=sharing
and put in Task2_NER+_image_classification/data/

4. If you want to run inference, please download ready models from 
https://drive.google.com/drive/folders/1BvGcyRhDOA4UyFLfqQhu5pA0vKHq8CAU?usp=sharing
and put them in Task2_NER+_image_classification/models/

## Model Architecture

The image classification model uses a VGG16 architecture pretrained on ImageNet with a custom classifier:
- Base model: VGG16 (pretrained)
- Custom classifier:
  - Flatten layer
  - Dense layer with 256 units and ReLU activation
  - Output layer with 10 units and softmax activation
The model is trained on the Animals-10 dataset, which contains 10 classes of animals:
- Dog
- Horse
- Elephant
- Butterfly
- Chicken
- Cat
- Cow
- Sheep
- Squirrel
- Spider

The NER component uses fine-tuned BERT model (bert-base-uncased) adapted for token classification
- Binary sequence labeling (0 for non-animal tokens, 1 for animal tokens)
- Input: Raw text, which is tokenized and processed to match the model's requirements
- Output: List of extracted animal entities found in the text
The animal NER component uses a custom synthetic dataset specifically designed for training models to recognize animal mentions in text. 
This approach allows for controlled data generation with perfect ground truth labeling, ensuring high-quality training data without the need for manual annotation.
- Targeted Entities: The dataset focuses on 10 common animal types: dog, horse, elephant, butterfly, chicken, cat, cow, sheep, squirrel, and spider
- Template-Based Generation: Uses a combination of natural language templates to create diverse sentence structures
- Binary Labeling: Each token is labeled with a binary classification (1 for animal entities, 0 for non-animal words)
- Varied Contexts: Includes both single-animal and multi-animal sentences to train the model on different contextual patterns
- Balanced Representation: Every animal class appears with equal frequency in the dataset


## Demo

See `demo.ipynb` for a complete demonstration of the pipeline.