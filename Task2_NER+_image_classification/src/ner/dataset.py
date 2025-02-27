from datasets import Dataset

def create_animal_dataset():

    animals = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "squirrel", "spider"]
    templates = [
        "The {animal} is here",
        "I see a {animal}",
        "There is a {animal}",
        "Look at the {animal}",
        "I found a {animal} and a {animal2}"
    ]
    
    texts = []
    tags = []
    
    for animal in animals:
        # Simple templates
        for template in templates:
            if "{animal2}" not in template:
                words = template.format(animal=animal).lower().split()
                labels = [1 if word == animal else 0 for word in words]
                texts.append(words)
                tags.append(labels)
        
        # Two animal templates
        for animal2 in animals:
            words = f"I see a {animal} and a {animal2}".lower().split()
            labels = [1 if word in [animal, animal2] else 0 for word in words]
            texts.append(words)
            tags.append(labels)
    
    return Dataset.from_dict({'tokens': texts, 'labels': tags})

def prepare_dataset(test_size=0.2, val_size=0.2):
  
    dataset = create_animal_dataset()
    
    # First split into train and test
    train_test = dataset.train_test_split(test_size=test_size)
    
    # Then split train into train and validation
    train_val = train_test['train'].train_test_split(test_size=val_size)
    
    return {
        'train': train_val['train'],
        'validation': train_val['test'],
        'test': train_test['test']
    }