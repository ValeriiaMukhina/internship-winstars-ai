from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

class NERModel:
    def __init__(self, model_path=None, model_name="bert-base-uncased", num_labels=2):
        """
        Initialize the NER model
        
        Args:
            model_path (str, optional): Path to a saved model. If None, initialize from pretrained
            model_name (str): Base model to use if model_path is None
            num_labels (int): Number of classification labels
        """
        if model_path:
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model.eval()
    
    def tokenize_and_align_labels(self, examples):
  
        tokenized_inputs = self.tokenizer(
            [" ".join(tokens) for tokens in examples["tokens"]],
            truncation=True,
            padding="max_length",
            max_length=128
        )
        
        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx])
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def predict(self, text):
       
        # The model expects space-separated tokens
        words = text.lower().split()
        
        # Tokenize like during training
        inputs = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get predicted labels
        predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
        
        # Map predictions back to original words
        entities = []
        word_ids = inputs.word_ids(batch_index=0)
        
        last_word_idx = -1
        for i, word_idx in enumerate(word_ids):
            # Skip special tokens and duplicate wordpieces
            if word_idx is None or word_idx == last_word_idx:
                continue
                
            if predictions[i] == 1 and word_idx < len(words):
                entities.append(words[word_idx])
                
            last_word_idx = word_idx
        
        return entities
    
    def save_model(self, output_path):

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)