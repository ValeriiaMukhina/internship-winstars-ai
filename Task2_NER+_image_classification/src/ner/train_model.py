import os
import argparse
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from src.ner.dataset import prepare_dataset
from src.ner.model import NERModel

def train_ner(
    output_dir="./animal_ner_model",
    final_model_path="./animal_ner_final",
    model_name="bert-base-uncased",
    num_labels=2,
    learning_rate=2e-5,
    batch_size=16,
    epochs=3,
    weight_decay=0.01,
    test_size=0.2,
    val_size=0.2,
    disable_wandb=True
):

    # Disable wandb if requested
    if disable_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    
    # Prepare datasets
    datasets = prepare_dataset(test_size=test_size, val_size=val_size)
    
    # Initialize model
    ner_model = NERModel(model_name=model_name, num_labels=num_labels)
    
    # Process datasets
    tokenized_datasets = {}
    for split, dataset in datasets.items():
        tokenized_datasets[split] = dataset.map(
            ner_model.tokenize_and_align_labels, 
            batched=True,
            remove_columns=dataset.column_names
        )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=ner_model.model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=ner_model.tokenizer,
        data_collator=DataCollatorForTokenClassification(ner_model.tokenizer)
    )
    
    # Train model
    trainer.train()
    
    # Save model
    trainer.save_model(final_model_path)
    
    return final_model_path