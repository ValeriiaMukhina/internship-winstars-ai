import argparse
from src.ner.model import NERModel

def run_inference(
    text,
    model_path,
    output_format="list"
):

    # Load the NER model
    ner_model = NERModel(model_path=model_path)
    
    # Get predictions
    detected_entities = ner_model.predict(text)
    
    # Format output according to the requested format
    if output_format == "list":
        return detected_entities
    elif output_format == "json":
        return {"entities": detected_entities, "text": text}
    elif output_format == "verbose":
        return {
            "text": text,
            "entities": detected_entities,
            "entity_count": len(detected_entities),
            "unique_entities": list(set(detected_entities))
        }
    else:
        raise ValueError(f"Unknown output format: {output_format}")
