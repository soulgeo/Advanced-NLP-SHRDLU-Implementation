import json
from pathlib import Path
from src.sequence_model import SequenceWrapper
from src.intent_classifier import IntentClassifier 

def train_intent_model():
    print("Initializing Intent Classifier...")
    classifier = IntentClassifier()
    
    print("Loading dataset from JSON...")
    # Your partner's load_dataset() method already handles reading the JSON!
    commands, intents = classifier.load_dataset()
    
    print(f"Training on {len(commands)} commands...")
    classifier.train(commands, intents)
    
    # Ensure the models directory exists
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    save_path = models_dir / "intent_model.pkl"
    classifier.save(save_path)
    
    # Quick test to verify it works
    test_cmd = "could you open the wooden box for me"
    prediction = classifier.predict(test_cmd)
    print(f"\nTest Command: '{test_cmd}'")
    print(f"Predicted Intent: {prediction}")

def train_sequence_model():
    # 1. Locate the dataset
    # If running this script from the project root (~/dev/nlp)
    dataset_path = Path("data/intent_commands.json")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {dataset_path}")

    # 2. Load the JSON data
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 3. Format the data for PyTorch
    # The SequenceWrapper expects: [ (["put", "block"], ["O", "B-SHAPE"]), ... ]
    training_data = []
    for item in dataset:
        tokens = item.get("tokens")
        tags = item.get("tags")
        
        # Ensure the data is valid before adding it
        if tokens and tags and len(tokens) == len(tags):
            training_data.append((tokens, tags))

    print(f"Successfully loaded {len(training_data)} formatted sentences.")

    # 4. Initialize the model
    wrapper = SequenceWrapper(embedding_dim=32, hidden_dim=32)
    
    # 5. Train
    print("Starting training loop...")
    wrapper.train(training_data)

    # 6. Save the trained model to disk
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True) # Create the models folder if it doesn't exist
    
    save_path = models_dir / "sequence_model.pt"
    wrapper.save(save_path)

if __name__ == "__main__":
    train_intent_model()
    train_sequence_model()
