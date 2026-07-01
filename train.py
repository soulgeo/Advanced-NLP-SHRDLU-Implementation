import json
import random
from pathlib import Path
import nltk
from src.sequence_model import SequenceWrapper
from src.intent_classifier import IntentClassifier 

def load_and_split_dataset(val_ratio=0.15):
    dataset_path = Path("data/intent_commands.json")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Shuffle before splitting
    random.seed(42)  # For reproducible splits
    random.shuffle(dataset)

    split_idx = int(len(dataset) * (1 - val_ratio))
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]
    return train_data, val_data

def evaluate_models(intent_classifier, sequence_wrapper, val_data, dataset_name="Validation"):
    print(f"\n--- Evaluating Models on {dataset_name} ({len(val_data)} examples) ---")
    
    # Evaluate Intent Classifier
    correct_intents = 0
    for item in val_data:
        text = item["text"]
        expected_intent = item["intent"]
        pred_intent = intent_classifier.predict(text)
        if pred_intent == expected_intent:
            correct_intents += 1
            
    intent_acc = (correct_intents / len(val_data)) * 100
    print(f"Intent Classification Accuracy: {intent_acc:.2f}%")
    
    # Evaluate Sequence Tagger
    total_tokens = 0
    correct_tokens = 0
    exact_match_sentences = 0
    
    for item in val_data:
        tokens = item["tokens"]
        expected_tags = item["tags"]
        
        # Reset the tagger's session memory before prediction
        sequence_wrapper.reset_session()
        pred_tags = sequence_wrapper.predict_tags(tokens)
        
        sentence_exact = True
        # Compare token by token up to the length of the expected or predicted tags
        limit = min(len(pred_tags), len(expected_tags))
        for i in range(limit):
            total_tokens += 1
            if pred_tags[i] == expected_tags[i]:
                correct_tokens += 1
            else:
                sentence_exact = False
                
        # If lengths mismatched, count extra/missing tokens as incorrect
        if len(pred_tags) != len(expected_tags):
            sentence_exact = False
            total_tokens += abs(len(pred_tags) - len(expected_tags))
            
        if sentence_exact:
            exact_match_sentences += 1
            
    token_acc = (correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    sentence_acc = (exact_match_sentences / len(val_data)) * 100
    print(f"Sequence Tag Token-Level Accuracy: {token_acc:.2f}%")
    print(f"Sequence Tag Full-Sentence Match Accuracy: {sentence_acc:.2f}%")

def main():
    # Load and split
    train_data, val_data = load_and_split_dataset()
    print(f"Loaded dataset: {len(train_data) + len(val_data)} total examples.")
    print(f"  Training set size: {len(train_data)}")
    print(f"  Validation set size: {len(val_data)}")

    # 1. Train Intent Classifier
    print("\nTraining Intent Classifier...")
    intent_classifier = IntentClassifier()
    # Format dataset for classifier
    train_cmds = [item["text"] for item in train_data]
    train_intents = [item["intent"] for item in train_data]
    intent_classifier.train(train_cmds, train_intents)
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    intent_classifier.save(models_dir / "intent_model.pkl")

    # 2. Train Sequence Tagger
    print("\nTraining Sequence Tagger...")
    train_seqs = []
    for item in train_data:
        tokens = item.get("tokens")
        tags = item.get("tags")
        if tokens and tags and len(tokens) == len(tags):
            train_seqs.append((tokens, tags))
            
    sequence_wrapper = SequenceWrapper(embedding_dim=32, hidden_dim=32)
    sequence_wrapper.train(train_seqs)
    sequence_wrapper.save(models_dir / "sequence_model.pt")

    # 3. Evaluate on Validation split
    evaluate_models(intent_classifier, sequence_wrapper, val_data, "Validation Set")

    # 4. Evaluate on Robustness Test set (from scratch)
    test_path = Path("data/test_commands.json")
    if test_path.exists():
        with open(test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        evaluate_models(intent_classifier, sequence_wrapper, test_data, "Curated Robustness Test Set")
    else:
        print("\nNo robustness test set found at data/test_commands.json")

if __name__ == "__main__":
    main()
