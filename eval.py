import json
from pathlib import Path
import nltk

from src.sequence_model import SequenceWrapper
from src.intent_classifier import IntentClassifier 
from src.parser import Parser
from src.ml_parser import MLParser
from src.hybrid_parser import HybridParser
from src.world import World

def evaluate_cfg_parser(cfg_parser, test_data, world):
    print("\n--- Evaluating NLTK CFG Parser ---")
    total = len(test_data)
    parsed_successfully = 0
    correct_intents = 0
    
    for item in test_data:
        text = item["text"]
        expected_intent = item["intent"]
        
        payload = cfg_parser.run(text, world)
        
        if payload["status"] != "PARSE_ERROR":
            parsed_successfully += 1
            if payload.get("intent") == expected_intent:
                correct_intents += 1
                
    parse_rate = (parsed_successfully / total) * 100
    intent_acc = (correct_intents / parsed_successfully) * 100 if parsed_successfully > 0 else 0
    
    print(f"CFG Parsing Success Rate (no syntax error): {parse_rate:.2f}% ({parsed_successfully}/{total})")
    print(f"CFG Intent Accuracy (on successfully parsed sentences): {intent_acc:.2f}%")

def evaluate_ml_parser(intent_classifier, sequence_wrapper, test_data):
    print("\n--- Evaluating ML Models ---")
    total = len(test_data)
    correct_intents = 0
    total_tokens = 0
    correct_tokens = 0
    exact_match_sentences = 0
    
    for item in test_data:
        text = item["text"]
        expected_intent = item["intent"]
        tokens = item["tokens"]
        expected_tags = item["tags"]
        
        # Intent evaluation
        pred_intent = intent_classifier.predict(text)
        if pred_intent == expected_intent:
            correct_intents += 1
            
        # Tagger evaluation
        sequence_wrapper.reset_session()
        pred_tags = sequence_wrapper.predict_tags(tokens)
        
        sentence_exact = True
        limit = min(len(pred_tags), len(expected_tags))
        for i in range(limit):
            total_tokens += 1
            if pred_tags[i] == expected_tags[i]:
                correct_tokens += 1
            else:
                sentence_exact = False
                
        if len(pred_tags) != len(expected_tags):
            sentence_exact = False
            total_tokens += abs(len(pred_tags) - len(expected_tags))
            
        if sentence_exact:
            exact_match_sentences += 1
            
    intent_acc = (correct_intents / total) * 100
    token_acc = (correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    sentence_acc = (exact_match_sentences / total) * 100
    
    print(f"Intent Classification Accuracy: {intent_acc:.2f}%")
    print(f"Sequence Tag Token-Level Accuracy: {token_acc:.2f}%")
    print(f"Sequence Tag Full-Sentence Match Accuracy: {sentence_acc:.2f}%")

def evaluate_hybrid_parser(hybrid_parser, test_data, world):
    print("\n--- Evaluating Hybrid Parser (CFG + ML Fallback) ---")
    total = len(test_data)
    resolved_count = 0
    correct_intents = 0
    
    for item in test_data:
        text = item["text"]
        expected_intent = item["intent"]
        
        # Reset ML session memory
        hybrid_parser.ml_parser.reset_session()
        payload = hybrid_parser.run(text, world)
        
        if payload["status"] != "PARSE_ERROR":
            resolved_count += 1
            if payload.get("intent") == expected_intent:
                correct_intents += 1
                
    resolve_rate = (resolved_count / total) * 100
    intent_acc = (correct_intents / total) * 100
    
    print(f"Hybrid Parse Success Rate (no syntax error): {resolve_rate:.2f}% ({resolved_count}/{total})")
    print(f"Hybrid Intent Accuracy (overall): {intent_acc:.2f}%")

def main():
    print("Loading models and initializing parsers...")
    models_dir = Path("models")
    world = World()
    
    # 1. CFG Parser
    cfg_parser = Parser()
    
    # 2. ML Parser
    intent_model = IntentClassifier()
    intent_model.load(models_dir / "intent_model.pkl")
    sequence_model = SequenceWrapper()
    sequence_model.load(models_dir / "sequence_model.pt")
    ml_parser = MLParser(intent_model, sequence_model)
    
    # 3. Hybrid Parser
    hybrid_parser = HybridParser(cfg_parser, ml_parser)
    
    # Load Robustness Test Set
    test_path = Path("data/test_commands.json")
    if not test_path.exists():
        print(f"Could not find robustness test set at {test_path}")
        return
        
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
        
    print(f"\n================ EVALUATION REPORT ({len(test_data)} test cases) ================")
    evaluate_cfg_parser(cfg_parser, test_data, world)
    evaluate_ml_parser(intent_model, sequence_model, test_data)
    evaluate_hybrid_parser(hybrid_parser, test_data, world)
    print("=================================================================")

if __name__ == "__main__":
    main()
