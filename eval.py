import json
from pathlib import Path

from src.sequence_model import SequenceWrapper
from src.intent_classifier import IntentClassifier 
from src.cfg_parser import CFGParser
from src.ml_parser import MLParser
from src.hybrid_parser import HybridParser
from src.world import World

def run_cfg_parser_evaluation(cfg_parser, test_data, world) -> dict:
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
                
    return {
        "total": total,
        "parsed_successfully": parsed_successfully,
        "correct_intents": correct_intents,
        "parse_rate": (parsed_successfully / total) * 100 if total > 0 else 0,
        "intent_acc": (correct_intents / parsed_successfully) * 100 if parsed_successfully > 0 else 0
    }

def run_ml_parser_evaluation(intent_classifier, sequence_wrapper, test_data, hf_grounder=None) -> dict:
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
            
        # Ground OOV tokens if grounder is available
        eval_tokens = tokens
        if hf_grounder:
            vocab = sequence_wrapper.word_to_ix
            eval_tokens = hf_grounder.translate_oov_tokens(list(tokens), vocab)
            
        # Tagger evaluation
        sequence_wrapper.reset_session()
        pred_tags = sequence_wrapper.predict_tags(eval_tokens)
        
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
            
    return {
        "total": total,
        "correct_intents": correct_intents,
        "intent_acc": (correct_intents / total) * 100 if total > 0 else 0,
        "total_tokens": total_tokens,
        "correct_tokens": correct_tokens,
        "token_acc": (correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0,
        "exact_match_sentences": exact_match_sentences,
        "sentence_acc": (exact_match_sentences / total) * 100 if total > 0 else 0
    }

def run_hybrid_parser_evaluation(hybrid_parser, test_data, world) -> dict:
    total = len(test_data)
    resolved_count = 0
    correct_intents = 0
    print('\033[H\033[2J', end='')
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
                
    return {
        "total": total,
        "resolved_count": resolved_count,
        "correct_intents": correct_intents,
        "resolve_rate": (resolved_count / total) * 100 if total > 0 else 0,
        "intent_acc": (correct_intents / total) * 100 if total > 0 else 0
    }

def print_evaluation_report(test_cases_count, cfg_res, ml_res_standard, ml_res_hf, hybrid_res):
    print(f"================ EVALUATION REPORT ({test_cases_count} test cases) ================")
    print("--- NLTK CFG Parser ---")
    print(f"CFG Parsing Success Rate (no syntax error): {cfg_res['parse_rate']:.2f}% ({cfg_res['parsed_successfully']}/{cfg_res['total']})")
    print(f"CFG Intent Accuracy (on successfully parsed sentences): {cfg_res['intent_acc']:.2f}%")
    
    print("\n--- ML Intent Classifier + Sequence Tagger (Standard) ---")
    print(f"Intent Classification Accuracy: {ml_res_standard['intent_acc']:.2f}%")
    print(f"Sequence Tag Token-Level Accuracy: {ml_res_standard['token_acc']:.2f}%")
    print(f"Sequence Tag Full-Sentence Match Accuracy: {ml_res_standard['sentence_acc']:.2f}%")
    
    print("\n--- ML Intent Classifier + Sequence Tagger (w/ HF Grounding) ---")
    print(f"Intent Classification Accuracy: {ml_res_hf['intent_acc']:.2f}%")
    print(f"Sequence Tag Token-Level Accuracy: {ml_res_hf['token_acc']:.2f}%")
    print(f"Sequence Tag Full-Sentence Match Accuracy: {ml_res_hf['sentence_acc']:.2f}%")
    
    print("\n--- Final Hybrid Parser Results ---")
    print(f"Parse Success Rate (no syntax error): {hybrid_res['resolve_rate']:.2f}% ({hybrid_res['resolved_count']}/{hybrid_res['total']})")
    print(f"Intent Accuracy (overall): {hybrid_res['intent_acc']:.2f}%")
    print("===================================================================\n")

def main():
    print("Loading models and initializing parsers...")
    models_dir = Path("models")
    world = World()
    
    # 1. CFG Parser
    cfg_parser = CFGParser()
    
    # 2. ML Parser
    intent_model = IntentClassifier()
    intent_model.load(models_dir / "intent_model.pkl")
    sequence_model = SequenceWrapper()
    sequence_model.load(models_dir / "sequence_model.pt")
    ml_parser = MLParser(intent_model, sequence_model)
    
    # 3. Hybrid Parser
    hybrid_parser = HybridParser(cfg_parser, ml_parser)
    
    # Load Robustness Test Set
    test_path = Path("data/test_set.json")
    if not test_path.exists():
        print(f"Could not find robustness test set at {test_path}")
        return
        
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
        
    print("\nExecuting CFG Parser evaluation...")
    cfg_res = run_cfg_parser_evaluation(cfg_parser, test_data, world)
    
    print("Executing ML Models (Standard) evaluation...")
    ml_res_standard = run_ml_parser_evaluation(intent_model, sequence_model, test_data)
    
    print("\nInitializing Hugging Face Grounder...")
    from src.hf_pipeline import HuggingFaceGrounder
    hf_grounder = HuggingFaceGrounder()
    ml_parser.hf_grounder = hf_grounder
    
    print("Executing ML Models (with HF Grounding) evaluation...")
    ml_res_hf = run_ml_parser_evaluation(intent_model, sequence_model, test_data, hf_grounder)
    
    print("Executing Hybrid Parser (Production) evaluation...")
    hybrid_res = run_hybrid_parser_evaluation(hybrid_parser, test_data, world)

    # Clear the terminal and print the results report
    print("")
    print('\033[H\033[2J', end='')
    print_evaluation_report(len(test_data), cfg_res, ml_res_standard, ml_res_hf, hybrid_res)

if __name__ == "__main__":
    main()
