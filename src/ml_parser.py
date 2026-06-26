import nltk
from src.intent_classifier import IntentClassifier
from src.sequence_model import SequenceWrapper 
import src.constants as constants

class MLParser:
    def __init__(self, intent_model, sequence_model):
        self.intent_classifier: IntentClassifier = intent_model
        self.sequence_tagger: SequenceWrapper = sequence_model

    def run(self, input_text: str, world) -> dict:
        # 1. Predict Intent
        intent = self.intent_classifier.predict(input_text)
        
        # 2. Predict Slots
        tokens = nltk.word_tokenize(input_text.lower())
        tags = self.sequence_tagger.predict_tags(tokens)
        
        # 3. Extract Attributes from Tags
        target_attrs = {}
        dest_attrs = {}
        relation = constants.REL_ON # Default
        
        for token, tag in zip(tokens, tags):
            if tag == 'B-T_COLOR': target_attrs['color'] = token
            elif tag == 'B-T_SHAPE':
                if 'shape' not in target_attrs:
                    target_attrs['shape'] = token
            elif tag == 'B-T_MAT': target_attrs['material'] = token
            elif tag == 'B-D_SHAPE': dest_attrs['shape'] = token
            elif tag == 'B-REL': relation = token # Might need mapping to constants

        # 4. Resolve References in the World (Reusing your World logic)
        valid_targets = world.find_objects(**target_attrs) if target_attrs else []
        
        if not valid_targets:
            return {"status": "NOT_FOUND", "status_args": {"message": "Could not identify target via ML."}}

        candidate = {"target": valid_targets[0]} # Simplified: grab first match

        if intent == constants.INTENT_PLACE:
            valid_dests = world.find_objects(**dest_attrs) if dest_attrs else ["table"]
            candidate["destination"] = {
                "relation": relation,
                "reference": valid_dests[0]
            }

        return {
            "intent": intent,
            "action_args": candidate,
            "status": "RESOLVED",
            "status_args": None
        }
