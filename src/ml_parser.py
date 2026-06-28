import nltk
from src.intent_classifier import IntentClassifier
from src.sequence_model import SequenceWrapper 
import src.constants as constants

class MLParser:
    def __init__(self, intent_model: IntentClassifier, sequence_model: SequenceWrapper):
        self.intent_classifier = intent_model
        self.sequence_tagger = sequence_model
        self.last_resolved_target = None  # Track the historical reference object

    # def reset_session(self):
    #     """Call this whenever a brand new conversation or command chain starts."""
    #     self.sequence_tagger.reset_session()  # Wipes LSTM memory
    #     self.last_resolved_target = None  # Wipes entity tracking memory

    def run(self, input_text: str, world) -> dict:
        intent = self.intent_classifier.predict(input_text)
        tokens = nltk.word_tokenize(input_text.lower())
        tags = self.sequence_tagger.predict_tags(tokens)
        
        target_attrs = {}
        dest_attrs = {}
        relation = constants.REL_ON
        
        for token, tag in zip(tokens, tags):
            if tag == 'B-T_COLOR': target_attrs['color'] = token
            elif tag == 'B-T_SHAPE': target_attrs['shape'] = token
            # ... rest of your tags ...

        # --- COREFERENCE RESOLUTION LOGIC ---
        # If the word captured is a pronoun, substitute it with the historical target
        if target_attrs.get('shape') == 'it' or target_attrs.get('color') == 'it':
            if self.last_resolved_target is not None:
                # Bypass world lookup and use memory directly
                valid_targets = [self.last_resolved_target]
            else:
                return {"status": "NOT_FOUND", "status_args": {"message": "Referred to 'it', but no previous object exists."}}
        else:
            # Normal lookup
            valid_targets = world.find_objects(**target_attrs) if target_attrs else []
        
        if not valid_targets:
            return {"status": "NOT_FOUND", "status_args": {"message": "Could not identify target."}}

        candidate = {"target": valid_targets[0]}
        
        # Update your conversation history tracker
        self.last_resolved_target = valid_targets[0]

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
