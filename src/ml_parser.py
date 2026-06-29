import nltk
from src.intent_classifier import IntentClassifier
from src.sequence_model import SequenceWrapper 
import src.constants as constants

class MLParser:
    def __init__(self, intent_model: IntentClassifier, sequence_model: SequenceWrapper):
        self.intent_classifier = intent_model
        self.sequence_tagger = sequence_model
        self.last_resolved_target = None

    def reset_session(self):
        """Call this whenever a brand new conversation or command chain starts."""
        self.sequence_tagger.reset_session()
        self.last_resolved_target = None

    def _extract_slots(self, tokens, tags):
        """Groups BIO tags into structured dictionary buckets."""
        slots = {
            "target": {}, "target_ref": {}, "dest": {},
            "t_rel": None, "d_rel": constants.REL_ON  # Default relation
        }

        attr_map = {
            "COLOR": "color",
            "SHAPE": "shape",
            "SIZE": "size",
            "MAT": "material",
            "ZONE": "location_id"
        }

        for token, tag in zip(tokens, tags):
            if tag == "O": continue
            
            prefix, _, label = tag.partition("-")
            if not label: continue

            if label.startswith("T_") and label not in ["T_REL", "T_PRON"]:
                suffix = label.replace("T_", "")
                if suffix in attr_map:
                    slots["target"][attr_map[suffix]] = token
            elif label == "T_PRON":
                slots["target"]["pron"] = token
            elif label == "T_REL":
                slots["t_rel"] = token
            elif label.startswith("TR_"):
                suffix = label.replace("TR_", "")
                if suffix in attr_map:
                    slots["target_ref"][attr_map[suffix]] = token
            elif label.startswith("D_") and label != "D_REL":
                suffix = label.replace("D_", "")
                if suffix in attr_map:
                    slots["dest"][attr_map[suffix]] = token
            elif label == "D_REL":
                slots["d_rel"] = token

        return slots

    def _normalize_relation(self, rel_str: str) -> str:
        """Converts raw tokens like 'inside' or 'underneath' to planner constants."""
        if not rel_str: return constants.REL_ON
        if rel_str in ["in", "inside", "into"]: return constants.REL_IN
        if rel_str in ["under", "below", "underneath", "beneath"]: return constants.REL_UNDER
        if rel_str in ["next", "beside", "near"]: return constants.REL_NEXT
        return constants.REL_ON

    def run(self, input_text: str, world) -> dict:
        intent = self.intent_classifier.predict(input_text)
        tokens = nltk.word_tokenize(input_text.lower())
        tags = self.sequence_tagger.predict_tags(tokens)
        
        slots = self._extract_slots(tokens, tags)

        if intent != constants.INTENT_PLACE and slots["dest"]:
            slots["target_ref"].update(slots["dest"])
            if slots["d_rel"]:
                slots["t_rel"] = slots["d_rel"]
            slots["dest"] = {}
            slots["d_rel"] = constants.REL_ON 

        valid_targets = []
        
        if "pron" in slots["target"]:
            if self.last_resolved_target is not None:
                valid_targets = [self.last_resolved_target]
            else:
                return {"status": "NOT_FOUND", "status_args": {"message": "Referred to 'it', but no previous object exists in memory."}}
                
        elif slots["target_ref"]:
            anchor_objs = world.find_objects(**slots["target_ref"])
            t_rel = self._normalize_relation(slots.get("t_rel"))
            
            for anchor in anchor_objs:
                valid_targets = world.find_objects(
                    **slots["target"], 
                    relation=t_rel, 
                    reference_object_id=anchor
                )
                if valid_targets:
                    break
                
        else:
            valid_targets = world.find_objects(**slots["target"]) if slots["target"] else []

        if not valid_targets:
            return {"status": "NOT_FOUND", "status_args": {"message": "Could not identify target."}}

        candidate = {"target": valid_targets[0]}
        self.last_resolved_target = valid_targets[0]

        if intent == constants.INTENT_PLACE:
            dest_ref = "table"
            d_rel = self._normalize_relation(slots.get("d_rel"))
            
            if slots["dest"]:
                if "location_id" in slots["dest"]:
                    dest_ref = slots["dest"]["location_id"]
                else:
                    dest_objs = world.find_objects(**slots["dest"])
                    if dest_objs:
                        dest_ref = dest_objs[0]

            candidate["destination"] = {
                "relation": d_rel,
                "reference": dest_ref
            }

        return {
            "intent": intent,
            "action_args": candidate,
            "status": "RESOLVED",
            "status_args": None
        }
