import nltk
from src.hf_pipeline import HuggingFaceGrounder
from src.intent_classifier import IntentClassifier
from src.sequence_model import SequenceWrapper 
import src.constants as constants

class MLParser:
    def __init__(self, intent_model: IntentClassifier, sequence_model: SequenceWrapper):
        self.intent_classifier = intent_model
        self.sequence_tagger = sequence_model
        self.last_resolved_target = None
        self.hf_grounder: HuggingFaceGrounder | None = None

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
            
            _, _, label = tag.partition("-")
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
            elif label == "D_PRON":
                slots["dest"]["pron"] = token
            elif label.startswith("D_") and label not in ["D_REL", "D_PRON"]:
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

    def _format_candidate(self, intent, candidate):
        """Returns a human-readable string for a single action candidate."""
        target = candidate.get("target")
        destination = candidate.get("destination")
        
        display_intent = intent.lower().replace("_", " ")
        
        msg = f"{display_intent} {target}"
        if destination:
            msg += f" {destination['relation']} {destination['reference']}"
        return msg

    def _build_response(self, intent, candidates):
        """Formats the final API payload based on the list of valid candidates."""
        if len(candidates) == 0:
            return {
                "intent": intent,
                "action_args": None,
                "status": "NOT_FOUND",
                "status_args": {
                    "message": "I couldn't find some of the mentioned objects in the world."
                },
            }

        elif len(candidates) > 1:
            return {
                "intent": intent,
                "action_args": None,
                "status": "AMBIGUOUS",
                "status_args": {
                    "message": "I found multiple physical possibilities for that command. Which did you mean?",
                    "candidates": candidates,
                    "candidate_strings": [
                        self._format_candidate(intent, c) for c in candidates
                    ],
                },
            }

        else:
            return {
                "intent": intent,
                "action_args": candidates[0],
                "status": "RESOLVED",
                "status_args": None,
            }

    def run(self, input_text: str, world, debug: bool = False) -> dict:
        intent = self.intent_classifier.predict(input_text)
        tokens = nltk.word_tokenize(input_text.lower())

        if hasattr(self, "hf_grounder") and self.hf_grounder:
            vocab = self.sequence_tagger.word_to_ix
            tokens = self.hf_grounder.translate_oov_tokens(tokens, vocab, debug=debug)

        tags = self.sequence_tagger.predict_tags(tokens)
        prev_target = self.last_resolved_target
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
                return {
                    "intent": intent,
                    "action_args": None,
                    "status": "NOT_FOUND",
                    "status_args": {"message": "Referred to 'it', but no previous object exists in memory."}
                }
                
        elif slots["target_ref"]:
            anchor_objs = world.find_objects(**slots["target_ref"])
            t_rel = self._normalize_relation(slots.get("t_rel"))
            
            for anchor in anchor_objs:
                found = world.find_objects(
                    **slots["target"], 
                    relation=t_rel, 
                    reference_object_id=anchor
                )
                valid_targets.extend(found)
        else:
            valid_targets = world.find_objects(**slots["target"]) if slots["target"] else []

        valid_targets = list(set(valid_targets))
        if not valid_targets:
            return {
                "intent": intent,
                "action_args": None,
                "status": "NOT_FOUND",
                "status_args": {"message": "Could not identify target."}
            }

        valid_dests = []
        d_rel = constants.REL_ON
        
        if intent == constants.INTENT_PLACE:
            d_rel = self._normalize_relation(slots.get("d_rel"))
            
            if slots["dest"]:
                if "pron" in slots["dest"]:
                    if prev_target is not None:
                        valid_dests = [prev_target]
                    else:
                        return {
                            "intent": intent,
                            "action_args": None,
                            "status": "NOT_FOUND",
                            "status_args": {"message": "Referred to 'it', but no previous object exists in memory."}
                        }
                elif "location_id" in slots["dest"]:
                    valid_dests = [slots["dest"]["location_id"]]
                else:
                    valid_dests = world.find_objects(**slots["dest"])
            
            # If the user omitted a destination or the lookup failed, fallback to the table
            if not valid_dests:
                valid_dests = ["table"]

        candidates = []
        for target_id in valid_targets:
            if intent != constants.INTENT_PLACE:
                candidates.append({"target": target_id})
            else:
                for dest_id in valid_dests:
                    candidates.append({
                        "target": target_id,
                        "destination": {
                            "relation": d_rel,
                            "reference": dest_id
                        }
                    })

        # Update historical memory ONLY if the command is completely unambiguous
        if len(candidates) == 1:
            self.last_resolved_target = candidates[0].get("target")

        # --- STEP 4: TRIGGER THE AMBIGUITY CHECKER ---
        return self._build_response(intent, candidates)
