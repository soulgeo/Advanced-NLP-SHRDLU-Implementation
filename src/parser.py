import nltk

import src.constants as constants
from src.grammar import grammar_string
from src.world import World


class Parser():
    def __init__(self):
        self.grammar = nltk.CFG.fromstring(grammar_string)
        self.parser = nltk.ChartParser(self.grammar)

        self.saved_obj = None


    def _get_attributes_of_np(self, np_tree):
        """Scans an NP tree for attributes like color and size and returns them in a dictionary"""
        attributes = {}
        attribute_trees = list(
            np_tree.subtrees(
                filter=lambda t: t.label()
                in ["COLOR", "SHAPE", "SIZE", "MATERIAL", "STATE"]
            )
        )
        for attr_tree in attribute_trees:
            label = attr_tree.label()
            dict_key = label.lower()
            if label in ["COLOR", "SHAPE", "MATERIAL"]:
                attributes[dict_key] = attr_tree.leaves()[0]
            elif label == "SIZE":
                size_tree = list(attr_tree.subtrees(lambda t: t.height() == 2))[0]
                size_dict = {
                    "SIZE_LARGE": constants.SIZE_LARGE,
                    "SIZE_MEDIUM": constants.SIZE_MEDIUM,
                    "SIZE_SMALL": constants.SIZE_SMALL,
                }
                attributes[dict_key] = size_dict[size_tree.label()]
            else:  # STATE
                state_tree = list(attr_tree.subtrees(lambda t: t.height() == 2))[0]
                state_dict = {
                    "STATE_OPEN": constants.STATE_OPEN,
                    "STATE_CLOSED": constants.STATE_CLOSED,
                }
                attributes[dict_key] = state_dict[state_tree.label()]

        return attributes


    def _resolve_reference(self, parse_tree, world):
        """Recursively finds all object IDs that match descriptions and locations in the tree."""
        # Setup
        object_trees = list(parse_tree.subtrees(lambda t: t.label() in ["NP", "ANAPHORIC"]))
        zone_trees = list(parse_tree.subtrees(lambda t: t.label() == "ZONE"))

        obj_tree = None
        attributes = {}
        if len(object_trees) > 0:
            obj_tree = object_trees[0]
            if obj_tree.label() == "NP":
                attributes = self._get_attributes_of_np(obj_tree)

        references = list(parse_tree.subtrees(lambda t: t.label() == "REF"))

        # Termination condition
        if len(references) == 0:
            if not obj_tree:
                zone = zone_trees[0].leaves()[0]
                return [zone]

            if obj_tree.label() == "ANAPHORIC":
                return [self.saved_obj] if self.saved_obj is not None else []

            obj_list = world.find_objects(**attributes)
            return obj_list

        # Recursion
        ref = references[0]
        relation_tree = list(
            ref.subtrees(
                lambda t: t.label() in ["REL_IN", "REL_ON", "REL_UNDER", "REL_NEXT"]
            )
        )[0]
        relation_dict = {
            "REL_IN": constants.REL_IN,
            "REL_ON": constants.REL_ON,
            "REL_UNDER": constants.REL_UNDER,
            "REL_NEXT": constants.REL_NEXT,
        }
        relation = relation_dict[relation_tree.label()]
        ref_obj_list = self._resolve_reference(ref[1], world)

        obj_list = []
        for obj in ref_obj_list:
            if obj in ["table", "floor"]:
                ref_kwargs = {"location_id": obj}
            else:
                ref_kwargs = {
                    "relation": relation,
                    "reference_object_id": obj,
                }
            objs = world.find_objects(**attributes, **ref_kwargs)
            obj_list.extend(objs)

        return obj_list


    def _get_action_candidates(self, trees, world):
        """Loops through syntax trees, queries the world, and flattens all physical possibilities."""
        candidates = []
        intent = ""

        for i, tree in enumerate(trees):
            # Extract the intent (only need to do this once)
            if i == 0:
                for subtree in tree.subtrees():
                    label = subtree.label()
                    if label in ["PICKUP", "PLACE", "OPEN", "CLOSE", "INSPECT"]:
                        intent = label
                        break

            # Extract the target
            target_nodes = list(tree.subtrees(lambda t: t.label() == "TARGET"))
            if not target_nodes:
                continue

            valid_target_objects = self._resolve_reference(target_nodes[0], world)
            if len(valid_target_objects) == 0:
                continue

            # Extract the destination (if PLACE)
            valid_dest_objects = []
            relation = ""
            if intent == "PLACE":
                dest_nodes = list(tree.subtrees(lambda t: t.label() == "DEST"))

                if len(dest_nodes) == 0:
                    valid_dest_objects = ["floor"]
                    relation = constants.REL_ON
                else:
                    dest = dest_nodes[0]
                    valid_dest_objects = self._resolve_reference(dest, world)
                    if len(valid_dest_objects) == 0:
                        continue

                    relation_tree = list(
                        dest.subtrees(
                            lambda t: t.label()
                            in ["REL_IN", "REL_ON", "REL_UNDER", "REL_NEXT"]
                        )
                    )[0]

                    relation_dict = {
                        "REL_IN": constants.REL_IN,
                        "REL_ON": constants.REL_ON,
                        "REL_UNDER": constants.REL_UNDER,
                        "REL_NEXT": constants.REL_NEXT,
                    }
                    relation = relation_dict[relation_tree.label()]

            # Append all candidates
            for target_obj in valid_target_objects:
                if intent != "PLACE":
                    candidates.append({"target": target_obj})
                    continue

                for dest_obj in valid_dest_objects:
                    candidates.append(
                        {
                            "target": target_obj,
                            "destination": {
                                "relation": relation,
                                "reference": dest_obj,
                            },
                        }
                    )

            # Save the target object if only one
            if len(valid_target_objects) == 1:
                self.saved_obj = valid_target_objects[0]

        return intent, candidates


    def _format_candidate(self, intent, candidate):
        """Returns a human-readable string for a single action candidate."""
        target = candidate.get("target")
        destination = candidate.get("destination")
        
        # Clean up intent for display (e.g., PICKUP -> pick up)
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

        else:  # Exactly 1 candidate
            return {
                "intent": intent,
                "action_args": candidates[0],
                "status": "RESOLVED",
                "status_args": None,
            }


    def run(self, input: str, world: World) -> dict:
        """Main orchestrator: tokenizes input, generates trees, and returns the API payload."""
        tokens = nltk.word_tokenize(input.lower())
        tokens = [t for t in tokens if t.isalnum()]

        try:
            trees = list(self.parser.parse(tokens))
        except ValueError as e:
            return {
                "intent": None,
                "action_args": None,
                "status": "PARSE_ERROR",
                "status_args": {"message": str(e)},
            }

        if len(trees) == 0:
            return {
                "intent": None,
                "action_args": None,
                "status": "PARSE_ERROR",
                "status_args": {
                    "message": "Sorry, but I don't understand. Please try again."
                },
            }

        intent, candidates = self._get_action_candidates(trees, world)
        return self._build_response(intent, candidates)
