import nltk

import src.constants as constants

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# ===============================
# LEXICON
# ===============================

# Verbs
ACT_PICKUP = ["pick", "pick up", "grab", "take", "lift", "lift up", "get"]
ACT_PLACE = ["place", "put", "drop", "leave"]
ACT_OPEN = ["open", "unlock", "unblock", "unclose", "unlid"]
ACT_CLOSE = ["close", "shut", "shut off", "lock", "block"]
ACT_INSPECT = ["inspect", "look at", "look into", "check", "check out"]

# Relations
REL_IN = ["in", "inside", "into"]
REL_ON = ["on", "onto", "on top of", "above"]
REL_UNDER = ["under", "below", "underneath", "beneath"]
REL_NEXT = ["next to", "beside", "near", "by", "close to"]

# Attributes
SIZE_LARGE = ["large", "big", "huge", "giant"]
SIZE_MEDIUM = ["medium", "average", "regular"]
SIZE_SMALL = ["small", "tiny", "little"]
STATE_OPEN = ["open", "unlocked"]
STATE_CLOSED = ["closed", "shut", "locked"]


# ===============================
# GRAMMAR DEFINITION
# ===============================
def make_rules(vocab_list):
    """
    Takes a list of strings and formats them for NLTK CFG.
    Automatically handles multi-word strings by splitting and quoting them.
    Example: "pick up" -> '"pick" "up"'
    """
    formatted_phrases = []
    for phrase in vocab_list:
        words = phrase.split()
        quoted_words = [f'"{w}"' for w in words]
        formatted_phrases.append(" ".join(quoted_words))
    return " | ".join(formatted_phrases)


grammar_string = f"""
    # --- SYNTAX ---
    S -> ACT TARGET | PLACE TARGET | PLACE TARGET DEST
    ACT -> PICKUP | OPEN | CLOSE | INSPECT
    TARGET -> NP | NP REF
    REF -> REL NP | REL NP REF | REL ZONE | REL DET ZONE
    DEST -> REL NP | REL NP REF | REL ZONE | REL DET ZONE
    NP -> DET AP | DET OBJ_NAME | AP | OBJ_NAME
    AP -> ADJ AP | ADJ OBJ_NAME
    ADJ -> COLOR | SIZE | MATERIAL | STATE
    OBJ_NAME -> SHAPE | OBJ_SYN

    # --- STATIC WORDS ---
    DET -> "the" | "a" | "an"
    OBJ_SYN -> "object" | "thing"

    # --- VERB CATEGORIES ---
    PICKUP -> {make_rules(ACT_PICKUP)}
    PLACE -> {make_rules(ACT_PLACE)}
    OPEN -> {make_rules(ACT_OPEN)}
    CLOSE -> {make_rules(ACT_CLOSE)}
    INSPECT -> {make_rules(ACT_INSPECT)}

    # --- RELATION CATEGORIES ---
    REL -> REL_IN | REL_ON | REL_UNDER | REL_NEXT
    REL_IN -> {make_rules(REL_IN)}
    REL_ON -> {make_rules(REL_ON)}
    REL_UNDER -> {make_rules(REL_UNDER)}
    REL_NEXT -> {make_rules(REL_NEXT)}

    # --- ATTRIBUTE CATEGORIES ---
    SHAPE -> {make_rules(constants.SHAPES)}
    ZONE -> {make_rules(constants.ZONES)}

    SIZE -> SIZE_LARGE | SIZE_MEDIUM | SIZE_SMALL
    SIZE_LARGE -> {make_rules(SIZE_LARGE)}
    SIZE_MEDIUM -> {make_rules(SIZE_MEDIUM)}
    SIZE_SMALL -> {make_rules(SIZE_SMALL)}

    STATE -> STATE_OPEN | STATE_CLOSED
    STATE_OPEN -> {make_rules(STATE_OPEN)}
    STATE_CLOSED -> {make_rules(STATE_CLOSED)}

    COLOR -> {make_rules(constants.COLORS)}
    MATERIAL -> {make_rules(constants.MATERIALS)}
"""

# ===============================
# PARSER ENGINE
# ===============================
grammar = nltk.CFG.fromstring(grammar_string)
parser = nltk.ChartParser(grammar)


def get_attributes_of_np(np_tree):
    """Scans the NP tree for attributes like color and size and returns them in a dictionary"""
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


def resolve_reference(parse_tree, world):
    """Recursively finds all object IDs that match descriptions and locations in the tree."""
    # Setup
    np_trees = list(parse_tree.subtrees(lambda t: t.label() == "NP"))
    zone_trees = list(parse_tree.subtrees(lambda t: t.label() == "ZONE"))

    np = None
    attributes = {}
    if len(np_trees) > 0:
        np = np_trees[0]
        attributes = get_attributes_of_np(np)

    references = list(parse_tree.subtrees(lambda t: t.label() == "REF"))

    # Termination condition
    if len(references) == 0:
        if not np:
            zone = zone_trees[0].leaves()[0]
            return [zone]

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
    ref_obj_list = resolve_reference(ref[1], world)

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


def get_action_candidates(trees, world):
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

        valid_target_objects = resolve_reference(target_nodes[0], world)
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
                valid_dest_objects = resolve_reference(dest, world)
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

    return intent, candidates


def build_response(intent, candidates):
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
            },
        }

    else:  # Exactly 1 candidate
        return {
            "intent": intent,
            "action_args": candidates[0],
            "status": "RESOLVED",
            "status_args": None,
        }


def parse_command(input: str, world) -> dict:
    """Main orchestrator: tokenizes input, generates trees, and returns the API payload."""
    tokens = nltk.word_tokenize(input.lower())
    tokens = [t for t in tokens if t.isalnum()]

    try:
        trees = list(parser.parse(tokens))
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

    intent, candidates = get_action_candidates(trees, world)
    return build_response(intent, candidates)
