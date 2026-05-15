import nltk

import src.constants as constants

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# ------- LEXICON -------

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


# ------- GRAMMAR DEFINITION -------

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
    DET -> "the" | "a" | "an" | "that" | "this"
    OBJ_SYN -> "object" | "thing" | "stuff"

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
