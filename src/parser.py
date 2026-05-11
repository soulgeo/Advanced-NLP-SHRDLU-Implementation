import nltk

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

grammar = nltk.CFG.fromstring(
    """
    S -> ACT ARGS
    ACT -> PICKUP | PLACE | OPEN | CLOSE | INSPECT
    ARGS -> NP | NP DEST
    DEST -> REL NP | REL ZONE | REL DET ZONE
    NP -> DET AP | DET OBJ_NAME | AP | OBJ_NAME
    AP -> ADJ AP | ADJ OBJ_NAME
    ADJ -> COLOR | SIZE | MATERIAL | STATE
    PICKUP -> "pick" | "pick" "up" | "grab" | "take" | "lift" | "lift" "up" | "get"
    PLACE -> "place" | "put" | "drop" | "leave"
    OPEN -> "open" | "unlock" | "unblock" | "unclose"
    CLOSE -> "close" | "shut" | "shut" "off" | "lock" | "block"
    INSPECT -> "inspect" | "look" "at" | "look" "into" | "check" | "check" "out"
    DET -> "the" | "a" | "an"
    REL -> "inside" | "next" "to" | "onto" | "on" "top" "of" | "below" | "under"
    OBJ_NAME -> "cube" | "pyramid" | "box" | "sphere"
    ZONE -> "table" | "floor"
    COLOR -> "red" | "green" | "blue" | "yellow" | "orange" | "white" | "black"
    SIZE -> "large" | "medium" | "small"
    MATERIAL -> "wooden" | "metal" | "plastic" | "rubber"
    STATE -> "open" | "closed" | "shut"
    """
)

parser = nltk.ChartParser(grammar)

def get_syntax_tree(input: str) -> list[str] | ValueError:
    tokens = nltk.word_tokenize(input.lower())
    tokens = [t for t in tokens if t.isalnum()]
    try:
        trees = list(parser.parse(tokens))
        return trees
    except ValueError as e:
        return e
