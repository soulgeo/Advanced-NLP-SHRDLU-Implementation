"""
Shared vocabulary for the nlp system.
"""

# CORE ATTRIBUTES
COLORS = [
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "brown",
    "pink",
    "purple",
    "cyan",
    "magenta",
    "white",
    "black",
]
SHAPES = ["block", "pyramid", "sphere", "cylinder", "cone", "box"]
MATERIALS = ["wooden", "metal", "plastic", "rubber", "paper"]
ZONES = ["table", "floor"]

# STANDARDIZED CATEGORIES
# Sizes
SIZE_LARGE = "large"
SIZE_MEDIUM = "medium"
SIZE_SMALL = "small"
SIZES = [SIZE_LARGE, SIZE_MEDIUM, SIZE_SMALL]

# States
STATE_OPEN = "open"
STATE_CLOSED = "closed"
STATES = [STATE_OPEN, STATE_CLOSED]

# STANDARDIZED RELATIONS
# The planner only needs to know these canonical versions
REL_IN = "in"
REL_ON = "on"
REL_UNDER = "under"
REL_NEXT = "next to"

# INTENTS / ACTIONS
# Standardized action labels for the JSON contract
INTENT_PICKUP = "PICKUP"
INTENT_PLACE = "PLACE"
INTENT_OPEN = "OPEN"
INTENT_CLOSE = "CLOSE"
INTENT_INSPECT = "INSPECT"
