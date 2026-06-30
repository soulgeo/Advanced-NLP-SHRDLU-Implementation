import json
import random
from pathlib import Path

# DOMAIN VOCABULARY
COLORS = ["red", "green", "blue", "yellow", "orange", "brown", "pink", "purple", "cyan", "magenta", "white", "black"]
SHAPES = ["block", "pyramid", "sphere", "cylinder", "cone", "box"]
MATERIALS = ["wooden", "metal", "plastic", "rubber", "paper"]
SIZES = ["large", "medium", "small"]
ZONES = ["table", "floor"]
PRONOUNS = ["it", "that"]

# Synonyms mapped 1-to-1 to single tokens to prevent BIO alignment bugs
ACT_PICKUP = ["pick", "grab", "take", "lift", "get", "retrieve"]
ACT_PLACE = ["place", "put", "drop", "set", "leave"]
ACT_OPEN = ["open", "unlock", "unclose", "unlid"]
ACT_CLOSE = ["close", "shut", "lock"]
ACT_INSPECT = ["inspect", "check", "examine"]

REL_ANCHOR = ["under", "below", "inside", "into", "near", "beside"]
REL_DEST = ["on", "onto", "above", "in", "inside"]

DETS = ["the", "a", "an", "this", "that"]


def generate_np(prefix: str, force_shape: str = None):
    """Generates a noun phrase and its corresponding BIO tags."""
    words, tags = [], []
    
    if random.random() > 0.2:
        words.append(random.choice(DETS))
        tags.append("O")
        
    if random.random() > 0.3:
        words.append(random.choice(SIZES))
        tags.append(f"B-{prefix}_SIZE")
        
    if random.random() > 0.2:
        words.append(random.choice(COLORS))
        tags.append(f"B-{prefix}_COLOR")
        
    if random.random() > 0.4:
        words.append(random.choice(MATERIALS))
        tags.append(f"B-{prefix}_MAT")
        
    shape = force_shape if force_shape else random.choice(SHAPES)
    words.append(shape)
    tags.append(f"B-{prefix}_SHAPE")
    
    return words, tags


def build_item(intent: str, words: list, tags: list):
    """Validates and packages a single dataset dictionary."""
    assert len(words) == len(tags), f"Alignment Error: {words} vs {tags}"
    return {
        "text": " ".join(words),
        "intent": intent,
        "tokens": words,
        "tags": tags
    }


def generate_synthetic_dataset():
    random.seed(42)  # Fixed seed for academic reproducibility
    dataset = []

    # =========================================================================
    # CATEGORY 1: TARGET REFERENCES (800 items)
    # Distribution: 200 PICKUP, 200 INSPECT, 200 OPEN, 200 CLOSE
    # =========================================================================
    ref_intents = [("PICKUP", ACT_PICKUP, None)] * 200 + \
                  [("INSPECT", ACT_INSPECT, None)] * 200 + \
                  [("OPEN", ACT_OPEN, "box")] * 200 + \
                  [("CLOSE", ACT_CLOSE, "box")] * 200

    for intent_name, verbs, forced_shape in ref_intents:
        words = [random.choice(verbs)]
        tags = ["O"]

        t_words, t_tags = generate_np("T", force_shape=forced_shape)
        words.extend(t_words)
        tags.extend(t_tags)

        words.append(random.choice(REL_ANCHOR))
        tags.append("B-T_REL")

        tr_words, tr_tags = generate_np("TR")
        words.extend(tr_words)
        tags.extend(tr_tags)

        dataset.append(build_item(intent_name, words, tags))

    # =========================================================================
    # CATEGORY 2: PRONOUNS (500 items)
    # Distribution: exactly 100 for each of the 5 intents
    # =========================================================================
    pron_intents = ["PICKUP", "PLACE", "OPEN", "CLOSE", "INSPECT"]
    
    for intent_name in pron_intents:
        verb_pool = {
            "PICKUP": ACT_PICKUP, "PLACE": ACT_PLACE, "OPEN": ACT_OPEN, 
            "CLOSE": ACT_CLOSE, "INSPECT": ACT_INSPECT
        }[intent_name]

        for _ in range(100):
            words = [random.choice(verb_pool), random.choice(PRONOUNS)]
            tags = ["O", "B-T_PRON"]

            if intent_name == "PLACE":
                words.append(random.choice(REL_DEST))
                tags.append("B-D_REL")
                
                # 50% chance destination is a static zone, 50% physical object
                if random.random() > 0.5:
                    words.extend(["the", random.choice(ZONES)])
                    tags.extend(["O", "B-D_ZONE"])
                else:
                    d_words, d_tags = generate_np("D")
                    words.extend(d_words)
                    tags.extend(d_tags)

            dataset.append(build_item(intent_name, words, tags))

    # =========================================================================
    # CATEGORY 3: STANDARD TARGETS WITH ZONES (300 items)
    # Distribution: Strictly PLACE
    # =========================================================================
    for _ in range(300):
        words = [random.choice(ACT_PLACE)]
        tags = ["O"]

        t_words, t_tags = generate_np("T")
        words.extend(t_words)
        tags.extend(t_tags)

        words.append(random.choice(["on", "onto"]))
        tags.append("B-D_REL")

        words.extend(["the", random.choice(ZONES)])
        tags.extend(["O", "B-D_ZONE"])

        dataset.append(build_item("PLACE", words, tags))

    random.shuffle(dataset)

    output_path = Path("data/synthetic_dataset_1600.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Dataset successfully generated: {len(dataset)} items written to {output_path}")


if __name__ == "__main__":
    generate_synthetic_dataset()
