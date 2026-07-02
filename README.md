# SHRDLU Grounded Natural Language Understanding System

This project implements a grounded natural language understanding (NLU) system inspired by SHRDLU. It interprets commands about a simulated 3D block-and-box world, resolves object references, and updates a stateful world model.

The system demonstrates a progressive engineering approach from classic rule-based symbols to modern machine learning and transformer-based models:
1. **Stage 1 (Symbolic CFG Parser):** A precise, rule-based parser utilizing NLTK's Context-Free Grammars (CFG) for parsing structured commands.
2. **Stage 2 (Machine Learning Parser):** A statistical pipeline combining a Naive Bayes classifier for intent classification and an LSTM neural network for sequence tagging (BIO slot filling).
3. **Stage 3 (Hugging Face Grounding):** A semantic alignment component utilizing Hugging Face's `all-MiniLM-L6-v2` Sentence Transformer to map out-of-vocabulary (OOV) tokens (e.g. synonyms) to known world attributes.
4. **Hybrid Parser (Production):** Integrates the symbolic CFG parser with the ML pipeline as a fallback, balancing symbolic precision and neural robustness.

---

## Directory Structure

```
project/
├── data/                    # JSON datasets for training & evaluation
├── docs/                    # Assignment requirements, project reports, and documentation
├── models/                  # Saved machine learning model artifacts (.pkl & .pt)
├── src/                     # Source code files (world, parser, planner, model wrappers)
├── main.py                  # Interactive CLI application
├── train.py                 # Training script for ML models
├── eval.py                  # Evaluation script for comparing parsers
├── test.py                  # PyUnit unit tests suite
├── pyproject.toml           # Project dependencies and configuration
└── README.md                # This file
```

---

## Installation

This project requires **Python >= 3.14**. To set up the virtual environment and install the dependencies, execute the following commands in the project root:

### Using standard venv and pip
```bash
# 1. Create a virtual environment
python3 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Install dependencies
pip install -r pyproject.toml
```

### Using uv (Recommended)
If you have `uv` installed, simply run:
```bash
uv sync
```

---

## Execution Instructions

To run the interactive command-line interface (CLI) application, use:

```bash
.venv/bin/python main.py
```

### CLI Interactive Commands
Within the prompt, you can use the following system commands:
* `/world` - Displays the current state, location, and containment relationships of all objects in the world.
* `/debug` - Toggles debug mode to show internal parser payloads and OOV mapping details.
* `/help` - Displays the command syntax guide.
* `/exit` (or `/quit`) - Quits the application.

### Supported Language Commands
The system understands commands for manipulated actions, such as:
* **PICKUP:** `"pick up the red block"`
* **PLACE:** `"put the red block inside the large wooden box"` or `"move the small cylinder on top of the blue pyramid"`
* **OPEN/CLOSE:** `"open the metal box"` or `"close the brown box"`
* **INSPECT:** `"inspect the cardboard box"` or `"look at the red sphere"`
* **Anaphora (Pronouns):** `"pick up the red block"`, then `"put it in the box"` (resolves `"it"` to the last referenced object).

---

## Scripts in Project Root

### 1. Running Unit Tests (`test.py`)
Run the comprehensive suite of unit tests verifying the correctness of the world state, CFG parser, reference resolution, and action planner:
```bash
.venv/bin/python test.py
```

### 2. Training the ML Models (`train.py`)
Train the Naive Bayes intent classifier and LSTM sequence tagger using the labeled data in `data/train_set.json`. The trained models will be saved into the `models/` directory:
```bash
.venv/bin/python train.py
```

### 3. Evaluating Parser Performance (`eval.py`)
Run a quantitative comparison of the different parsing approaches (CFG, ML standard, ML with HF grounding, and Hybrid) over a robustness test dataset of 50 test cases:
```bash
.venv/bin/python eval.py
```
This produces a detailed evaluation report showing parsing success rates, intent accuracy, and token-level/sentence-level slot-filling accuracies.
