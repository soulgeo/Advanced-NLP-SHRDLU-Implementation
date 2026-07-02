import json
import math
import re
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import src.constants as constants


INTENTS = {
    constants.INTENT_PICKUP,
    constants.INTENT_PLACE,
    constants.INTENT_OPEN,
    constants.INTENT_CLOSE,
    constants.INTENT_INSPECT,
}


class IntentClassifier:
    def __init__(self):
        self.class_document_counts = Counter()
        self.class_token_counts = defaultdict(Counter)
        self.class_total_tokens = Counter()
        self.vocabulary = set()
        self.total_documents = 0
        self.is_trained = False

    def _tokenize(self, command: str) -> list:
        return re.findall(r"[a-z]+(?:'[a-z]+)?", command.lower())

    def load_dataset(self):
        """Loads the labeled command dataset from JSON."""
        project_root = Path(__file__).resolve().parent.parent
        dataset_path = project_root / "data" / "train_set.json"

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}"
            )

        with open(dataset_path, "r", encoding="utf-8") as file:
            dataset = json.load(file)

        if not isinstance(dataset, list):
            raise ValueError(
                "Dataset must be a JSON list."
            )

        commands = []
        intents = []

        for index, item in enumerate(dataset, start=1):
            text = (item.get("text") or "").strip()
            intent = (item.get("intent") or "").strip().upper()
            tokens = item.get("tokens")
            tags = item.get("tags")

            if not text:
                raise ValueError(
                    f"Empty text found in dataset item {index}."
                )

            if intent not in INTENTS:
                raise ValueError(
                    f"Invalid intent '{intent}' in dataset item {index}."
                )

            if not isinstance(tokens, list):
                raise ValueError(
                    f"Tokens must be a list in dataset item {index}."
                )

            if not isinstance(tags, list):
                raise ValueError(
                    f"Tags must be a list in dataset item {index}."
                )

            if len(tokens) != len(tags):
                raise ValueError(
                    f"Tokens and tags have different lengths "
                    f"in dataset item {index}."
                )

            if not tokens:
                raise ValueError(
                    f"Empty token list found in dataset item {index}."
                )

            commands.append(text)
            intents.append(intent)

        if not commands:
            raise ValueError(
                "Dataset contains no examples."
            )

        return commands, intents

    def train(self, commands: list, intents: list):
        """Trains the classifier using the labeled commands."""
        if len(commands) != len(intents):
            raise ValueError(
                "Commands and intents must have the same length."
            )

        if not commands:
            raise ValueError(
                "Cannot train with an empty dataset."
            )

        self.class_document_counts.clear()
        self.class_token_counts.clear()
        self.class_total_tokens.clear()
        self.vocabulary.clear()

        self.total_documents = 0
        self.is_trained = False

        for command, intent in zip(commands, intents):
            intent = intent.strip().upper()

            if intent not in INTENTS:
                raise ValueError(
                    f"Invalid intent: {intent}"
                )

            tokens = self._tokenize(command)

            if not tokens:
                raise ValueError(
                    f"Command contains no valid words: {command}"
                )

            self.class_document_counts[intent] += 1
            self.class_token_counts[intent].update(tokens)
            self.class_total_tokens[intent] += len(tokens)
            self.vocabulary.update(tokens)
            self.total_documents += 1

        self.is_trained = True

    def _get_log_probability(self, tokens: list, intent: str) -> float:
        """Calculates the Naive Bayes log probability for one intent."""
        document_count = self.class_document_counts[intent]

        if document_count == 0:
            return float("-inf")

        score = math.log(
            document_count / self.total_documents
        )

        vocabulary_size = len(self.vocabulary)

        denominator = (
            self.class_total_tokens[intent] + vocabulary_size
        )

        for token in tokens:
            token_count = self.class_token_counts[intent][token]

            score += math.log(
                (token_count + 1) / denominator
            )

        return score

    def predict_scores(self, command: str) -> dict:
        """Returns confidence scores for every possible intent."""
        if not self.is_trained:
            raise RuntimeError(
                "The classifier has not been trained yet."
            )

        tokens = self._tokenize(command)

        if not tokens:
            raise ValueError(
                "Command contains no valid words."
            )

        log_scores = {}

        for intent in INTENTS:
            log_scores[intent] = self._get_log_probability(
                tokens,
                intent
            )

        max_log_score = max(log_scores.values())

        probabilities = {}
        total = 0

        for intent, score in log_scores.items():
            probabilities[intent] = math.exp(
                score - max_log_score
            )
            total += probabilities[intent]

        for intent in probabilities:
            probabilities[intent] /= total

        return dict(
            sorted(
                probabilities.items(),
                key=lambda item: item[1],
                reverse=True
            )
        )

    def predict(self, command: str) -> str:
        """Returns the most likely intent for a command."""
        scores = self.predict_scores(command)
        return next(iter(scores))

    def save(self, filepath: str | Path):
        """Saves the trained Naive Bayes parameters to a file."""
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model.")
            
        model_state = {
            "class_document_counts": self.class_document_counts,
            "class_token_counts": self.class_token_counts,
            "class_total_tokens": self.class_total_tokens,
            "vocabulary": self.vocabulary,
            "total_documents": self.total_documents
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(model_state, f)
            
        print(f"Intent model successfully saved to {filepath}")

    def load(self, filepath: str | Path):
        """Loads trained parameters from a file without needing the dataset."""
        with open(filepath, "rb") as f:
            model_state = pickle.load(f)
            
        self.class_document_counts = model_state["class_document_counts"]
        self.class_token_counts = model_state["class_token_counts"]
        self.class_total_tokens = model_state["class_total_tokens"]
        self.vocabulary = model_state["vocabulary"]
        self.total_documents = model_state["total_documents"]
        
        self.is_trained = True


if __name__ == "__main__":
    classifier = IntentClassifier()

    commands, intents = classifier.load_dataset()
    counts = Counter(intents)

    print("Intent Classifier.")
    print(f"Dataset commands: {len(commands)}")

    for intent in sorted(INTENTS):
        print(f"{intent}: {counts[intent]}")

    print("Dataset loaded successfully.")
    print("No training has been performed yet.")
