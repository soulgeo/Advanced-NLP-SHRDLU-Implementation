import csv
import math
import re
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
    # Initializes the Naive Bayes classifier state
    def __init__(self):
        self.class_document_counts = Counter()
        self.class_token_counts = defaultdict(Counter)
        self.class_total_tokens = Counter()
        self.vocabulary = set()
        self.total_documents = 0
        self.is_trained = False

    # Splits a command into lowercase word tokens
    def _tokenize(self, command: str) -> list:
        return re.findall(r"[a-z]+(?:'[a-z]+)?", command.lower())

    # Loads the labeled command dataset
    def load_dataset(self):
        project_root = Path(__file__).resolve().parent.parent
        dataset_path = project_root / "data" / "intent_commands.csv"

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}"
            )

        commands = []
        intents = []

        with open(dataset_path, "r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)

            required_columns = {"command", "intent"}

            if (
                not reader.fieldnames
                or not required_columns.issubset(reader.fieldnames)
            ):
                raise ValueError(
                    "CSV must contain the columns: command,intent"
                )

            for row_number, row in enumerate(reader, start=2):
                command = (row.get("command") or "").strip()
                intent = (row.get("intent") or "").strip().upper()

                if not command:
                    raise ValueError(
                        f"Empty command found at row {row_number}."
                    )

                if intent not in INTENTS:
                    raise ValueError(
                        f"Invalid intent '{intent}' at row {row_number}."
                    )

                commands.append(command)
                intents.append(intent)

        return commands, intents

    # Trains the classifier using the labeled commands
    def train(self, commands: list, intents: list):
        if len(commands) != len(intents):
            raise ValueError(
                "Commands and intents must have the same length."
            )

        self.class_document_counts.clear()
        self.class_token_counts.clear()
        self.class_total_tokens.clear()
        self.vocabulary.clear()

        self.total_documents = 0
        self.is_trained = False

        for command, intent in zip(commands, intents):
            tokens = self._tokenize(command)

            if not tokens:
                continue

            self.class_document_counts[intent] += 1
            self.class_token_counts[intent].update(tokens)
            self.class_total_tokens[intent] += len(tokens)
            self.vocabulary.update(tokens)
            self.total_documents += 1

        self.is_trained = True

    # Calculates the Naive Bayes log probability for one intent
    def _get_log_probability(self, tokens: list, intent: str) -> float:
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

    # Returns confidence scores for every possible intent
    def predict_scores(self, command: str) -> dict:
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

    # Returns the most likely intent for a command
    def predict(self, command: str) -> str:
        scores = self.predict_scores(command)
        return next(iter(scores))


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