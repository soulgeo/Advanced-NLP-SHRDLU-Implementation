import torch
from sentence_transformers import SentenceTransformer, util

import src.constants as constants


class HuggingFaceGrounder:
    def __init__(
        self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        print(f"Loading Hugging Face model [{model_id}]...")
        self.model = SentenceTransformer(model_id)

        self.ontology = {
            "color": constants.COLORS,
            "shape": constants.SHAPES,
            "material": constants.MATERIALS,
            "size": constants.SIZES,
        }

        self.flat_ontology = []
        for tokens in self.ontology.values():
            self.flat_ontology.extend(tokens)

        self.flat_embeds = self.model.encode(
            self.flat_ontology, convert_to_tensor=True
        )

    def translate_oov_tokens(
        self, tokens: list, vocab: dict, threshold: float = 0.50, debug: bool = False
    ) -> list:
        """Translates Out-Of-Vocabulary words into known ontology words."""
        translated_tokens = []

        for token in tokens:
            if token in vocab:
                translated_tokens.append(token)
                continue

            token_embed = self.model.encode(token, convert_to_tensor=True)
            similarities = util.cos_sim(token_embed, self.flat_embeds)[0]

            best_idx = int(torch.argmax(similarities).item())
            best_score = similarities[best_idx].item()

            if debug:
                print(
                    f"DEBUG: OOV '{token}' -> nearest: '{self.flat_ontology[best_idx]}' (Score: {best_score:.2f})"
                )

            if best_score >= threshold:
                translated_tokens.append(self.flat_ontology[best_idx])
            else:
                translated_tokens.append(
                    token
                )  # Unrelated word, leave it alone

        return translated_tokens
