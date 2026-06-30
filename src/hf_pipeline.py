from sentence_transformers import SentenceTransformer, util
import torch

import src.constants as constants

class HuggingFaceGrounder:
    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"Loading Hugging Face model [{model_id}]...")
        self.model = SentenceTransformer(model_id)
        
        self.ontology = {
            "color": constants.COLORS,
            "shape": constants.SHAPES,
            "material": constants.MATERIALS,
            "size": constants.SIZES
        }
        
        self.ontology_embeds = {
            category: self.model.encode(tokens, convert_to_tensor=True) 
            for category, tokens in self.ontology.items()
        }

    def ground_slot(self, messy_word: str, category: str, threshold: float = 0.52) -> str:
        if not messy_word or category not in self.ontology:
            return messy_word

        word_embed = self.model.encode(messy_word, convert_to_tensor=True)
        
        target_embeds = self.ontology_embeds[category]
        canonical_words = self.ontology[category]
        
        # Calculate cosine similarity using the built-in utility
        similarities = util.cos_sim(word_embed, target_embeds)[0]
        
        best_idx = int(torch.argmax(similarities).item())
        best_score = similarities[best_idx].item()

        if best_score >= threshold:
            return canonical_words[best_idx]
            
        return messy_word
