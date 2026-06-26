from src.ml_parser import MLParser


class HybridParser:
    def __init__(self, cfg_parser, ml_parser):
        self.cfg_parser = cfg_parser
        self.ml_parser: MLParser = ml_parser

    def run(self, user_input, world):
        # 1. Try the highly accurate Rule-Based Parser
        payload = self.cfg_parser.run(user_input, world)
        
        # 2. Fallback to ML if the Grammar fails
        if payload["status"] == "PARSE_ERROR":
            print("[System] CFG Parsing failed. Falling back to ML model...")
            ml_payload = self.ml_parser.run(user_input, world)
            
            # Always return the ML payload if CFG parsing failed
            # This allows us to see the ML model's output, even if it also fails
            return ml_payload
            
        return payload
