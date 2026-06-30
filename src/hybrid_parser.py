from src.ml_parser import MLParser


class HybridParser:
    def __init__(self, cfg_parser, ml_parser):
        self.cfg_parser = cfg_parser
        self.ml_parser: MLParser = ml_parser

    def run(self, user_input, world, debug=False):
        payload = self.cfg_parser.run(user_input, world)
        
        if payload["status"] == "PARSE_ERROR":
            if debug:
                print("DEBUG: CFG Parsing failed. Falling back to ML model...")
            ml_payload = self.ml_parser.run(user_input, world, debug=debug)
            return ml_payload
            
        return payload
