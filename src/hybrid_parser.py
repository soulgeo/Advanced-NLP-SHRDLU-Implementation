from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ml_parser import MLParser


class HybridParser:
    def __init__(self, cfg_parser, ml_parser):
        self.cfg_parser = cfg_parser
        self.ml_parser: "MLParser" = ml_parser

    def run(self, user_input, world, debug=False):
        self.cfg_parser.saved_obj = self.ml_parser.last_resolved_target
        payload = self.cfg_parser.run(user_input, world)
        
        if payload["status"] == "PARSE_ERROR":
            if debug:
                print("DEBUG: CFG Parsing failed. Falling back to ML model...")
            ml_payload = self.ml_parser.run(user_input, world, debug=debug)
            self.cfg_parser.saved_obj = self.ml_parser.last_resolved_target
            return ml_payload
            
        # Sync session memory after successful CFG parse
        self.ml_parser.last_resolved_target = self.cfg_parser.saved_obj
        return payload
