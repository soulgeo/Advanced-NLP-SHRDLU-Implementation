import inspect
import json
import os
import readline
import threading

from src.hybrid_parser import HybridParser
from src.cfg_parser import CFGParser
from src.planner import Planner
from src.world import World

data_dir = ".history"
hist_file = os.path.join(data_dir, "command_history.txt")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if os.path.exists(hist_file):
    readline.read_history_file(hist_file)


class BackgroundLoader:
    def __init__(self):
        self.ml_parser = None
        self.exception = None
        self.loaded_event = threading.Event()
        self.thread = threading.Thread(target=self._load, daemon=True)
        self.thread.start()

    def _load(self):
        try:
            import os
            import warnings
            import logging
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ["HF_HUB_VERBOSITY"] = "error"
            warnings.filterwarnings("ignore")
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

            try:
                from huggingface_hub.utils.tqdm import disable_progress_bars
                disable_progress_bars()
            except ImportError:
                pass

            from src.intent_classifier import IntentClassifier
            from src.sequence_model import SequenceWrapper
            from src.hf_pipeline import HuggingFaceGrounder
            from src.ml_parser import MLParser

            intent_model = IntentClassifier()
            intent_model.load("models/intent_model.pkl")

            sequence_model = SequenceWrapper()
            sequence_model.load("models/sequence_model.pt")

            hf_grounder = HuggingFaceGrounder()

            ml_parser = MLParser(intent_model, sequence_model)
            ml_parser.hf_grounder = hf_grounder

            self.ml_parser = ml_parser
        except Exception as e:
            self.exception = e
        finally:
            self.loaded_event.set()

    def get_ml_parser(self):
        if not self.loaded_event.is_set():
            print("\033[93m[System: Loading ML models in background, please wait...]\033[00m")
        self.loaded_event.wait()
        if self.exception:
            raise self.exception
        return self.ml_parser


class LazyMLParserProxy:
    def __init__(self, loader):
        self.loader = loader
        self._last_resolved_target = None

    @property
    def last_resolved_target(self):
        if self.loader.loaded_event.is_set():
            return self.loader.get_ml_parser().last_resolved_target
        return self._last_resolved_target

    @last_resolved_target.setter
    def last_resolved_target(self, value):
        self._last_resolved_target = value
        if self.loader.loaded_event.is_set():
            self.loader.get_ml_parser().last_resolved_target = value

    def reset_session(self):
        if self.loader.loaded_event.is_set():
            self.loader.get_ml_parser().reset_session()
        self._last_resolved_target = None

    def run(self, input_text: str, world, debug: bool = False) -> dict:
        real_parser = self.loader.get_ml_parser()
        real_parser.last_resolved_target = self._last_resolved_target
        res = real_parser.run(input_text, world, debug=debug)
        self._last_resolved_target = real_parser.last_resolved_target
        return res


def main():
    debug = False

    world = World()
    planner = Planner(world)

    cfg_parser = CFGParser()

    loader = BackgroundLoader()
    ml_parser = LazyMLParserProxy(loader)

    parser = HybridParser(cfg_parser, ml_parser)

    # Clear the terminal (simulates Ctrl + L)
    print('\033[H\033[2J', end='')

    print("SHRDLU Parser.")
    print("Type \"/help\" for command syntax or \"/exit\" to quit.")

    try:
        while True:
            user_input = input("\033[93m{}\033[00m".format("SHRDLU > "))
            if user_input == "":
                continue

            if user_input[0] == "/":
                command = user_input[1:]
                if command in ["exit", "quit"]:
                    break

                if command == "world":
                    print(world.describe())
                    continue

                if command == "debug":
                    debug = not debug
                    print(f"Debug mode {"on" if debug == True else "off"}.")
                    continue

                if command == "help":
                    print(inspect.cleandoc("""
                    AVAILABLE SHRDLU COMMANDS:
                    - PICKUP [object]              (e.g., "pick up the red block")
                    - PLACE [obj] [relation] [ref] (e.g., "put the block on the table")
                    - OPEN/CLOSE [object]          (e.g., "open the wooden box")
                    - INSPECT [object]             (e.g., "look at the blue sphere")

                    OTHER COMMANDS:
                    - /debug               Toggle debug mode.
                    - /exit (or /quit)     Quit.
                    - /help                Display available commands.
                    - /world               Display the current state of each object in the world.
                    """))
                    continue

                print("\033[91mCommand not recognized.\033[00m")
                continue

            payload = parser.run(user_input, world, debug=debug)

            if debug == True:
                print("DEBUG: Parser payload:")
                print(json.dumps(payload, indent=4))

            if payload["status"] in ["PARSE_ERROR", "NOT_FOUND"]:
                print(f"\033[91m{payload['status_args']['message']}\033[00m")
                continue

            if payload["status"] == "AMBIGUOUS":
                args = payload["status_args"]
                print(f"\033[96m{args['message']}\033[00m")
                candidates = args["candidates"]

                for i, candidate_msg in enumerate(args["candidate_strings"]):
                    print(f"\033[96m{i+1}. {candidate_msg}\033[00m")

                readline.set_auto_history(False)
                try:
                    while True:
                        choice = input(
                            f"Choose one of the actions (1-{len(candidates)} or \"c\" to cancel): "
                        )
                        if choice.lower() in ["c", "cancel"]:
                            payload["status"] = "CANCELED"
                            break

                        try:
                            choice = int(choice)
                        except ValueError:
                            print("\033[91mInvalid input. Please enter a number or 'c'.\033[00m")
                            continue

                        if choice < 1 or choice > len(candidates):
                            print("\033[91mChoice out of bounds. Try again.\033[00m")
                            continue

                        payload["action_args"] = candidates[choice - 1]
                        payload["status"] = "RESOLVED"
                        break
                finally:
                    readline.set_auto_history(True)

            if payload["status"] == "CANCELED":
                continue

            result = planner.execute(payload)
            if result.get("status") == "SUCCESS":
                print(f"\033[92m{result['message']}\033[00m")
            else:
                print(f"\033[91m{result['message']}\033[00m")

    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        readline.write_history_file(hist_file)


if __name__ == "__main__":
    main()
