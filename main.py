import inspect
import json
import os
import readline

from src.hybrid_parser import HybridParser
from src.intent_classifier import IntentClassifier
from src.sequence_model import SequenceWrapper
from src.hf_pipeline import HuggingFaceGrounder
from src.ml_parser import MLParser
from src.cfg_parser import CFGParser
from src.planner import Planner
from src.world import World

data_dir = ".history"
hist_file = os.path.join(data_dir, "command_history.txt")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if os.path.exists(hist_file):
    readline.read_history_file(hist_file)


def main():
    debug = False

    world = World()
    planner = Planner(world)

    cfg_parser = CFGParser()

    intent_model = IntentClassifier()
    intent_model.load("models/intent_model.pkl")

    sequence_model = SequenceWrapper()
    sequence_model.load("models/sequence_model.pt")

    hf_grounder = HuggingFaceGrounder()

    ml_parser = MLParser(intent_model, sequence_model)
    ml_parser.hf_grounder = hf_grounder

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
