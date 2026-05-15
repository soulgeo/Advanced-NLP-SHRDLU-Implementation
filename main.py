import os
import readline

from src.parser import parse_command
from src.planner import Planner
from src.world import World

data_dir = ".history"
hist_file = os.path.join(data_dir, "command_history.txt")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Load history safely
if os.path.exists(hist_file):
    readline.read_history_file(hist_file)

def main():
    world = World()
    planner = Planner(world)

    try:
        while True:
            user_input = input("SHRDLU > ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if user_input == "":
                continue
            if user_input == "/world":
                continue

            payload = parse_command(user_input, world)

            if payload["status"] in ["PARSE_ERROR", "NOT_FOUND"]:
                print(payload["status_args"]["message"])
                continue

            if payload["status"] == "AMBIGUOUS":
                args = payload["status_args"]
                print(args["message"])
                candidates = args["candidates"]
                
                for i, candidate_msg in enumerate(args["candidate_strings"]):
                    print(f"{i+1}. {candidate_msg}")

                while True:
                    choice = int(
                        input(
                            f"Choose one of the actions (1-{len(candidates)}): "
                        )
                    )
                    if choice < 1 or choice > len(candidates):
                        print("Invalid choice. Try again.")
                        continue
                    payload["action_args"] = candidates[choice - 1]
                    payload["status"] = "RESOLVED"
                    break

            # RESOLVED state, or ambiguity solved.
            print(planner.execute(payload))

    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        readline.write_history_file(hist_file)


if __name__ == "__main__":
    main()
