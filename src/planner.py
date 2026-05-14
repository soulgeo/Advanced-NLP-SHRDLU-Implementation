from typing import Optional
from world import World
import src.constants as constants


class Planner:
    # Planner gets initialized and connected to the World
    def __init__(self, world: World):
        self.world = world

    # Gets the JSON payload from the parser.
    def execute(self, payload: dict) -> str:
        if payload.get("status") != "RESOLVED":
            return "The order is not fully understood (UNRESOLVED)."

        intent = payload.get("intent")
        args = payload.get("action_args") or {}
        target_id = args.get("target")

        if not target_id or target_id not in self.world.objects:
            return f"The object: '{target_id}' was not found in the world."

        # Routing
        if intent == constants.INTENT_PICKUP:
            return self._pickup(target_id)
        elif intent == constants.INTENT_PLACE:
            dest = args.get("destination") or {}
            return self._place(target_id, dest.get("relation"), dest.get("reference"))
        elif intent == constants.INTENT_OPEN:
            return self._open(target_id)
        elif intent == constants.INTENT_CLOSE:
            return self._close(target_id)
        elif intent == constants.INTENT_INSPECT:
            return self._inspect(target_id)
        else:
            return f"Unknown Intent '{intent}'."

    # Object pick up: Checks whether the hand holds something else and if the object getting picked up does not have anything on top of it
    def _pickup(self, obj_id: str) -> str:
        if self.world.holding is not None:
            return "Another, object already being held"
        if not self.world.is_clear(obj_id):
            return f"Unable to pick-up {obj_id}. There is something on top of it."

        self.world.holding = obj_id
        self.world.on[obj_id] = None

        for box_id, contents in self.world.contains.items():
            if obj_id in contents:
                contents.remove(obj_id)
                break

        return f"{obj_id} has been successfully picked up."

    # The object gets placed according to the rules
    def _place(self, obj_id: str, relation: str, ref_id: Optional[str]) -> str:
        if self.world.holding != obj_id:
            return f"{obj_id} is not being held. The object cannot be placed."

        if not relation:
            return "No destination was determined."

        # PLaced on the table or the floor
        if relation in constants.ZONES:
            self.world.objects[obj_id].location_id = relation
            self.world.on[obj_id] = None
            self.world.holding = None
            return f"{obj_id} placed on the {relation}."

        if not ref_id or ref_id not in self.world.objects:
            return f"Unable to find '{ref_id}'."

        ref_obj = self.world.objects[ref_id]

        if relation == constants.REL_ON:
            if not self.world.is_clear(ref_id):
                return f"The {obj_id} cannot be placed on top of {ref_id}, it already has an object on top of it."
            if ref_obj.shape in ["pyramid", "sphere"]:
                return f"Unable to place something on top of {ref_obj.shape}."

            self.world.on[obj_id] = ref_id
            self.world.objects[obj_id].location_id = ref_obj.location_id
            self.world.holding = None
            return f"The {obj_id} has been placed on top of {ref_id}."

        elif relation == constants.REL_IN:
            if ref_obj.shape != "box":
                return f"The {ref_id} is not a box."
            if ref_obj.state == constants.STATE_CLOSED:
                return f"{ref_id} is closed. Please open it"

            self.world.contains[ref_id].append(obj_id)
            self.world.on[obj_id] = None
            self.world.objects[obj_id].location_id = f"INSIDE_{ref_id}"
            self.world.holding = None
            return f"The {obj_id} has been placed inside {ref_id}."

        return f"Unknown relation: '{relation}'."

    # Opens a container checks if the object is a box and if there is anything on top of it
    def _open(self, obj_id: str) -> str:
        obj = self.world.objects[obj_id]
        if obj.shape != "box":
            return f"{obj_id} cannot be opened."
        if not self.world.is_clear(obj_id):
            return f"{obj_id} cannot be opened, There is something on top of it."
        if obj.state == constants.STATE_OPEN:
            return f"{obj_id} is already open."

        obj.state = constants.STATE_OPEN
        return f"{obj_id} has been opened."

    # Closes a container. Checks if it is box shaped and if there is anything on top of it.
    def _close(self, obj_id: str) -> str:
        obj = self.world.objects[obj_id]
        if obj.shape != "box":
            return f"The {obj_id} cannot close."
        if not self.world.is_clear(obj_id):
            return f"{obj_id} cannot be closed, it has something on top of it"
        if obj.state == constants.STATE_CLOSED:
            return f"{obj_id} is already closed."

        obj.state = constants.STATE_CLOSED
        return f"{obj_id} has been closed."

    # Utility function. Returns a formatted string with the attributes, location, and contents of an object.
    def _inspect(self, obj_id: str) -> str:
        obj = self.world.objects[obj_id]
        state_info = f", State: {obj.state}" if obj.state else ""
        location_info = f", Location: {obj.location_id}"

        support = self.world.on.get(obj_id)
        if support:
            location_info = f", On top of: {support}"

        contains_info = ""
        if obj.shape == "box" and obj.state == constants.STATE_OPEN:
            items = self.world.contains.get(obj_id, [])
            contains_info = f", Contains: {items}" if items else ", Contains: nothing"

        return f"INSPECT {obj_id} -> [Color: {obj.color}, Size: {obj.size}, Material: {obj.material}{state_info}{location_info}{contains_info}]"