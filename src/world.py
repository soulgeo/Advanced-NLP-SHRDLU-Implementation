from dataclasses import dataclass, field
from typing import Dict, List, Optional
import src.constants as constants


@dataclass
class Obj:
    id: str
    shape: str
    color: str
    size: str
    material: str
    state: Optional[str] = None  # Either "open" or "closed"
    location_id: str = constants.ZONES[0]  # Default is "table"


class World:
    # Initializes the world state
    def __init__(self):
        self.objects: Dict[str, Obj] = {}
        self.on: Dict[str, Optional[str]] = {}
        self.holding: Optional[str] = None
        self.contains: Dict[str, List[str]] = {}

        self._initialize_world()

    # Initializes the world with 12 objects
    def _initialize_world(self):
        initial_objects = [
            Obj("box_wood_1", "box", "brown", "large", "wooden", state=constants.STATE_OPEN, location_id="floor"),
            Obj("box_metal_1", "box", "grey", "medium", "metal", state=constants.STATE_CLOSED, location_id="table"),
            Obj("box_cardboard_1", "box", "brown", "small", "paper", state=constants.STATE_OPEN, location_id="table"),
            Obj("block_red_1", "block", "red", "small", "wooden", location_id="table"),
            Obj("block_blue_1", "block", "blue", "medium", "plastic", location_id="table"),
            Obj("block_green_1", "block", "green", "large", "wooden", location_id="floor"),
            Obj("sphere_green_1", "sphere", "green", "small", "rubber", location_id="floor"),
            Obj("sphere_red_1", "sphere", "red", "medium", "plastic", location_id="table"),
            Obj("pyramid_yellow_1", "pyramid", "yellow", "medium", "wooden", location_id="table"),
            Obj("pyramid_blue_1", "pyramid", "blue", "small", "plastic", location_id="floor"),
            Obj("cylinder_metal_1", "cylinder", "grey", "large", "metal", location_id="floor"),
            Obj("cylinder_red_1", "cylinder", "red", "small", "plastic", location_id="table"),
        ]

        for obj in initial_objects:
            self.objects[obj.id] = obj
            self.on[obj.id] = None
            if obj.shape == "box":
                self.contains[obj.id] = []

    # Checks if the object has something on top of it: Returns True if the object is clear
    def is_clear(self, obj_id: str) -> bool:
        return all(support != obj_id for support in self.on.values())

    # Returns the ID of the object sitting on top of obj_id.
    def top_of(self, obj_id: str) -> Optional[str]:
        for x, support in self.on.items():
            if support == obj_id:
                return x
        return None

    # Searches the world and returns a list with object ID's based on the Parser's request
    def find_objects(
            self,
            shape: Optional[str] = None,
            color: Optional[str] = None,
            size: Optional[str] = None,
            material: Optional[str] = None,
            state: Optional[str] = None,
            location_id: Optional[str] = None,
            relation: Optional[str] = None,
            reference_object_id: Optional[str] = None
    ) -> list[str]:

        matches = []
        for obj_id, obj in self.objects.items():

            if shape and obj.shape != shape: continue
            if color and obj.color != color: continue
            if size and obj.size != size: continue
            if material and obj.material != material: continue
            if state and obj.state != state: continue
            if location_id and obj.location_id != location_id: continue

            if relation and reference_object_id:
                if relation == constants.REL_IN:
                    if obj_id not in self.contains.get(reference_object_id, []):
                        continue
                elif relation in [constants.REL_ON, "ON TOP OF"]:
                    if self.on.get(obj_id) != reference_object_id:
                        continue

            matches.append(obj_id)
        return matches