from unittest import TestCase, main

from src.cfg_parser import CFGParser as Parser
from src.planner import Planner
from src.world import World
import src.constants as constants

class TestWorld(TestCase):
    def setUp(self):
        self.world: World = World()

    def test_find_objects_by_description(self):
        red_objects = self.world.find_objects(color="red")
        self.assertIn("block_red_1", red_objects)
        self.assertIn("sphere_red_1", red_objects)
        self.assertIn("cylinder_red_1", red_objects)
        self.assertEqual(len(red_objects), 3)

        large_wooden = self.world.find_objects(size="large", material="wooden")
        self.assertIn("box_wood_1", large_wooden)
        self.assertIn("block_green_1", large_wooden)
        self.assertEqual(len(large_wooden), 2)

    def test_find_objects_by_zone(self):
        floor_objects = self.world.find_objects(location_id="floor")
        self.assertIn("box_wood_1", floor_objects)
        self.assertIn("block_green_1", floor_objects)
        self.assertIn("sphere_green_1", floor_objects)
        self.assertIn("pyramid_blue_1", floor_objects)
        self.assertIn("cylinder_metal_1", floor_objects)
        self.assertEqual(len(floor_objects), 5)

    def test_find_objects_by_reference(self):
        self.world.on["block_red_1"] = "box_wood_1"
        
        objs_on_box = self.world.find_objects(relation=constants.REL_ON, reference_object_id="box_wood_1")
        self.assertEqual(objs_on_box, ["block_red_1"])

        self.world.contains["box_wood_1"] = ["block_blue_1"]
        objs_in_box = self.world.find_objects(relation=constants.REL_IN, reference_object_id="box_wood_1")
        self.assertEqual(objs_in_box, ["block_blue_1"])

    def test_find_objects_no_matches(self):
        matches = self.world.find_objects(color="purple", shape="box")
        self.assertEqual(matches, [])

    def test_describe(self):
        import re
        description = re.sub(r'\x1b\[[0-9;]*m', '', self.world.describe())
        self.assertIn("Holding:", description)
        self.assertIn("box_wood_1 is on the floor open", description)
        self.assertIn("block_red_1 is on the table", description)

class TestParser(TestCase):
    def setUp(self) -> None:
        self.world: World = World()
        self.parser: Parser = Parser()

    def test_resolved(self):
        response = self.parser.run("pick up the red block", self.world)
        self.assertEqual(response["status"], "RESOLVED")
        self.assertEqual(response["intent"], constants.INTENT_PICKUP)
        self.assertEqual(response["action_args"]["target"], "block_red_1")

        response = self.parser.run("put the red block on the floor", self.world)
        self.assertEqual(response["status"], "RESOLVED")
        self.assertEqual(response["intent"], constants.INTENT_PLACE)
        self.assertEqual(response["action_args"]["target"], "block_red_1")
        self.assertEqual(response["action_args"]["destination"]["reference"], "floor")

    def test_not_found(self):
        response = self.parser.run("pick up the purple block", self.world)
        self.assertEqual(response["status"], "NOT_FOUND")

    def test_object_memory(self):
        self.parser.run("pick up the red block", self.world)
        self.assertEqual(self.parser.saved_obj, "block_red_1")

        response = self.parser.run("put it on the floor", self.world)
        self.assertEqual(response["status"], "RESOLVED")
        self.assertEqual(response["action_args"]["target"], "block_red_1")

    def test_phrasal_verbs(self):
        response = self.parser.run("pickup the red block", self.world)
        self.assertEqual(response["status"], "RESOLVED")
        self.assertEqual(response["intent"], constants.INTENT_PICKUP)

    def test_nested_reference(self):
        response = self.parser.run("pick up the small red block on the table", self.world)
        self.assertEqual(response["status"], "RESOLVED")
        self.assertEqual(response["action_args"]["target"], "block_red_1")

    def test_target_ambiguity(self):
        response = self.parser.run("pick up the red object", self.world)
        self.assertEqual(response["status"], "AMBIGUOUS")
        self.assertEqual(len(response["status_args"]["candidates"]), 3)

    def test_destination_ambiguity(self):
        response = self.parser.run("put the red block in the box", self.world)
        self.assertEqual(response["status"], "AMBIGUOUS")
        self.assertGreater(len(response["status_args"]["candidates"]), 1)


class TestPlanner(TestCase):
    def setUp(self) -> None:
        self.world: World = World()
        self.planner: Planner = Planner(self.world)

    def test_inspect_object(self):
        result = self.planner.execute({
            "intent": constants.INTENT_INSPECT,
            "action_args": {"target": "block_red_1"}
        })
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(len(self.world.objects), 12)

    def test_pickup_object(self):
        result = self.planner.execute({
            "intent": constants.INTENT_PICKUP,
            "action_args": {"target": "block_red_1"}
        })
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(self.world.holding, "block_red_1")
        self.assertEqual(self.planner.current_holding, "block_red_1")

        result2 = self.planner.execute({
            "intent": constants.INTENT_PICKUP,
            "action_args": {"target": "block_blue_1"}
        })
        self.assertEqual(result2["status"], "ERROR")
        self.assertEqual(self.world.holding, "block_red_1")

    def test_unblocking_pickup(self):
        self.world.on["block_red_1"] = "block_blue_1"

        result = self.planner.execute({
            "intent": constants.INTENT_PICKUP,
            "action_args": {"target": "block_blue_1"}
        })
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(self.world.holding, "block_blue_1")
        self.assertIsNone(self.world.on["block_red_1"])
        self.assertEqual(self.world.objects["block_red_1"].location_id, "table")

    def test_place_object(self):
        self.planner.execute({
            "intent": constants.INTENT_PICKUP,
            "action_args": {"target": "block_red_1"}
        })
        result = self.planner.execute({
            "intent": constants.INTENT_PLACE,
            "action_args": {
                "target": "block_red_1",
                "destination": {"relation": constants.REL_ON, "reference": "table"}
            }
        })
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(self.world.objects["block_red_1"].location_id, "table")
        self.assertIsNone(self.world.holding)

        self.planner.execute({
            "intent": constants.INTENT_PICKUP,
            "action_args": {"target": "block_red_1"}
        })
        result = self.planner.execute({
            "intent": constants.INTENT_PLACE,
            "action_args": {
                "target": "block_red_1",
                "destination": {"relation": constants.REL_ON, "reference": "block_blue_1"}
            }
        })
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(self.world.on["block_red_1"], "block_blue_1")
        self.assertIsNone(self.world.holding)

    def test_place_invalid(self):
        self.planner.execute({
            "intent": constants.INTENT_PICKUP,
            "action_args": {"target": "block_red_1"}
        })
        result = self.planner.execute({
            "intent": constants.INTENT_PLACE,
            "action_args": {
                "target": "block_red_1",
                "destination": {"relation": constants.REL_ON, "reference": "sphere_red_1"}
            }
        })
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(self.world.holding, "block_red_1")

    def test_open_close_box(self):
        self.assertEqual(self.world.objects["box_metal_1"].state, constants.STATE_CLOSED)

        result = self.planner.execute({
            "intent": constants.INTENT_OPEN,
            "action_args": {"target": "box_metal_1"}
        })
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(self.world.objects["box_metal_1"].state, constants.STATE_OPEN)

        result = self.planner.execute({
            "intent": constants.INTENT_CLOSE,
            "action_args": {"target": "box_metal_1"}
        })
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(self.world.objects["box_metal_1"].state, constants.STATE_CLOSED)

    def test_place_inside_box(self):
        self.world.objects["box_metal_1"].state = constants.STATE_OPEN

        self.planner.execute({
            "intent": constants.INTENT_PICKUP,
            "action_args": {"target": "block_red_1"}
        })
        result = self.planner.execute({
            "intent": constants.INTENT_PLACE,
            "action_args": {
                "target": "block_red_1",
                "destination": {"relation": constants.REL_IN, "reference": "box_metal_1"}
            }
        })
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("block_red_1", self.world.contains["box_metal_1"])
        self.assertEqual(self.world.objects["block_red_1"].location_id, "INSIDE_box_metal_1")

    def test_deep_unblocking(self):
        self.world.on["block_red_1"] = "block_blue_1"
        self.world.on["block_blue_1"] = "block_green_1"

        result = self.planner.execute({
            "intent": constants.INTENT_PICKUP,
            "action_args": {"target": "block_green_1"}
        })
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(self.world.holding, "block_green_1")
        self.assertIsNone(self.world.on["block_red_1"])
        self.assertIsNone(self.world.on["block_blue_1"])
        self.assertEqual(self.world.objects["block_red_1"].location_id, "table")
        self.assertEqual(self.world.objects["block_blue_1"].location_id, "table")

    def test_place_under(self):
        result = self.planner.execute({
            "intent": constants.INTENT_PLACE,
            "action_args": {
                "target": "block_red_1",
                "destination": {"relation": constants.REL_UNDER, "reference": "block_blue_1"}
            }
        })
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(self.world.on["block_blue_1"], "block_red_1")
        self.assertEqual(self.world.objects["block_red_1"].location_id, "table")
        self.assertIsNone(self.world.holding)

    def test_open_constraints(self):
        result = self.planner.execute({
            "intent": constants.INTENT_OPEN,
            "action_args": {"target": "block_red_1"}
        })
        self.assertEqual(result["status"], "ERROR")

        self.world.on["block_red_1"] = "box_metal_1"
        result2 = self.planner.execute({
            "intent": constants.INTENT_OPEN,
            "action_args": {"target": "box_metal_1"}
        })
        self.assertEqual(result2["status"], "ERROR")
        self.assertEqual(self.world.objects["box_metal_1"].state, constants.STATE_CLOSED)

    def test_place_in_closed_box(self):
        self.world.objects["box_metal_1"].state = constants.STATE_CLOSED

        result = self.planner.execute({
            "intent": constants.INTENT_PLACE,
            "action_args": {
                "target": "block_red_1",
                "destination": {"relation": constants.REL_IN, "reference": "box_metal_1"}
            }
        })
        self.assertEqual(result["status"], "ERROR")
        self.assertNotIn("block_red_1", self.world.contains["box_metal_1"])


class TestLazyMLParserProxy(TestCase):
    def test_proxy_synchronization_during_load(self):
        import threading
        from main import LazyMLParserProxy
        from src.world import World
        from src.cfg_parser import CFGParser
        from src.hybrid_parser import HybridParser

        class FakeLoader:
            def __init__(self):
                self.ml_parser = None
                self.loaded_event = threading.Event()

            def get_ml_parser(self):
                self.loaded_event.wait()
                return self.ml_parser

        class FakeMLParser:
            def __init__(self):
                self.last_resolved_target = None
            def reset_session(self):
                self.last_resolved_target = None
            def run(self, user_input, world, debug=False):
                return {"intent": "CLOSE", "action_args": None, "status": "NOT_FOUND"}

        loader = FakeLoader()
        ml_proxy = LazyMLParserProxy(loader)
        world = World()
        cfg_parser = CFGParser()
        parser = HybridParser(cfg_parser, ml_proxy)

        # Step 1: ML models are NOT loaded yet.
        payload1 = parser.run("inspect the wooden box", world)
        self.assertEqual(payload1["action_args"]["target"], "box_wood_1")
        self.assertEqual(ml_proxy.last_resolved_target, "box_wood_1")

        # Step 2: ML models finish loading.
        loader.ml_parser = FakeMLParser()
        loader.loaded_event.set()

        # Step 3: Run 'close it'.
        payload2 = parser.run("close it", world)
        self.assertEqual(payload2["status"], "RESOLVED")
        self.assertEqual(payload2["action_args"]["target"], "box_wood_1")


if __name__ == '__main__':
    main()
