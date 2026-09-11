import math
import unittest

from arena_mujoco.best_fleet import FleetSimulation
from arena_mujoco.competition_events import (
    initialize_competition_events, robot_fully_outside,
    update_competition_events, validate_initial_setup,
)
from arena_mujoco.runtime import ArenaSimulation


class CompetitionEventTests(unittest.TestCase):
    def setUp(self):
        self.sim = ArenaSimulation(robot="competition", tape_mode="none")
        self.model, self.data, self.meta = self.sim.model, self.sim.data, self.sim.metadata
        self.robot_address = self.model.joint("competition_robot_free").qposadr[0]
        self.data.qpos[self.robot_address:self.robot_address + 7] = [
            .995, .800, .020, math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]

    def place(self, name, xyz):
        address = self.model.joint(f"{name}_free").qposadr[0]
        self.data.qpos[address:address + 3] = xyz

    def test_diagram_start_is_verified_and_cannot_be_initialized_later(self):
        result = initialize_competition_events(self.model, self.data, self.meta)
        self.assertTrue(result["valid"], result["errors"])
        low, high = result["robot_bounds_xyz"]["competition_robot"]
        self.assertGreaterEqual(low[0], .863)
        self.assertGreater(high[2], .14)
        self.data.time = .1
        self.assertFalse(validate_initial_setup(self.model, self.data, self.meta)["valid"])
        with self.assertRaises(ValueError):
            initialize_competition_events(self.model, self.data, self.meta)

    def test_raised_part_overhang_fails_even_with_robot_center_inside(self):
        gid = self.model.geom("robot_comp_controller").id
        self.model.geom_pos[gid, 1] = .20
        result = validate_initial_setup(self.model, self.data, self.meta)
        self.assertFalse(result["valid"])
        self.assertTrue(any("envelope exceeds" in error for error in result["errors"]))

    def test_modified_metadata_does_not_legalize_rearranged_pieces(self):
        for name in ("Medical_Kit_01", "Biological_Sample_01"):
            self.place(name, [.99, .95, .01002 if name.startswith("Medical") else .00252])
            entry = next(entry for entry in self.meta["objects"] if entry["name"] == name)
            entry["initial_position_m"] = [.99, .95, .01002]
        result = validate_initial_setup(self.model, self.data, self.meta)
        self.assertFalse(result["valid"])
        self.assertFalse(result["objects"]["Medical_Kit_01"]["on_diagram_mark"])
        self.assertFalse(result["objects"]["Biological_Sample_01"]["on_diagram_mark"])

    def test_physical_overlap_and_floating_kit_are_rejected(self):
        self.place("Medical_Kit_01", [.947, 1.098, .01002])
        self.place("Medical_Kit_02", [1.02, .85, .20])
        result = validate_initial_setup(self.model, self.data, self.meta)
        self.assertFalse(result["valid"])
        self.assertTrue(result["overlaps"])
        self.assertFalse(result["objects"]["Medical_Kit_02"]["preloaded"])

    def test_supported_robot_preload_is_allowed(self):
        self.place("Medical_Kit_01", [1.020, .855, .044])
        result = validate_initial_setup(self.model, self.data, self.meta)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["preloaded_kits"], ["Medical_Kit_01"])

    def test_full_extent_and_each_exit_transition(self):
        initialize_competition_events(self.model, self.data, self.meta)
        self.data.qpos[self.robot_address] = -.01
        self.assertFalse(robot_fully_outside(self.model, self.data, self.meta))
        update_competition_events(self.model, self.data, self.meta)
        self.assertEqual(self.meta["competition_events"]["robot_outside_count"], 0)
        self.data.qpos[self.robot_address] = -.20
        self.assertTrue(robot_fully_outside(self.model, self.data, self.meta))
        for _ in range(2):
            update_competition_events(self.model, self.data, self.meta)
        self.assertEqual(self.meta["competition_events"]["robot_outside_count"], 1)
        self.data.qpos[self.robot_address] = .50
        update_competition_events(self.model, self.data, self.meta)
        self.data.qpos[self.robot_address] = -.20
        update_competition_events(self.model, self.data, self.meta, human_intervention=True)
        events = self.meta["competition_events"]
        self.assertEqual(events["robot_outside_count"], 2)
        self.assertTrue(events["human_intervention"])
        self.assertEqual(len(events["events"]), 3)

    def test_fleet_overlapping_exits_and_independent_reentry(self):
        sim = FleetSimulation()
        self.assertTrue(sim.setup["valid"], sim.setup["errors"])
        addresses = {key: int(sim.model.joint(f"fleet_{key}_competition_robot_free").qposadr[0])
                     for key in ("red", "yellow")}
        initial_x = {key: float(sim.data.qpos[address]) for key, address in addresses.items()}

        def observe(red_outside, yellow_outside, count):
            # Explicit native geometry snapshots test event transitions, without
            # needing the drive controller to leave and re-enter the arena.
            for key, outside in (("red", red_outside), ("yellow", yellow_outside)):
                sim.data.qpos[addresses[key]] = -.5 if outside else initial_x[key]
            sim.data.time += sim.control_dt
            events = update_competition_events(sim.model, sim.data, sim.metadata)
            self.assertEqual(events["robot_outside_count"], count)
            self.assertEqual(events["robot_was_fully_outside"], red_outside or yellow_outside)
            self.assertEqual(len(events["events"]), count)
            score = sim.score()
            self.assertEqual(score["penalty_points"], -10 * count)
            if count:
                self.assertFalse(score["official_success"])

        observe(False, False, 0)
        observe(True, False, 1)
        observe(True, True, 2)  # Yellow exits while red remains outside.
        observe(True, True, 2)  # Repeated observations do not add penalties.
        observe(False, True, 2)
        observe(True, True, 3)  # Red re-exits while yellow remains outside.
        observe(True, False, 3)
        observe(True, True, 4)
        observe(False, False, 4)
        observe(True, True, 6)  # Simultaneous exits count separately too.
        self.assertEqual([event["robot"] for event in sim.metadata["competition_events"]["events"]],
                         [f"fleet_{key}_competition_robot"
                          for key in ("red", "yellow", "red", "yellow", "red", "yellow")])


if __name__ == "__main__":
    unittest.main()
