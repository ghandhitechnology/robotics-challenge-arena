#!/usr/bin/env python3
"""Independent six-robot contact-physics comparison with the five-robot design.

The variant spec is injected only while the native XML is constructed. The
baseline JSON and module files remain unchanged. All subsequent motion uses
wheel torques, jaws, lifts and the existing KIT gates.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena_mujoco import best_fleet
from arena_mujoco.best_mission import Driver
from arena_mujoco.best_routes import departure_waypoints, Waypoint


class SixDriver(Driver):
    def tasks(self):
        original = super().tasks() if self.key != "aux" else []
        if self.key == "yellow":
            return original[:1]
        if self.key == "green":
            return [original[0], ("Cylinder_Green_09", "right", [1.08, .88], "right")]
        if self.key == "aux":
            return [("Cylinder_Yellow_01", "left", [.12, .10], "left"),
                    ("Cylinder_Yellow_02", "left", [.12, .17], "left"),
                    ("Cylinder_Green_03", "left", [1.08, .76], "right")]
        return original

    def corridor(self, side, y):
        if self.key == "aux" and self.side != side:
            lanes = {"left": .225, "right": .830}
            yield from self.line([lanes[self.side], .420])
            yield from self.line([lanes[side], .420])
            self.side = side
        yield from super().corridor(side, y)

    def work(self, count=None):
        if self.key in ("lab", "kit", "red"):
            yield from super().work(count)
            return
        if self.key == "yellow":
            while self.peers["red"].jobs_done < 1 or self.sim.pose("red")[0][1] > .70:
                yield from self.tick()
        for name, side, target, destination_side in self.tasks():
            yield from self.pickup(name, side)
            yield from self.deliver(name, target, destination_side)
            self.jobs_done += 1
        self.phase = "park"
        if self.key == "yellow":
            yield from self.line([.225, 1.105])
            yield from self.line([.385, 1.105], heading=0.)
        else:
            yield from self.line([.830, 1.100 if self.key == "green" else .630])
        self.phase = "finished"
        yield from self.hold(.2)


def run(output, seed=0, randomize=False, drive_limits=(.35, 2.5)):
    variant = copy.deepcopy(best_fleet.design())
    variant["name"] += " — sixth flexible courier experiment"
    variant["count"] = 6
    variant["start"]["empty_cell"] = "none; sixth courier occupies upper-left after kit preload"
    variant["start"]["deployment_order"] = ["lab", "red", "yellow", "kit", "aux", "green"]
    for robot in variant["robots"]:
        if robot["id"] == "yellow":
            robot.update(role="one_upper_yellow_to_upper_pcc", target_count=1)
        elif robot["id"] == "green":
            robot.update(role="two_upper_green_to_recovery", target_count=2)
    auxiliary = copy.deepcopy(next(r for r in variant["robots"] if r["id"] == "yellow"))
    auxiliary.update(id="aux", role="two_lower_yellow_and_one_lower_green", target_count=3, start_xy_m=[.933, 1.0885],
                     rgba=[.75, .25, .80, 1.])
    variant["robots"].append(auxiliary)
    # Factory injection precedes mjData creation; it never changes a running world.
    with patch.object(best_fleet, "design", return_value=variant):
        sim = best_fleet.FleetSimulation(seed=seed, randomize=randomize, drive_limits=drive_limits)
    if not sim.setup["valid"]:
        raise ValueError(sim.setup["errors"])
    drivers = {key: SixDriver(sim, key) for key in sim.robots}
    for driver in drivers.values():
        driver.peers = drivers
    order = ("lab", "red", "yellow", "kit", "aux", "green")
    cleared, departed, completed, failures, trace = [], {}, [], {}, []
    began = time.perf_counter()

    def program(driver):
        driver.phase = "wait for departure"
        while order[len(cleared)] != driver.key:
            yield from driver.tick()
        driver.phase = "deploy"
        if driver.key == "aux":
            points = (Waypoint((.780, 1.0885), math.pi), Waypoint((.550, 1.0885), math.pi),
                      Waypoint((.550, .800), math.pi / 2), Waypoint((.550, .420), math.pi / 2),
                      Waypoint((.225, .420), math.pi),
                      Waypoint((.225, .150)), Waypoint((.385, .150), 0.))
            clear_index = 1
        else:
            points = departure_waypoints(driver.key, sim.pose(driver.key)[0])
            clear_index = {"lab": 1, "red": 0, "yellow": 1, "kit": 1, "green": 0}[driver.key]
        for index, waypoint in enumerate(points):
            if driver.key == "aux" and index == 3:
                driver.phase = "wait for KIT to clear central crossing"
                while drivers["kit"].jobs_done < 1:
                    yield from driver.tick()
                driver.phase = "deploy"
            yield from driver.line(waypoint.xy, heading=waypoint.heading)
            if index == clear_index:
                cleared.append(driver.key)
        if driver.key in ("red", "yellow", "aux"):
            driver.side = "left"
        departed[driver.key] = float(sim.data.time)
        yield from driver.work()

    programs = {key: program(driver) for key, driver in drivers.items()}
    while programs and sim.data.time < 120:
        for key, generator in list(programs.items()):
            try:
                next(generator)
            except StopIteration:
                completed.append(key)
                programs.pop(key)
                sim.command(key, 0., 0., drivers[key].lift, drivers[key].jaw)
            except (RuntimeError, FloatingPointError) as error:
                failures[key] = str(error)
                programs.pop(key)
                sim.command(key, 0., 0., drivers[key].lift, drivers[key].jaw)
        sim.step(record=True)
        if round(sim.data.time / sim.control_dt) % 50 == 0:
            trace.append({"time_s": float(sim.data.time), "robots": {key: {"xy": sim.pose(key)[0].tolist(), "phase": d.phase} for key, d in drivers.items()}})
    declaration = float(sim.data.time)
    score = sim.score()
    for key, d in drivers.items():
        sim.command(key, 0., 0., d.lift, d.jaw)
    hold = []
    for index in range(250):
        sim.step(record=True)
        if (index + 1) % 5 == 0:
            hold.append(sim.score()["score"])
    report = {"variant": variant, "initial_setup": sim.setup, "declaration_seconds": declaration,
              "deployment_seconds": max(departed.values(), default=0), "departed": departed,
              "completed_robots": completed, "unfinished_robots": list(programs), "failures": failures,
              "events": {key: d.events for key, d in drivers.items()}, "traffic_trace": trace,
              "score_at_declaration": score, "score_after_five_seconds": sim.score(),
              "minimum_hold_score": min(hold), "wall_seconds": time.perf_counter() - began,
              "success": bool(score["official_success"] and not failures and not programs and min(hold) == 160)}
    sim.save(output, report)
    Path(output, "six_robot_variant.json").write_text(json.dumps(variant, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("success", "declaration_seconds", "failures", "minimum_hold_score")}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output/best_design/six_trial_01")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize", action="store_true")
    args = parser.parse_args()
    run(**vars(args))
