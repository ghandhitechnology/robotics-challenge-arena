"""Synchronized fleet mission with explicit contact-only pickup and release.

The geometric planner currently uses simulator poses. Learned vision is evaluated
separately until camera localization and closed-loop task evaluation pass.
"""
from __future__ import annotations

import math
import time
import numpy as np

from .best_fleet import FleetSimulation, wrap


class Driver:
    def __init__(self, simulation, key, policy=None):
        self.sim, self.key, self.policy = simulation, key, policy
        self.lift, self.jaw = .035, .025
        self.phase = "waiting"
        self.events = []
        self.side = "right"

    def tick(self, forward=0., heading=0.):
        if self.policy is None:
            v, yaw = .35 * math.tanh(forward / .055), 2.5 * math.tanh(heading / .35)
        else:
            v, yaw = self.policy(forward, heading)
        self.sim.command(self.key, v, yaw, self.lift, self.jaw)
        yield

    def hold(self, seconds):
        for _ in range(math.ceil(seconds / self.sim.control_dt)):
            yield from self.tick()

    def turn(self, goal, tolerance=.012):
        began = self.sim.data.time
        anchor = self.sim.pose(self.key)[0]
        settled = 0
        while settled < 2:
            position, heading = self.sim.pose(self.key)
            error = wrap(goal - heading)
            forward = float((anchor - position) @ np.array([math.cos(heading), math.sin(heading)]))
            yield from self.tick(forward, error)
            settled = settled + 1 if abs(error) < tolerance else 0
            if self.sim.data.time - began > 8:
                raise RuntimeError(f"{self.key} turn timeout at {self.phase}: {error:.4f}")

    def line(self, target, *, heading=None, tolerance=.002):
        target = np.asarray(target, float)
        position, _ = self.sim.pose(self.key)
        if np.linalg.norm(target - position) < tolerance:
            return
        if heading is None:
            delta = target - position
            heading = math.atan2(delta[1], delta[0])
        yield from self.turn(heading)
        direction = np.array([math.cos(heading), math.sin(heading)])
        lateral_axis = np.array([-direction[1], direction[0]])
        began = self.sim.data.time
        settled = 0
        while settled < 2:
            position, current = self.sim.pose(self.key)
            error = target - position
            forward = float(error @ direction)
            lateral = float(error @ lateral_axis)
            sign = 1 if forward >= 0 else -1
            aim = heading + np.clip(math.atan2(lateral, max(abs(forward), .025)) * sign, -.35, .35)
            angle = wrap(aim - current)
            yield from self.tick(forward * max(0., math.cos(angle)), angle)
            settled = settled + 1 if abs(forward) < tolerance and abs(lateral) < max(tolerance, .0012) else 0
            if self.sim.data.time - began > 12:
                raise RuntimeError(f"{self.key} line timeout at {self.phase}: goal {target}, pose {position}")
        yield from self.turn(heading)

    def position(self, name):
        return self.sim.data.body(name).xpos.copy()

    def corridor(self, side, y):
        lanes = {"left": .260, "right": .830}
        if self.side != side:
            yield from self.line([lanes[self.side], .5905])
            yield from self.line([lanes[side], .5905])
            self.side = side
        yield from self.line([lanes[side], y])

    def pickup(self, name, side):
        self.phase = f"pick {name}"
        kit = name.startswith("Medical")
        sample = name.startswith("Biological")
        position = self.position(name)
        heading = math.pi / 2 if kit else (0. if side == "left" or sample else math.pi)
        direction = np.array([math.cos(heading), math.sin(heading)])
        reach = .105 if self.key == "lab" else .065
        self.lift = .035
        self.jaw = .009 if kit else (.025 if sample else .008)
        if kit:
            yield from self.corridor("right", .960)
            yield from self.line([position[0], .960])
            retreat = np.array([position[0], .960])
        else:
            yield from self.corridor(side, float(position[1]))
            retreat = np.array([.260 if side == "left" else .830, position[1]])
        yield from self.turn(heading)
        if sample:
            yield from self.line(position[:2] - (reach + .07) * direction, heading=heading)
        self.lift = 0.
        yield from self.hold(.65)
        position = self.position(name)
        yield from self.line(position[:2] - reach * direction, heading=heading, tolerance=.0006)
        self.jaw = 0.
        yield from self.hold(.4)
        self.lift = .035
        yield from self.hold(.7)
        if self.position(name)[2] < .020:
            raise RuntimeError(f"{self.key} failed grasp of {name}: {self.position(name)}")
        self.events.append({"event": "grasp", "object": name, "time_s": float(self.sim.data.time)})
        yield from self.line(retreat, heading=heading)
        if kit:
            yield from self.line([.830, .960])

    def deliver(self, name, destination, side):
        self.phase = f"deliver {name}"
        target = np.asarray(destination, float)
        heading = math.pi if side == "left" else 0.
        yield from self.corridor(side, target[1])
        yield from self.turn(heading)
        if self.position(name)[2] < .020:
            raise RuntimeError(f"{self.key} dropped {name} in transit")
        for _ in range(2):
            offset = self.position(name)[:2] - self.sim.pose(self.key)[0]
            yield from self.line(target - offset, heading=heading, tolerance=.0006)
            yield from self.hold(.08)
        self.lift = 0.
        yield from self.hold(.75)
        self.jaw = .025
        yield from self.hold(.55)
        self.events.append({"event": "release", "object": name, "time_s": float(self.sim.data.time),
                            "position_m": self.position(name).tolist()})
        yield from self.line([.260 if side == "left" else .830, target[1]], heading=heading)
        self.lift = .035

    def tasks(self):
        tasks = {
            "lab": [(f"Biological_Sample_{i:02d}", "right", [1.068, y], "right")
                    for i, y in ((1, .5905), (2, .4905), (3, .3905))],
            "kit": [(f"Medical_Kit_{i:02d}", "right", [.09, y], "left")
                    for i, y in ((1, 1.10), (2, .70), (3, .50), (4, .08))],
            "red": [("Cylinder_Red_07", "left", [.12, .80], "left"),
                    ("Cylinder_Red_05", "left", [.12, .40], "left"),
                    ("Cylinder_Red_06", "right", [.12, .60], "left")],
            "yellow": [("Cylinder_Yellow_11", "left", [.12, .94], "left"),
                       ("Cylinder_Yellow_01", "left", [.12, .20], "left"),
                       ("Cylinder_Yellow_02", "right", [.12, .28], "left")],
            "green": [("Cylinder_Green_10", "right", [1.08, 1.03], "right"),
                      ("Cylinder_Green_04", "right", [1.08, .88], "right"),
                      ("Cylinder_Green_09", "left", [1.08, .76], "right")],
        }
        return tasks[self.key]

    def work(self, count=None):
        if self.key == "lab":
            self.sim.extension_targets[self.key] = .04
            yield from self.hold(1.2)
        for name, side, target, destination_side in self.tasks()[:count]:
            yield from self.pickup(name, side)
            yield from self.deliver(name, target, destination_side)
        self.phase = "finished"
        yield from self.hold(.2)


def run(*, seed=0, randomize=False, output=None, only=None, task_count=None, maximum_seconds=120.):
    sim = FleetSimulation(seed=seed, randomize=randomize, only=only)
    if not sim.setup["valid"]:
        raise ValueError(sim.setup["errors"])
    drivers = {key: Driver(sim, key) for key in sim.robots}
    began = time.perf_counter()
    failures = {}
    completed = []
    # Serialize departure through the narrow start cells. The mission work then
    # runs concurrently under the same physics clock.
    def depart(driver):
        driver.phase = "deploy"
        x, y = sim.pose(driver.key)[0]
        if x > 1.0:
            yield from driver.line([x, y - .016], heading=math.pi / 2)
            yield from driver.line([.933, y - .016])
        yield from driver.line([.933, .690], heading=math.pi / 2)
        yield from driver.line([.830, .690])
        stage = {"lab": [.830, .300], "red": [.260, .700],
                 "yellow": [.260, 1.07], "kit": [.830, .940], "green": [.830, 1.10]}[driver.key]
        if driver.key in ("red", "yellow"):
            yield from driver.line([.830, .5905])
            yield from driver.line([.260, .5905])
            driver.side = "left"
        yield from driver.line(stage)

    def advance(programs):
        for key, generator in list(programs.items()):
            try:
                next(generator)
            except StopIteration:
                sim.command(key, 0., 0., drivers[key].lift, drivers[key].jaw)
                programs.pop(key)
                completed.append(key)
            except (RuntimeError, FloatingPointError) as error:
                failures[key] = str(error)
                sim.command(key, 0., 0., drivers[key].lift, drivers[key].jaw)
                programs.pop(key)
        sim.step(record=True)

    for key in sim.robots:
        program = {key: depart(drivers[key])}
        while program and sim.data.time < maximum_seconds:
            advance(program)
        if failures or sim.data.time >= maximum_seconds:
            break
    deployment_time = float(sim.data.time)
    completed.clear()
    programs = {key: driver.work(task_count) for key, driver in drivers.items() if key not in failures}
    if not failures:
        while programs and sim.data.time < maximum_seconds:
            advance(programs)
    declaration = float(sim.data.time)
    score = sim.score()
    for key, driver in drivers.items():
        sim.command(key, 0., 0., driver.lift, driver.jaw)
    for _ in range(250):
        sim.step(record=True)
    settled_score = sim.score()
    report = {"seed": seed, "randomized_physics": randomize, "initial_setup": sim.setup,
              "policy": "geometric teacher", "observations": "simulator poses",
              "deployment_seconds": deployment_time, "declaration_seconds": declaration,
              "completed_robots": completed, "failures": failures,
              "events": {key: driver.events for key, driver in drivers.items()},
              "score_at_declaration": score, "score_after_five_seconds": settled_score,
              "official_time_pass": declaration <= 120 and score["task_score"] == 160,
              "wall_seconds": time.perf_counter() - began, "maximum_torque_nm": sim.max_torque,
              "maximum_tilt_rad": sim.max_tilt, "success": False}
    report["success"] = bool(report["official_time_pass"] and not failures and settled_score["task_score"] == 160)
    if output:
        sim.save(output, report)
    return report
