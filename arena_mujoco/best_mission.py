"""Synchronized fleet mission with explicit contact-only pickup and release.

The geometric planner currently uses simulator poses. Learned vision is evaluated
separately until camera localization and closed-loop task evaluation pass.
"""
from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from .best_fleet import FleetSimulation, wrap
from .best_routes import DepartureSequencer


def mission_sources():
    root = Path(__file__).resolve().parents[1]
    names = ("arena_spec.json", "best_design.json", "competition_rules.json",
             "arena_mujoco/best_fleet.py", "arena_mujoco/best_mission.py",
             "arena_mujoco/best_routes.py", "arena_mujoco/best_drive.py",
             "arena_mujoco/builder.py", "arena_mujoco/geometry.py",
             "arena_mujoco/materials.py", "arena_mujoco/competition_robot.py",
             "arena_mujoco/competition_scoring.py", "arena_mujoco/competition_events.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


class Driver:
    def __init__(self, simulation, key, policy=None):
        self.sim, self.key, self.policy = simulation, key, policy
        self.lift, self.jaw = .035, .025
        self.phase = "waiting"
        self.events = []
        self.side = "right"
        self.peers = {}
        self.jobs_done = 0
        self.learned_calls = 0
        self.geometric_calls = 0

    def tick(self, forward=0., heading=0., *, target=None, target_heading=None, precision=False):
        if self.policy is None or target is None or precision:
            linear_limit, yaw_limit = self.sim.robot_drive_limits[self.key]
            v, yaw = linear_limit * math.tanh(forward / .055), yaw_limit * math.tanh(heading / .35)
            self.geometric_calls += 1
        else:
            v, yaw = self.policy.command(self.sim, target, target_heading)
            self.learned_calls += 1
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
            yield from self.tick(forward, error, target=anchor, target_heading=goal,
                                 precision=abs(error) < .08)
            settled = settled + 1 if abs(error) < tolerance else 0
            if self.sim.data.time - began > 8:
                raise RuntimeError(f"{self.key} turn timeout at {self.phase}: {error:.4f}")

    def line(self, target, *, heading=None, tolerance=.005, retries=2):
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
            if (abs(forward) < tolerance and abs(lateral) > max(tolerance, .0012)
                    and self.sim.data.time - began > 2.5 and retries > 0):
                # A differential drive cannot correct sideways error in place.
                # Back away on the approach heading, then approach again.
                yield from self.line(target - .025 * direction, heading=heading,
                                     tolerance=.004, retries=retries - 1)
                yield from self.line(target, heading=heading, tolerance=tolerance,
                                     retries=retries - 1)
                return
            sign = 1 if forward >= 0 else -1
            aim = heading + np.clip(math.atan2(lateral, max(abs(forward), .025)) * sign, -.35, .35)
            angle = wrap(aim - current)
            yield from self.tick(forward * max(0., math.cos(angle)), angle,
                                 target=target, target_heading=heading,
                                 precision=np.linalg.norm(error) < .030)
            settled = settled + 1 if abs(forward) < tolerance and abs(lateral) < max(tolerance, .0012) else 0
            if self.sim.data.time - began > 12:
                raise RuntimeError(f"{self.key} line timeout at {self.phase}: goal {target}, pose {position}")
        yield from self.turn(heading)

    def position(self, name):
        return self.sim.data.body(name).xpos.copy()

    def corridor(self, side, y):
        current_y = self.sim.pose(self.key)[0][1]
        lanes = {"left": .225, "right": .830}
        if self.side != side:
            crossing_y = .710 if self.key == "green" else (.5905 if self.key == "kit" else .500)
            yield from self.line([lanes[self.side], crossing_y])
            yield from self.line([lanes[side], crossing_y])
            self.side = side
        position = self.sim.pose(self.key)[0]
        if abs(position[0] - lanes[side]) > .006:
            yield from self.line([lanes[side], position[1]])
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
        at_pickup = np.linalg.norm(position[:2] - reach * direction - self.sim.pose(self.key)[0]) < .015
        if at_pickup and not kit:
            retreat = np.array([.225 if side == "left" else .830, position[1]])
        elif kit:
            yield from self.corridor("right", .960)
            yield from self.line([position[0], .960])
            retreat = np.array([position[0], .960])
        else:
            yield from self.corridor(side, float(position[1]))
            retreat = np.array([.225 if side == "left" else .830, position[1]])
        yield from self.turn(heading)
        if sample:
            yield from self.line(position[:2] - (reach + .07) * direction, heading=heading)
        self.lift = 0.
        yield from self.hold(.65)
        position = self.position(name)
        yield from self.line(position[:2] - reach * direction, heading=heading,
                             tolerance=.0006 if sample else .0015)
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
        sample = name.startswith("Biological")
        for _ in range(2 if sample else 1):
            offset = self.position(name)[:2] - self.sim.pose(self.key)[0]
            yield from self.line(target - offset, heading=heading, tolerance=.0006 if sample else .003)
            yield from self.hold(.08)
        self.lift = 0.
        yield from self.hold(.75)
        self.jaw = .025
        yield from self.hold(.55)
        self.events.append({"event": "release", "object": name, "time_s": float(self.sim.data.time),
                            "position_m": self.position(name).tolist()})
        yield from self.line([.225 if side == "left" else .830, target[1]], heading=heading)
        self.lift = .035

    def tasks(self):
        tasks = {
            "lab": [(f"Biological_Sample_{i:02d}", "right", [1.068, y], "right")
                    for i, y in ((1, .3905), (2, .4905), (3, .5905))],
            "kit": [(f"Medical_Kit_{i:02d}", "right", [.09, y], "left")
                    for i, y in ((1, 1.10), (2, .70), (3, .50), (4, .08))],
            "red": [("Cylinder_Red_07", "left", [.12, .80], "left"),
                    ("Cylinder_Red_05", "left", [.12, .40], "left"),
                    ("Cylinder_Red_06", "left", [.12, .68], "left")],
            "yellow": [("Cylinder_Yellow_11", "left", [.12, .94], "left"),
                       ("Cylinder_Yellow_01", "left", [.12, .10], "left"),
                       ("Cylinder_Yellow_02", "left", [.12, .17], "left")],
            "green": [("Cylinder_Green_10", "right", [1.08, 1.03], "right"),
                      ("Cylinder_Green_04", "right", [1.08, .88], "right"),
                      ("Cylinder_Green_09", "right", [1.08, .76], "right")],
        }
        if self.key == "green" and getattr(self.sim, "green_upper_first", False):
            tasks["green"] = [("Cylinder_Green_10", "right", [1.08, 1.03], "right"),
                              ("Cylinder_Green_09", "right", [1.08, .88], "right"),
                              ("Cylinder_Green_04", "right", [1.08, .76], "right")]
        return tasks[self.key]

    def work(self, count=None):
        if self.key == "kit":
            yield from self.distribute_preloaded_kits()
            return
        if self.key == "lab":
            self.sim.extension_targets[self.key] = .04
            yield from self.hold(1.2)
        for name, side, target, destination_side in self.tasks()[:count]:
            if self.key in ("red", "yellow"):
                self.phase = "wait for KIT to clear upper west lane"
                while "kit" in self.peers and self.peers["kit"].jobs_done < 2:
                    yield from self.tick()
            if self.key == "yellow" and self.jobs_done == 0:
                self.phase = "wait for RED to leave upper delivery"
                while "red" in self.peers and (self.peers["red"].jobs_done < 1 or self.sim.pose("red")[0][1] > .70):
                    yield from self.tick()
            if self.key == "yellow" and self.jobs_done == 1:
                self.phase = "wait for RED to clear west lane"
                while "red" in self.peers and self.peers["red"].phase != "finished":
                    yield from self.tick()
            if self.key == "green" and name == "Cylinder_Green_04":
                self.phase = "wait for LAB to clear lower right lane"
                while "lab" in self.peers and self.peers["lab"].phase != "finished":
                    yield from self.tick()
            yield from self.pickup(name, side)
            yield from self.deliver(name, target, destination_side)
            self.jobs_done += 1
        self.phase = "park clear of transit"
        if self.key == "red":
            yield from self.line([.225, .700])
            yield from self.line([.385, .700], heading=0.)
        elif self.key == "green":
            yield from self.line([.830, 1.100])
        if self.key == "lab":
            self.sim.extension_targets[self.key] = 0.
            yield from self.line([.830, .075])
            yield from self.line([1.02, .075])
        self.phase = "finished"
        yield from self.hold(.2)

    def distribute_preloaded_kits(self):
        """Three real floor gates unload four supported preloads by gravity."""
        for gate, y, names in ((0, .60, ["Medical_Kit_01", "Medical_Kit_02"]),
                               (1, 1.035, ["Medical_Kit_03"]),
                               (2, .20, ["Medical_Kit_04"])):
            self.phase = f"unload kit gate {gate}"
            yield from self.corridor("left", y)
            yield from self.turn(math.pi)
            yield from self.line([.141, y], heading=math.pi)
            self.sim.gate_targets[self.key][gate] = math.pi / 2
            yield from self.hold(1.5)
            for name in names:
                self.events.append({"event": "release", "object": name,
                                    "time_s": float(self.sim.data.time),
                                    "position_m": self.position(name).tolist()})
            yield from self.line([.225, y], heading=math.pi)
            self.jobs_done += 1
        self.phase = "park through bottom and center aisle"
        yield from self.line([.225, .075])
        yield from self.line([.550, .075])
        yield from self.line([.550, .5905])
        self.phase = "finished"
        yield from self.hold(.2)


def run(*, seed=0, randomize=False, output=None, only=None, task_count=None,
        maximum_seconds=120., time_penalty_per_second=.5, drive_limits=(.35, 2.5),
        drive_policy=None, lab_drive_limits=None, green_upper_first=False):
    if not 0 <= time_penalty_per_second < 100 / 120:
        raise ValueError("Time cost must stay below the reward for one 10-point delivery over a full match")
    source_hashes = mission_sources()
    overrides = {"lab": lab_drive_limits} if lab_drive_limits is not None else None
    sim = FleetSimulation(seed=seed, randomize=randomize, only=only, drive_limits=drive_limits,
                          robot_drive_limits=overrides)
    sim.green_upper_first = green_upper_first
    sim.metadata["source_sha256"] = source_hashes
    if not sim.setup["valid"]:
        raise ValueError(sim.setup["errors"])
    policy_record = None
    if drive_policy is not None:
        from .best_drive import BestDriveController, BestDrivePolicy
        weights = Path(drive_policy)
        if weights.is_dir():
            weights = weights / "weights.npz"
        training_path = weights.parent / "training.json"
        training = json.loads(training_path.read_text())
        weight_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
        if training["artifact_sha256"] != weight_hash:
            raise ValueError("Drive weights differ from their training report")
        learned = BestDrivePolicy(weights)
        controllers = {key: BestDriveController(learned, key) for key in sim.robots}
        for controller in controllers.values():
            controller.reset(sim)
        policy_record = {"weights_sha256": weight_hash,
                         "training_report_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
                         "geometric_precision_distance_m": .030,
                         "geometric_precision_heading_rad": .08}
    else:
        controllers = {}
    drivers = {key: Driver(sim, key, controllers.get(key)) for key in sim.robots}
    for driver in drivers.values():
        driver.peers = drivers
    began = time.perf_counter()
    failures = {}
    completed = []
    traffic_trace = []
    # Serialize departure through the narrow start cells. The mission work then
    # runs concurrently under the same physics clock.
    sequencer = DepartureSequencer(sim.robots)

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
        if round(sim.data.time / sim.control_dt) % 50 == 0:
            traffic_trace.append({"time_s": float(sim.data.time), "robots": {key: {"xy": sim.pose(key)[0].tolist(), "phase": driver.phase, "jobs_done": driver.jobs_done} for key, driver in drivers.items()}})

    programs = {}
    def program(driver):
        yield from sequencer.depart(driver)

        yield from driver.work(task_count)
    for key, driver in drivers.items():
        programs[key] = program(driver)
    score_stop = False
    while programs and sim.data.time < maximum_seconds:
        advance(programs)
        if round(sim.data.time / sim.control_dt) % 5 == 0:
            current_score = sim.score()
            if current_score["score"] == 160 and current_score["official_success"]:
                score_stop = True
                break
    deployment_time = max(sequencer.completed.values(), default=0.)
    declaration = float(sim.data.time)
    score = sim.score()
    for key, driver in drivers.items():
        sim.command(key, 0., 0., driver.lift, driver.jaw)
    hold_checks = []
    for index in range(250):
        sim.step(record=True)
        if (index + 1) % 5 == 0:
            check = sim.score()
            hold_checks.append({"time_s": float(sim.data.time), "task_score": check["task_score"],
                                "score": check["score"], "robot_currently_outside": check["robot_currently_outside"],
                                "physics_warning_count": check["physics_warning_count"]})
    hold_violations = [check for check in hold_checks if check["score"] != 160 or
                       check["robot_currently_outside"] or check["physics_warning_count"]]
    settled_score = sim.score()
    report = {"seed": seed, "randomized_physics": randomize, "initial_setup": sim.setup,
              "policy": "learned drive with geometric precision docking" if drive_policy else "geometric teacher",
              "learned_drive": policy_record, "observations": "simulator poses",
              "controller_calls": {key: {"learned": driver.learned_calls, "geometric": driver.geometric_calls}
                                   for key, driver in drivers.items()},
              "requested_drive_limits": sim.metadata["requested_drive_limits"],
              "green_upper_first": green_upper_first,
              "deployment_seconds": deployment_time, "departed": sequencer.completed, "declaration_seconds": declaration,
              "completed_robots": completed,
              "unfinished_robots": [] if score_stop else list(programs),
              "programs_stopped_after_full_score": list(programs) if score_stop else [],
              "termination": "all160points_secured" if score_stop else ("time_limit" if programs else "programs_complete"),
              "failures": failures, "traffic_trace": traffic_trace,
              "five_second_hold": {"sample_interval_s": .1, "samples": len(hold_checks), "minimum_score": min(check["score"] for check in hold_checks), "violations": hold_violations},
              "events": {key: driver.events for key, driver in drivers.items()},
              "score_at_declaration": score, "score_after_five_seconds": settled_score,
              "official_time_pass": bool(score["official_success"]),
              "wall_seconds": time.perf_counter() - began, "maximum_torque_nm": sim.max_torque,
              "maximum_tilt_rad": sim.max_tilt, "success": False,
              "objective": {"score_reward_per_point": 10.,
                            "time_penalty_per_second": time_penalty_per_second,
                            "time_cost": -time_penalty_per_second * declaration,
                            "return": 10. * score["score"] - time_penalty_per_second * declaration,
                            "selection": "highest official score, then shortest declaration time"}}
    report["success"] = bool(report["official_time_pass"] and not failures and
                             (score_stop or not programs) and not hold_violations and
                             settled_score["task_score"] == 160)
    report["source_sha256"] = source_hashes
    report["sources_stable_during_run"] = source_hashes == mission_sources()
    report["success"] = report["success"] and report["sources_stable_during_run"]
    if output:
        sim.save(output, report)
    return report
