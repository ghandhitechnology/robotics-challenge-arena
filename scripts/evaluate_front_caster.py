#!/usr/bin/env python3
"""Native LAB front-caster candidate; preserves the baseline source and JSON."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena_mujoco import best_fleet, best_mission

BASE_BUILDER = best_fleet.add_compact_robot


def add_supported_robot(root, entry, shared):
    metadata = BASE_BUILDER(root, entry, shared)
    if entry["id"] != "lab" or "front_anti_tip_caster" in metadata:
        return metadata
    prefix = "fleet_lab_"
    base = root.find(f"worldbody/body[@name='{metadata['name']}']")
    rear = base.find(f"body[@name='{prefix}comp_caster_body']")
    front = copy.deepcopy(rear)
    for element in front.iter():
        for key, value in list(element.attrib.items()):
            if "caster" in value:
                element.set(key, value.replace("caster", "front_caster"))
    front.set("pos", "0 .020 -.0185")
    front.find("geom").set("mass", ".012")
    base.append(front)
    ET.SubElement(base, "geom", name=prefix + "robot_comp_front_caster_bracket", type="box",
                  pos="0 .020 -.0025", size=".008 .008 .0025", mass=".004", contype="4",
                  conaffinity="7", rgba=".55 .60 .66 1")
    ET.SubElement(base, "geom", name=prefix + "robot_comp_front_caster_housing", type="cylinder",
                  pos="0 .020 -.010", size=".008 .005", mass=".002", contype="4",
                  conaffinity="7", rgba=".55 .60 .66 1")
    contact = root.find("contact")
    for body in base.iter("body"):
        if body is not front:
            ET.SubElement(contact, "exclude", body1=front.get("name"), body2=body.get("name"))
    metadata["mass_kg"] += .018
    metadata["front_caster_candidate"] = {"local_y_m": .020, "ball_radius_m": .006, "floor_clearance_m": .0005,
                                          "added_mass_kg": .018, "motor_count_change": 0}
    return metadata


class SupportedFleet(best_fleet.FleetSimulation):
    def __init__(self, **kwargs):
        with patch.object(best_fleet, "add_compact_robot", add_supported_robot):
            super().__init__(**kwargs)
        self.metadata["prototype"] = "LAB 12 mm front anti-tip ball caster at y=20 mm, 0.5 mm floor clearance"


def run(output, full=False, seed=0, randomize=False):
    with patch.object(best_mission, "FleetSimulation", SupportedFleet):
        report = best_mission.run(output=output, only=None if full else ["lab"], seed=seed, randomize=randomize)
    print(json.dumps({key: report[key] for key in ("success", "declaration_seconds", "failures", "maximum_tilt_rad")}, indent=2))
    print("score", report["score_at_declaration"]["task_score"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output/best_design/front_caster_20mm_gap_lab")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize", action="store_true")
    run(**vars(parser.parse_args()))
