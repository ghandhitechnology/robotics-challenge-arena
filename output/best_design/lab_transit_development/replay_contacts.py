"""Read saved native states and inspect LAB/sample contact geometry."""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

directory = Path(sys.argv[1])
report = json.loads((directory / "report.json").read_text())
events = [event for event in report["events"]["lab"] if event["object"] == "Biological_Sample_01"]
grasp = next((event["time_s"] for event in events if event["event"] == "grasp"), None)
release = next((event["time_s"] for event in events if event["event"] == "release"), None)
# The programmed release lowers the lift for .75 s, then opens the jaws for .55 s.
# Exclude that intentional lowering from the carry-loss check.
carry_end = release - 1.3 if release is not None else 35.
model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
data = mujoco.MjData(model)
trace = np.load(directory / "trajectory.npz")
rows = []
raised = False
drop = None
separation = None
for index, time in enumerate(trace["time"]):
    if not 10 <= time <= 35:
        continue
    data.time = time
    data.qpos[:] = trace["qpos"][index]
    data.qvel[:] = trace["qvel"][index]
    mujoco.mj_forward(model, data)
    sample = data.body("Biological_Sample_01").xpos.copy()
    lab = data.body("fleet_lab_competition_robot").xpos.copy()
    tool = data.site("fleet_lab_gripper_center").xpos.copy()
    raised = raised or sample[2] > .03
    carrying = grasp is not None and grasp <= time < carry_end
    if carrying and raised and sample[2] < .02 and drop is None:
        drop = float(time)
    if carrying and np.linalg.norm(sample-tool) > .015 and separation is None:
        separation = float(time)
    contacts = []
    for contact in data.contact:
        names = [model.geom(int(gid)).name for gid in contact.geom]
        touches_sample = "Biological_Sample_01_geom" in names
        cross_robot = (any(name.startswith("fleet_lab_") for name in names)
                       and any(name.startswith("fleet_") and not name.startswith("fleet_lab_") for name in names))
        if touches_sample or cross_robot:
            contacts.append({"geoms": names, "distance_m": float(contact.dist)})
    rows.append({"time_s": float(time), "sample_xyz": sample.tolist(), "lab_xyz": lab.tolist(),
                 "kit_xyz": data.body("fleet_kit_competition_robot").xpos.tolist(),
                 "tool_xyz": tool.tolist(), "sample_tool_distance_m": float(np.linalg.norm(sample-tool)),
                 "contacts": contacts})
result = {"method": "Contact geometry replay from recorded qpos and qvel; no contact forces inferred.",
          "grasp_s": grasp, "release_s": release, "carry_window_end_s": carry_end,
          "first_height_below_20mm_while_carrying_s": drop,
          "first_sample_tool_separation_above_15mm_while_carrying_s": separation, "rows": rows}
(directory / "lab_contact_replay.json").write_text(json.dumps(result, indent=2) + "\n")
collisions = [(r["time_s"], c) for r in rows for c in r["contacts"]
              if any(n.startswith("fleet_") and not n.startswith("fleet_lab_") for n in c["geoms"])]
partners = {}
for time, contact in collisions:
    pair = " / ".join(sorted(contact["geoms"]))
    if pair not in partners:
        partners[pair] = {"first_s": time, "last_s": time, "samples": 0,
                          "minimum_distance_m": contact["distance_m"]}
    row = partners[pair]
    row["last_s"] = time
    row["samples"] += 1
    row["minimum_distance_m"] = min(row["minimum_distance_m"], contact["distance_m"])
focus = separation if separation is not None else drop
summary = {k:v for k,v in result.items() if k != "rows"}
summary.update(seed=report["seed"], success=report["success"], failures=report["failures"],
               source_sha256=report["source_sha256"], cross_robot_contact_pairs=partners,
               first_collision_state=next((r for r in rows if collisions and r["time_s"] == collisions[0][0]), None),
               loss_window=[r for r in rows if focus is not None and focus-.2 <= r["time_s"] <= focus+.1])
(directory / "lab_contact_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k:v for k,v in summary.items() if k not in ("source_sha256", "loss_window", "first_collision_state")}, indent=2))
