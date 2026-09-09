"""Export the miniature module and run repeatable native contact-grasp coupons."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena_mujoco.swarm_robot import DESIGN, add_swarm_robot


def scene(timestep=.001):
    root = ET.Element("mujoco", model="swarm_module_contact_coupon")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", timestep=str(timestep), integrator="implicitfast",
                  solver="Newton", cone="elliptic", iterations="50", impratio="10")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="1 1 .01",
                  friction=".3 .00001 .000001", rgba=".91 .91 .88 1")
    ET.SubElement(world, "light", pos=".1 -.15 .3", diffuse=".8 .8 .8", ambient=".3 .3 .3")
    ET.SubElement(world, "light", pos="-.15 .1 .2", diffuse=".5 .5 .5")
    return root, world


def contact_coupon(kind, timestep=.001):
    """Two free modules close, lift and translate one free wood object.

    The deterministic controller isolates mechanical capability. A passing
    coupon is evidence about these initial conditions, not a learned policy.
    """
    radius = {"disc": .028, "kit": .0125, "cylinder": .010}[kind]
    height = .005 if kind == "disc" else .020
    root, world = scene(timestep)
    body = ET.SubElement(world, "body", name="payload", pos=f"0 0 {height/2+.00005}")
    ET.SubElement(body, "freejoint", name="payload_free")
    ET.SubElement(body, "geom", name="payload_geom", type="box" if kind == "kit" else "cylinder",
                  size=f"{radius} {radius} {height/2}" if kind == "kit" else f"{radius} {height/2}",
                  density="600", friction=".3 .00001 .000001", solref=".004 1")
    robots = [add_swarm_robot(root, 0, (0, -radius-.030, .0072)),
              add_swarm_robot(root, 1, (0, radius+.030, .0072), math.pi)]
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    wheel_vel = [[int(model.joint(n).dofadr[0]) for n in s["wheel_joints"]] for s in robots]
    motors = [[model.actuator(n).id for n in s["motor_names"]] for s in robots]
    lifts = [model.actuator(s["lift_motor"]).id for s in robots]
    pads = [{model.geom(n).id for n in s["pad_geoms"]} for s in robots]
    payload_id, floor_id = model.geom("payload_geom").id, model.geom("floor").id
    bottom_min, bottom_max, both_pad_steps, floor_steps = math.inf, -math.inf, 0, 0
    carry_steps, peak_torque = 0, 0.
    rest_positions = []
    for step in range(round(29/timestep)):
        t = step*timestep
        carry = .014 if 2.5 < t < 20 else 0.
        for i in range(2):
            speed = -carry*(1 if i == 0 else -1)/DESIGN["wheel_radius_m"] - 3.
            if t >= 22:
                speed = 3. if t < 24 else 0.
            rotation = data.xmat[model.body(robots[i]["name"]).id].reshape(3, 3)
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            error = (i*math.pi-yaw+math.pi) % (2*math.pi)-math.pi
            differential = np.clip(8*error, -2, 2)*DESIGN["wheel_track_m"]/(2*DESIGN["wheel_radius_m"])
            for j, actuator in enumerate(motors[i]):
                wheel_target = speed + differential*(1 if j == 0 else -1)
                torque = np.clip(DESIGN["velocity_gain"]*(wheel_target-data.qvel[wheel_vel[i][j]]), -.002, .002)
                data.ctrl[actuator] = torque if t > .2 else 0.
            data.ctrl[lifts[i]] = np.clip((t-1)*.005, 0, .008) if t < 20 else max(0., .008-(t-20)*.005)
        peak_torque = max(peak_torque, float(np.max(np.abs(data.ctrl[np.asarray(motors)]))))
        mujoco.mj_step(model, data)
        if t >= 24:
            rest_positions.append(data.xpos[model.body("payload").id].copy())
        if 2.8 <= t <= 19.8:
            carry_steps += 1
            rotation = data.geom_xmat[payload_id].reshape(3, 3)
            vertical_radius = (np.abs(rotation[2]) @ np.array([radius, radius, height/2])
                               if kind == "kit" else height/2*abs(rotation[2, 2]) +
                               radius*math.sqrt(max(0., 1-rotation[2, 2]**2)))
            bottom = float(data.geom_xpos[payload_id, 2]-vertical_radius)
            bottom_min, bottom_max = min(bottom_min, bottom), max(bottom_max, bottom)
            touched = set()
            floor_touch = False
            for contact in data.contact:
                pair = {int(contact.geom1), int(contact.geom2)}
                if payload_id not in pair:
                    continue
                floor_touch |= floor_id in pair
                for i, pad_ids in enumerate(pads):
                    if pad_ids & pair:
                        touched.add(i)
            both_pad_steps += len(touched) == 2
            floor_steps += floor_touch
    final = data.xpos[model.body("payload").id].copy()
    rest_drift = float(np.linalg.norm(np.ptp(np.asarray(rest_positions), axis=0)))
    result = dict(kind=kind, timestep_s=timestep, duration_s=29.,
                  minimum_bottom_during_carry_m=bottom_min,
                  maximum_bottom_during_carry_m=bottom_max,
                  translation_y_m=float(final[1]), final_position_m=final.tolist(),
                  both_modules_pad_contact_fraction=both_pad_steps/carry_steps,
                  payload_floor_contact_steps_during_carry=floor_steps,
                  maximum_wheel_torque_nm=peak_torque,
                  solver_warning_count=int(data.warning.number.sum()),
                  external_forces=False, object_attachments=False,
                  final_bottom_m=float(final[2]-height/2), rest_drift_m=rest_drift,
                  controller={"inward_wheel_target_rad_s": -3., "carry_velocity_m_s": .014,
                              "yaw_gain_s_inv": 8., "max_yaw_rate_rad_s": 2.,
                              "carry_interval_s": [2.5, 20.], "lower_interval_s": [20., 21.6],
                              "backoff_interval_s": [22., 24.], "rest_interval_s": [24., 29.]},
                  initial_module_poses=[[0, -radius-.030, .0072, 0], [0, radius+.030, .0072, math.pi]])
    result["passed"] = bool(bottom_min > .006 and final[1] > .20 and abs(final[2]-height/2) < .0002 and
                            both_pad_steps == carry_steps and floor_steps == 0 and rest_drift < .0001 and
                            not data.warning.number.any() and np.isfinite(data.qpos).all())
    return result


def export(out, render=False):
    out.mkdir(parents=True, exist_ok=True)
    root, _ = scene()
    robot = add_swarm_robot(root, 0, (0, 0, .007))
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geoms = []
    for gid in range(model.ngeom):
        if model.geom(gid).name == "floor":
            continue
        entry = dict(name=model.geom(gid).name, type=int(model.geom_type[gid]),
                     size_m=model.geom_size[gid].tolist(), position_m=data.geom_xpos[gid].tolist(),
                     rotation_matrix=data.geom_xmat[gid].reshape(3, 3).tolist(),
                     rgba=model.geom_rgba[gid].tolist())
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = model.geom_dataid[gid]
            start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
            vertices = model.mesh_vert[start:start+count]
            entry["vertices_world_m"] = (vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]).tolist()
            start, count = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
            entry["triangles"] = model.mesh_face[start:start+count].tolist()
        geoms.append(entry)
    robot["center_of_mass_body_m"] = (data.subtree_com[model.body(robot["name"]).id]-np.array([0, 0, .007])).tolist()
    (out/"robot.xml").write_text(xml+"\n")
    (out/"geometry.json").write_text(json.dumps(dict(units="meters", robot=robot, geoms=geoms), indent=2)+"\n")
    (ROOT/"swarm_robot_design.json").write_text(json.dumps(robot, indent=2)+"\n")
    if render:
        from PIL import Image
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = [0, 0, .01]
        camera.distance, camera.azimuth, camera.elevation = .14, 135, -30
        # Set framebuffer dimensions before constructing the renderer.
        model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
        with mujoco.Renderer(model, height=720, width=960) as renderer:
            renderer.update_scene(data, camera=camera)
            Image.fromarray(renderer.render()).save(out/"module.png")
    return out/"robot.xml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"output/swarm/robot")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    print(export(args.out, args.render))
    if args.validate:
        results = [contact_coupon(kind, timestep) for timestep in (.001, .002)
                   for kind in ("cylinder", "kit", "disc")]
        (args.out/"contact_validation.json").write_text(json.dumps(results, indent=2)+"\n")
        print(json.dumps(results, indent=2))
        if not all(item["passed"] for item in results):
            raise SystemExit("A contact coupon failed")


if __name__ == "__main__":
    main()
