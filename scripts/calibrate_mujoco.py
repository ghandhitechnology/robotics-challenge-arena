#!/usr/bin/env python3
"""Convert kit measurements in JSON into a MuJoCo profile patch and report."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena_mujoco.materials import load_profile


TEMPLATE = {
    "schema_version": 1,
    "conditions": {"date": None, "temperature_c": None, "humidity_percent": None,
                   "wood_finish": None, "map_paper": None, "tape_product": None,
                   "tape_application_dwell_s": None, "surface_cleaning": None},
    "contacts": [
        {"pair": pair, "static_incline_deg": None,
         "kinetic_pull": {"mass_kg": None, "force_n": None}}
        for pair in ("wood_paper", "wood_vinyl", "rubber_paper", "rubber_vinyl", "wood_wood")
    ],
    "walls": [{"side": side, "pair": "wood_wood", "static_incline_deg": None,
               "kinetic_pull": {"normal_force_n": None, "force_n": None}}
              for side in ("west", "east", "south", "north")],
    "densities": [
        {"material": "wood", "label": "cylinder", "mass_kg": None,
         "geometry": {"shape": "cylinder", "diameter_m": 0.020, "height_m": 0.020}},
        {"material": "wood", "label": "laboratory", "mass_kg": None,
         "geometry": {"shape": "perforated_plate", "size_m": [0.150, 0.345, 0.003],
                      "holes": [{"diameter_m": 0.060, "count": 3}]}},
        {"material": "vinyl", "label": "tape coupon", "mass_kg": None,
         "geometry": {"shape": "box", "size_m": [0.020, 1.0, 0.00015]}},
    ],
    "drops": [{"label": "wood cylinder on paper", "drop_height_m": None,
               "rebound_height_m": None}],
    "tape": {
        "tensile": {"width_m": 0.020, "thickness_m": 0.00015, "gauge_length_m": 0.100,
                    "samples": [{"extension_m": 0.0, "force_n": 0.0},
                                {"extension_m": None, "force_n": None}]},
        "curl_radius_m": None, "curl_sign": 1,
        "peel": [{"angle_deg": angle, "width_m": 0.020, "force_n": None,
                  "speed_m_s": None, "support": "actual map paper"} for angle in (90, 180)],
    },
    "robot": {"max_motor_torque_nm": None, "max_wheel_speed_rad_s": None,
              "motor_no_load_speed_rad_s": None, "velocity_gain": None},
}


def number(value, label, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError(f"{label}: expected a finite {'nonnegative' if zero else 'positive'} value")
    return value


def average(value, label, *, zero=False):
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError(f"{label}: measurement list is empty")
    return statistics.mean(number(v, label, zero=zero) for v in values)


def volume_m3(geometry):
    """Actual material volume, including through-hole removal for the lab."""
    shape = geometry["shape"]
    if shape == "cylinder":
        diameter = number(geometry["diameter_m"], "diameter_m")
        return math.pi * diameter**2 / 4 * number(geometry["height_m"], "height_m")
    if shape not in ("box", "perforated_plate"):
        raise ValueError(f"Unsupported density geometry {shape!r}")
    size = geometry["size_m"]
    if len(size) != 3:
        raise ValueError("size_m must contain three dimensions")
    size = [number(v, "size_m") for v in size]
    volume = math.prod(size)
    for hole in geometry.get("holes", []) if shape == "perforated_plate" else []:
        diameter = number(hole["diameter_m"], "hole diameter_m")
        count = number(hole.get("count", 1), "hole count")
        if int(count) != count or diameter > min(size[:2]):
            raise ValueError("Hole count must be integral and diameter must fit the plate")
        depth = number(hole.get("depth_m", size[2]), "hole depth_m")
        if depth > size[2]:
            raise ValueError("Hole depth exceeds plate thickness")
        volume -= math.pi * diameter**2 / 4 * depth * count
    if volume <= 0:
        raise ValueError("Removed holes leave no positive material volume")
    return volume


def contact_coefficients(record, gravity):
    measured = {}
    if record.get("static_incline_deg") is not None:
        angles = record["static_incline_deg"]
        angles = angles if isinstance(angles, list) else [angles]
        if not angles:
            raise ValueError("static_incline_deg is empty")
        if any(number(a, "static_incline_deg", zero=True) >= 90 for a in angles):
            raise ValueError("Incline angle must be below 90 degrees")
        measured["mu_static"] = statistics.mean(math.tan(math.radians(a)) for a in angles)
    pull = record.get("kinetic_pull", {})
    if pull.get("force_n") is not None:
        if pull.get("normal_force_n") is not None:
            normal = average(pull["normal_force_n"], "normal_force_n")
        else:
            normal = average(pull.get("mass_kg"), "kinetic_pull.mass_kg") * gravity
        measured["mu_kinetic_pull"] = average(pull["force_n"], "force_n", zero=True) / normal
    coast = record.get("sliding_coast", {})
    if coast.get("distance_m") is not None:
        # A sliding coupon has translational energy only; rolling wheels require inertia.
        speed = number(coast.get("initial_speed_m_s"), "initial_speed_m_s")
        distances = coast["distance_m"] if isinstance(coast["distance_m"], list) else [coast["distance_m"]]
        if not distances:
            raise ValueError("sliding_coast.distance_m is empty")
        measured["mu_kinetic_coast"] = statistics.mean(
            speed**2 / (2 * gravity * number(d, "distance_m")) for d in distances)
    estimates = [value for key, value in measured.items() if key.startswith("mu_kinetic_")]
    if estimates:
        measured["mu_kinetic"] = statistics.mean(estimates)
    if "mu_static" in measured and measured.get("mu_kinetic", 0) > measured["mu_static"]:
        raise ValueError("Measured kinetic friction exceeds static friction; review the measurements")
    return measured


def calibrate(measurements, base_profile=None):
    """Return a loadable partial profile and a separate measurement derivation report."""
    if measurements.get("schema_version") != 1:
        raise ValueError("Expected measurement schema_version 1")
    base = load_profile(base_profile)
    gravity = number(base["gravity_m_s2"], "gravity_m_s2")
    if isinstance(base_profile, (str, Path)):
        patch = json.loads(Path(base_profile).read_text())
    else:
        patch = copy.deepcopy(base_profile) if base_profile else {}
    observations, skipped = [], []

    def record(kind, **values):
        observations.append({"kind": kind, **values})

    seen = set()
    for item in measurements.get("contacts", []):
        pair = item["pair"]
        if pair not in base["materials"] or pair in seen:
            raise ValueError(f"Unknown or duplicate material pair {pair!r}")
        seen.add(pair)
        coefficients = contact_coefficients(item, gravity)
        if not coefficients:
            skipped.append(f"contacts.{pair}")
            continue
        values = list(base["materials"][pair])
        for key, index in (("mu_static", 0), ("mu_kinetic", 1)):
            if key in coefficients:
                values[index] = coefficients[key]
        if values[1] > values[0]:
            raise ValueError(f"{pair}: measured value conflicts with retained static/kinetic prior; supply both measurements")
        patch.setdefault("materials", {})[pair] = values
        record("contact", pair=pair, **coefficients, retained_torsion_m=values[2], retained_rolling_m=values[3])

    calibrated_base = load_profile({**base, "materials": {**base["materials"], **patch.get("materials", {})}})
    seen.clear()
    for item in measurements.get("walls", []):
        side, pair = item["side"], item.get("pair", "wood_wood")
        if side not in base["wall_friction_scale"] or side in seen or pair not in base["materials"]:
            raise ValueError(f"Unknown or duplicate wall side/material pair: {side}/{pair}")
        seen.add(side)
        coefficients = contact_coefficients(item, gravity)
        if not coefficients:
            skipped.append(f"walls.{side}")
            continue
        key, index = ("mu_kinetic", 1) if "mu_kinetic" in coefficients else ("mu_static", 0)
        reference = calibrated_base["materials"][pair][index]
        if reference <= 0:
            raise ValueError(f"{side}: a zero base friction coefficient cannot be scaled")
        scale = coefficients[key] / reference
        patch.setdefault("wall_friction_scale", {})[side] = scale
        record("wall", side=side, pair=pair, **coefficients, friction_scale=scale, fitted_to=key,
               note="One wall multiplier scales all material coefficients, including torsion and rolling; validate each contacting material.")

    density_samples = {}
    for item in measurements.get("densities", []):
        if item.get("mass_kg") is None:
            skipped.append(f"densities.{item.get('label', item['material'])}")
            continue
        material = item["material"]
        if material not in ("wood", "vinyl"):
            raise ValueError("Density material must be wood or vinyl")
        mass = average(item["mass_kg"], "mass_kg")
        volume = volume_m3(item["geometry"])
        density_samples.setdefault(material, []).append((mass, volume))
        record("density", label=item.get("label", material), material=material, mass_kg=mass,
               volume_m3=volume, density_kg_m3=mass / volume)
    for material, samples in density_samples.items():
        density = sum(mass for mass, _ in samples) / sum(volume for _, volume in samples)
        if material == "wood":
            patch["wood_density_kg_m3"] = density
        else:
            patch.setdefault("tape", {})["density_kg_m3"] = density

    for item in measurements.get("drops", []):
        if item.get("drop_height_m") is None or item.get("rebound_height_m") is None:
            skipped.append(f"drops.{item.get('label', 'unlabeled')}")
            continue
        drop = number(item["drop_height_m"], "drop_height_m")
        rebound = average(item["rebound_height_m"], "rebound_height_m", zero=True)
        if rebound > drop:
            raise ValueError("Rebound height exceeds drop height")
        record("drop", label=item.get("label", "unlabeled"), restitution_estimate=math.sqrt(rebound / drop),
               note="Height-based estimate; fit simulated drops at multiple speeds before changing contact damping.")

    tape = measurements.get("tape", {})
    tensile = tape.get("tensile", {})
    samples = tensile.get("samples", [])
    if samples and all(p.get("extension_m") is not None and p.get("force_n") is not None for p in samples):
        if len(samples) < 2:
            raise ValueError("Tensile fit requires at least two samples")
        length = number(tensile["gauge_length_m"], "gauge_length_m")
        area = number(tensile["width_m"], "width_m") * number(tensile["thickness_m"], "thickness_m")
        x = [number(p["extension_m"], "extension_m", zero=True) for p in samples]
        y = [number(p["force_n"], "force_n", zero=True) for p in samples]
        if max(x) / length > 0.02:
            raise ValueError("Use a small-strain tensile window at or below 2%; inspect linearity before fitting")
        mx, my = statistics.mean(x), statistics.mean(y)
        variance = sum((v - mx)**2 for v in x)
        if variance == 0:
            raise ValueError("Tensile samples need distinct extension values")
        slope = number(sum((a - mx) * (b - my) for a, b in zip(x, y)) / variance, "tensile slope")
        modulus = slope * length / area
        residual = sum((b - (my + slope * (a - mx)))**2 for a, b in zip(x, y))
        total = sum((b - my)**2 for b in y)
        patch.setdefault("tape", {})["young_pa"] = modulus
        record("tensile", effective_young_pa=modulus, force_extension_slope_n_m=slope,
               force_intercept_n=my - slope * mx, r_squared=1 - residual / total,
               max_strain=max(x) / length, note="Effective homogeneous shell modulus; compare bending separately.")
    elif samples:
        skipped.append("tape.tensile")
    if tape.get("curl_radius_m") is not None:
        sign = tape.get("curl_sign", 1)
        if sign not in (-1, 1):
            raise ValueError("curl_sign must be -1 or 1")
        curvature = sign / number(tape["curl_radius_m"], "curl_radius_m")
        patch.setdefault("tape", {})["rest_curvature_m_inv"] = curvature
        record("curl", rest_curvature_m_inv=curvature)
    for item in tape.get("peel", []):
        if item.get("force_n") is None:
            skipped.append(f"tape.peel.{item.get('angle_deg', 'unknown')}")
            continue
        angle = number(item["angle_deg"], "peel angle_deg")
        if angle not in (90, 180):
            raise ValueError("Peel calibration supports measured 90 or 180 degree tests")
        normalized = average(item["force_n"], "peel force_n") / number(item["width_m"], "peel width_m")
        if item.get("speed_m_s") is not None:
            number(item["speed_m_s"], "peel speed_m_s")
        record("peel", angle_deg=angle, resistance_n_per_m=normalized,
               ideal_energy_estimate_j_m2=normalized * (1 - math.cos(math.radians(angle))),
               speed_m_s=item.get("speed_m_s"), support=item.get("support"),
               status="ideal_inextensible_backing_estimate",
               note="Peel force alone does not identify peak traction or a calibrated fracture-energy law; no bond parameter was updated.")
    allowed_robot = {"max_motor_torque_nm", "max_wheel_speed_rad_s", "motor_no_load_speed_rad_s", "velocity_gain"}
    for key, value in measurements.get("robot", {}).items():
        if key not in allowed_robot:
            raise ValueError(f"Unsupported robot calibration key {key!r}; update robot geometry/inertia in robot.py")
        if value is not None:
            patch.setdefault("robot", {})[key] = number(value, f"robot.{key}")
        else:
            skipped.append(f"robot.{key}")
    # The same validation used by the runtime catches invalid static/kinetic pairs.
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict):
            merged[key].update(value)
        else:
            merged[key] = value
    load_profile(merged)
    robot = merged.get("robot", {})
    if robot.get("motor_no_load_speed_rad_s", 25.0) < robot.get("max_wheel_speed_rad_s", 20.0):
        raise ValueError("motor_no_load_speed_rad_s must be at least max_wheel_speed_rad_s")
    return patch, {"schema_version": 1, "conditions": measurements.get("conditions", {}),
                   "observations": observations, "skipped_incomplete_measurements": skipped,
                   "retained_priors": "Unmeasured fields retain the supplied base profile; report contains estimates that were not applied."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-template", type=Path, help="Write blank measurement JSON and exit")
    parser.add_argument("--input", type=Path, help="Completed measurement JSON")
    parser.add_argument("--output", type=Path, help="Loadable profile patch JSON")
    parser.add_argument("--report", type=Path, help="Derivation report; default: OUTPUT.report.json")
    parser.add_argument("--base-profile", type=Path, help="Existing profile for retained parameters")
    args = parser.parse_args()
    if args.write_template:
        args.write_template.parent.mkdir(parents=True, exist_ok=True)
        args.write_template.write_text(json.dumps(TEMPLATE, ensure_ascii=False, indent=2) + "\n")
        print(f"Measurement template: {args.write_template}")
        return
    if args.input is None or args.output is None:
        parser.error("Provide --write-template PATH or both --input PATH and --output PATH")
    try:
        patch, report = calibrate(json.loads(args.input.read_text()), args.base_profile)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    report["base_profile"] = str(args.base_profile) if args.base_profile else "arena_mujoco.materials.DEFAULT_PROFILE"
    report["input"] = str(args.input)
    report_path = args.report or args.output.with_suffix(".report.json")
    if report_path.resolve() == args.output.resolve():
        parser.error("Profile and report must have different paths")
    for path, content in ((args.output, patch), (report_path, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Profile patch: {args.output}\nDerivation report: {report_path}")


if __name__ == "__main__":
    main()
