"""Unmeasured material priors, explicitly separated from arena geometry."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import numpy as np


DEFAULT_PROFILE = {
    "schema_version": 1,
    "status": "Uncalibrated priors; replace with measurements from the actual kit.",
    "timestep_s": 0.0002,
    "wood_density_kg_m3": 600.0,
    "gravity_m_s2": 9.81,
    "contact": {"timeconst_s": 0.001, "dampratio": 1.0,
                "impedance": [0.98, 0.999, 0.0001], "slip_speed_m_s": 0.002},
    "materials": {
        "wood_paper": [0.55, 0.43, 0.0004, 0.00002],
        "wood_vinyl": [0.42, 0.32, 0.0004, 0.00002],
        "wood_wood": [0.50, 0.38, 0.0004, 0.00002],
        "rubber_paper": [0.95, 0.75, 0.0002, 0.00003],
        "rubber_vinyl": [0.85, 0.65, 0.0002, 0.00003],
        "rubber_wood": [0.85, 0.65, 0.0002, 0.00003],
        "plastic_paper": [0.35, 0.25, 0.0002, 0.00001],
        "plastic_vinyl": [0.30, 0.22, 0.0002, 0.00001],
        "plastic_wood": [0.35, 0.25, 0.0002, 0.00001],
        "vinyl_vinyl": [0.40, 0.30, 0.0001, 0.00001],
    },
    "wall_friction_scale": {"west": 1.0, "east": 1.0, "south": 1.0, "north": 1.0},
    "tape": {"spacing_m": 0.01, "joins": "butt", "density_kg_m3": 1350.0,
             "young_pa": 1.0e7, "poisson": 0.45, "rayleigh_damping_s": 1.0e-5,
             "rest_curvature_m_inv": 0.0, "edge_lift_m": 0.0,
             "peak_traction_pa": 30000.0, "fracture_energy_j_m2": 120.0,
             "damage_onset_m": 0.00005, "shear_weight": 0.5,
             "adhesive_friction": 1.0, "allow_rebond": False},
    "robot": {},
    "randomization": {"density_fraction": 0.15, "friction_fraction": 0.20,
                      "adhesion_fraction": 0.30, "young_fraction": 0.25,
                      "motor_fraction": 0.10},
}


def _merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def load_profile(profile=None, *, seed=None, randomize=False):
    result = copy.deepcopy(DEFAULT_PROFILE)
    if isinstance(profile, (str, Path)):
        profile = json.loads(Path(profile).read_text())
    if profile:
        _merge(result, profile)
    if randomize:
        rng = np.random.default_rng(seed)
        r = result["randomization"]
        scale = lambda fraction: float(rng.uniform(1-fraction, 1+fraction))
        result["wood_density_kg_m3"] *= scale(r["density_fraction"])
        for values in result["materials"].values():
            factor = scale(r["friction_fraction"])
            values[:] = [v*factor for v in values]
        result["tape"]["peak_traction_pa"] *= scale(r["adhesion_fraction"])
        result["tape"]["fracture_energy_j_m2"] *= scale(r["adhesion_fraction"])
        result["tape"]["young_pa"] *= scale(r["young_fraction"])
        result["robot"]["motor_strength_scale"] = scale(r["motor_fraction"])
    if not 0 < result["timestep_s"] <= 0.001:
        raise ValueError("timestep_s must be in (0, 0.001]")
    for name, values in result["materials"].items():
        if len(values) != 4 or not all(np.isfinite(values)) or min(values) < 0 or values[0] < values[1]:
            raise ValueError(f"{name}: expected [mu_static >= mu_kinetic, torsion_m, rolling_m]")
    return result


def material_pair(profile, first, second):
    table = profile["materials"]
    return table.get(f"{first}_{second}", table.get(f"{second}_{first}", table["wood_wood"]))


def friction5(values):
    return [values[0], values[0], values[2], values[3], values[3]]
