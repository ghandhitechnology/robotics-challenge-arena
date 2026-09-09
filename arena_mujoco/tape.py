"""Stateful underside bonds for native MuJoCo flex tape.

The builder supplies the PVC flex and one hidden spherical quadrature geom per
bond node. Explicit geom pairs provide adhesive forces through MuJoCo's contact
solver. This controller changes their tensile capacity; it adds no explicit
stiff spring force and gives no adhesion to the visible vinyl top.

Capacity has a plateau followed by linear softening in mixed opening/slip.
MuJoCo's soft contact supplies the near-zero-separation regularization. This is
not an exact bilinear traction/separation spring: ``2*Gc/sigma`` sets a nominal
failure length, and the reported damage/work energies are proxies, not a
verified thermodynamic balance or a calibrated peel-test result.

Each metadata['tape_nodes'] entry requires:
  body_name, pair_name, area_m2, rest_relative_m,
  support_body_name (defaults to 'world'), support_anchor_local_m.
``rest_relative_m`` is the initial node-centre offset from the support surface,
expressed in the support's initial local frame. For the floor it is [0,0,radius]
and the anchor is [world_x,world_y,0]. An interlayer anchor is the lower support
proxy's top surface in its body's local coordinates. Optional ``geom_name``,
``vertex_id`` and ``normal_default`` resolve ambiguous geometry. Triangle winding
is assumed consistent. Initial normals are oriented toward ``normal_default``.

Call initialize(data), then before_step(data), mj_step(model,data),
after_step(data). State evolves only after an accepted step. Each independent
environment needs its own model: pair adhesion/friction/gap are model arrays.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mujoco
import numpy as np


_DEFAULTS = {
    "peak_traction_pa": 30000.0,
    "fracture_energy_j_m2": 120.0,
    "shear_strength_pa": 30000.0,
    "shear_fracture_energy_j_m2": 120.0,
    "shear_weight": 1.0,
    "damage_onset_m": 0.0001,
    "capture_gap_m": 0.0001,
    "max_bond_gap_m": 0.008,
    "normal_alignment_min": 0.0,
    "rate_strength_coefficient": 0.0,
    "reference_speed_m_s": 0.01,
    "rate_strength_max_factor": 3.0,
    "initial_dwell_s": 0.0,
    "dwell_time_constant_s": 0.0,
    "dwell_min_strength_factor": 1.0,
    "contamination_fraction": 0.0,
    "wear_rate_per_j_m2": 0.0,
    "allow_rebond": False,
    "rebond_pressure_pa": 1000.0,
    "rebond_dwell_s": 1.0,
    "rebond_gap_m": 0.0001,
    "rebond_max_slip_speed_m_s": 0.001,
    "rebond_damage": 0.25,
}
_ALIASES = {
    "sigma_pa": "peak_traction_pa",
    "sigma": "peak_traction_pa",
    "Gc": "fracture_energy_j_m2",
    "gc_j_m2": "fracture_energy_j_m2",
    "tau_pa": "shear_strength_pa",
    "bond_gap_m": "max_bond_gap_m",
}


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite coordinates")
    return result


def _name_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    result = mujoco.mj_name2id(model, kind, name)
    if result < 0:
        raise ValueError(f"Unknown {kind.name} name: {name}")
    return result


def _normalize(vectors: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > 1e-14
    result = fallback.copy()
    result[valid] = vectors[valid] / lengths[valid, None]
    return result


class TapeController:
    """Manage irreversible floor/interlayer adhesive contact capacities.

    ``params`` may be a flat tape dictionary or a larger dictionary containing
    ``tape``. Unknown keys are retained outside this controller and ignored here,
    allowing the same dictionary to hold flex/material builder parameters.
    All built-in strength, rate, damage and wear coefficients are uncalibrated.
    """

    def __init__(self, model: mujoco.MjModel, metadata: Mapping[str, Any],
                 params: Mapping[str, Any] | None = None):
        self.model = model
        if not hasattr(model, "pair_adhesion"):
            raise RuntimeError("TapeController requires MuJoCo native pair adhesion")
        supplied = dict(params or {})
        supplied = dict(supplied.get("tape", supplied))
        for old, new in _ALIASES.items():
            if old in supplied and new not in supplied:
                supplied[new] = supplied[old]
        self.params = {key: supplied.get(key, value) for key, value in _DEFAULTS.items()}
        if "shear_strength_pa" not in supplied:
            self.params["shear_strength_pa"] = (float(supplied.get("adhesive_friction", 1.0))
                                                * float(self.params["peak_traction_pa"]))
        if "shear_fracture_energy_j_m2" not in supplied:
            self.params["shear_fracture_energy_j_m2"] = self.params["fracture_energy_j_m2"]
        self._validate_parameters()
        p = self.params
        self.failure_opening_m = 2 * p["fracture_energy_j_m2"] / p["peak_traction_pa"]
        self.failure_slip_m = 2 * p["shear_fracture_energy_j_m2"] / p["shear_strength_pa"]
        self._onset_ratio = p["damage_onset_m"] / self.failure_opening_m
        if not 0 <= self._onset_ratio < 1:
            raise ValueError("damage_onset_m must be below 2*Gc/peak_traction_pa")
        self.interaction_gap_m = max(
            p["capture_gap_m"], min(self.failure_opening_m, p["max_bond_gap_m"]),
        )
        entries = list(metadata.get("tape_nodes", []))
        self.count = len(entries)
        self.pair_names = [str(node["pair_name"]) for node in entries]
        if len(set(self.pair_names)) != self.count:
            raise ValueError("Every tape node must have a distinct bond pair")
        self.body_ids = np.array([
            _name_id(model, mujoco.mjtObj.mjOBJ_BODY, str(node["body_name"]))
            for node in entries
        ], dtype=int)
        self.pair_ids = np.array([
            _name_id(model, mujoco.mjtObj.mjOBJ_PAIR, name) for name in self.pair_names
        ], dtype=int)
        self.support_ids = np.array([
            0 if node.get("support_body_name", "world") == "world" else
            _name_id(model, mujoco.mjtObj.mjOBJ_BODY, str(node["support_body_name"]))
            for node in entries
        ], dtype=int)
        if np.any(self.support_ids == self.body_ids):
            raise ValueError("A tape node cannot bond to its own body")
        self.areas_m2 = np.array([float(node["area_m2"]) for node in entries])
        if not np.isfinite(self.areas_m2).all() or np.any(self.areas_m2 <= 0):
            raise ValueError("Bond quadrature areas must be finite and positive")
        self.rest_relative_m = np.array([
            _vector(node["rest_relative_m"], "rest_relative_m") for node in entries
        ]).reshape(-1, 3)
        self._initial_anchors = np.array([
            _vector(node["support_anchor_local_m"], "support_anchor_local_m")
            for node in entries
        ]).reshape(-1, 3)
        self._default_normals = np.array([
            _vector(node.get("normal_default", [0, 0, 1]), "normal_default")
            for node in entries
        ]).reshape(-1, 3)
        normal_lengths = np.linalg.norm(self._default_normals, axis=1)
        if np.any(normal_lengths <= 1e-14):
            raise ValueError("normal_default cannot be zero")
        self._default_normals /= normal_lengths[:, None]
        self._initial_damage = np.array([float(node.get("initial_damage", 0)) for node in entries])
        self._initial_contamination = np.array([
            float(node.get("contamination_fraction", p["contamination_fraction"])) for node in entries
        ])
        for values in (self._initial_damage, self._initial_contamination):
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValueError("Initial damage and contamination must be in [0,1]")

        self._body_vertex = np.full(model.nbody, -1, dtype=int)
        for body in np.unique(model.flex_vertbodyid):
            vertices = np.flatnonzero(model.flex_vertbodyid == body)
            if len(vertices) == 1:
                self._body_vertex[body] = vertices[0]
        self.vertex_ids = np.array([
            int(node.get("vertex_id", self._body_vertex[body]))
            for node, body in zip(entries, self.body_ids)
        ], dtype=int)
        if np.any(self.vertex_ids >= model.nflexvert) or np.any(self.vertex_ids < -1):
            raise ValueError("vertex_id is outside the model's global flex vertex array")
        for vertex, body in zip(self.vertex_ids, self.body_ids):
            if vertex >= 0 and model.flex_vertbodyid[vertex] != body:
                raise ValueError("vertex_id and body_name refer to different bodies")
        self._support_vertices = self._body_vertex[self.support_ids]
        faces, groups = [], []
        for flex in range(model.nflex):
            if model.flex_dim[flex] != 2:
                continue
            start, count = model.flex_elemdataadr[flex], model.flex_elemnum[flex]
            triangle = model.flex_elem[start:start + 3 * count].reshape(-1, 3)
            faces.extend(triangle + model.flex_vertadr[flex])
            groups.extend([flex] * count)
        self._triangles = np.asarray(faces, dtype=int).reshape(-1, 3)
        self._face_groups = np.asarray(groups, dtype=int)
        self._face_signs = np.ones(len(faces))
        self._first_face = np.full(model.nflexvert, -1, dtype=int)
        for face_index, vertices in enumerate(self._triangles):
            for vertex in vertices:
                if self._first_face[vertex] < 0:
                    self._first_face[vertex] = face_index
        self._pair_lookup: dict[tuple[int, int], int] = {}
        for index, pair in enumerate(self.pair_ids):
            g1, g2 = int(model.pair_geom1[pair]), int(model.pair_geom2[pair])
            node_geoms = [geom for geom in (g1, g2) if model.geom_bodyid[geom] == self.body_ids[index]]
            if len(node_geoms) != 1:
                raise ValueError(f"{self.pair_names[index]} must include its node body's proxy")
            node_geom = node_geoms[0]
            support_geom = g2 if node_geom == g1 else g1
            if model.geom_bodyid[support_geom] != self.support_ids[index]:
                raise ValueError(f"{self.pair_names[index]} does not use its declared support body")
            if int(model.geom_type[node_geom]) != int(mujoco.mjtGeom.mjGEOM_SPHERE):
                raise ValueError("Adhesive quadrature proxies must be spheres: force is per contact")
            if int(model.geom_type[support_geom]) not in (
                int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_SPHERE),
                int(mujoco.mjtGeom.mjGEOM_BOX),
            ):
                raise ValueError("Bond supports must be plane, box or spherical quadrature proxies")
            if node_geom != _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, str(
                entries[index].get("geom_name", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, node_geom)),
            )):
                raise ValueError("geom_name does not match the node's bond pair")
            if model.pair_dim[pair] < 3:
                raise ValueError("Adhesive bonds require condim >= 3 for shear resistance")
            key = tuple(sorted((g1, g2)))
            if key in self._pair_lookup:
                raise ValueError("Duplicate proxy contact pairs would duplicate adhesive force")
            self._pair_lookup[key] = index
        # MuJoCo derives this flag when compiling MJCF. Runtime pair changes
        # alone do not enable the passive attractive branch if XML started at 0.
        if self.count:
            self.model.flg_adhesion = True
        self.initialized = False
        self._reference_frames = np.tile(np.eye(3), (model.nflexvert, 1, 1))
        self._reference_support_rotations = np.tile(np.eye(3), (self.count, 1, 1))

    def _validate_parameters(self) -> None:
        for key, value in self.params.items():
            if key == "allow_rebond":
                if not isinstance(value, (bool, np.bool_)):
                    raise ValueError("allow_rebond must be a boolean")
                continue
            value = float(value)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
            self.params[key] = value
        for key in ("peak_traction_pa", "fracture_energy_j_m2", "shear_strength_pa",
                    "shear_fracture_energy_j_m2", "capture_gap_m", "max_bond_gap_m",
                    "reference_speed_m_s", "rebond_dwell_s"):
            if self.params[key] <= 0:
                raise ValueError(f"{key} must be positive")
        if self.params["max_bond_gap_m"] < self.params["capture_gap_m"]:
            raise ValueError("max_bond_gap_m must be at least capture_gap_m")
        for key in ("normal_alignment_min", "dwell_min_strength_factor",
                    "contamination_fraction", "rebond_damage"):
            if self.params[key] > 1:
                raise ValueError(f"{key} must be in [0,1]")
        if self.params["rate_strength_max_factor"] < 1:
            raise ValueError("rate_strength_max_factor must be at least one")

    def _refresh_positions(self, data: mujoco.MjData) -> None:
        # mj_step advances qpos after computing xpos. Refresh geometry only;
        # preserve the solved contact forces needed for the work diagnostic.
        mujoco.mj_kinematics(self.model, data)
        if self.model.nflex:
            mujoco.mj_flex(self.model, data)

    def _surface_frames(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        count = self.model.nflexvert
        fallback = np.tile([0.0, 0.0, 1.0], (count, 1))
        summed = np.zeros((count, 3))
        frames = np.tile(np.eye(3), (count, 1, 1))
        if not len(self._triangles):
            return fallback, frames
        points = data.flexvert_xpos[self._triangles]
        edge = points[:, 1] - points[:, 0]
        face_normals = np.cross(edge, points[:, 2] - points[:, 0]) * self._face_signs[:, None]
        np.add.at(summed, self._triangles.ravel(), np.repeat(face_normals, 3, axis=0))
        normals = _normalize(summed, fallback)
        has_face = self._first_face >= 0
        tangent = np.tile([1.0, 0.0, 0.0], (count, 1))
        tangent[has_face] = edge[self._first_face[has_face]]
        tangent -= np.sum(tangent * normals, axis=1)[:, None] * normals
        fallback_tangent = np.cross(np.tile([0.0, 1.0, 0.0], (count, 1)), normals)
        degenerate = np.linalg.norm(fallback_tangent, axis=1) < 1e-12
        fallback_tangent[degenerate] = [1, 0, 0]
        fallback_tangent = _normalize(fallback_tangent, np.tile([1., 0., 0.], (count, 1)))
        tangent = _normalize(tangent, fallback_tangent)
        frames[:, :, 0] = tangent
        frames[:, :, 1] = np.cross(normals, tangent)
        frames[:, :, 2] = normals
        return normals, frames

    def _geometry(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        self._refresh_positions(data)
        normals, frames = self._surface_frames(data)
        node_pos = data.xpos[self.body_ids].copy()
        node_flex = self.vertex_ids >= 0
        node_pos[node_flex] = data.flexvert_xpos[self.vertex_ids[node_flex]]
        rotations = data.xmat[self.support_ids].reshape(-1, 3, 3).copy()
        support_flex = self._support_vertices >= 0
        vertices = self._support_vertices[support_flex]
        rotations[support_flex] = (
            frames[vertices] @ self._reference_frames[vertices].transpose(0, 2, 1)
            @ self._reference_support_rotations[support_flex]
        )
        anchors = data.xpos[self.support_ids] + np.einsum("nij,nj->ni", rotations, self.anchors_local_m)
        offset = np.einsum("nij,nj->ni", rotations, self.rest_relative_m)
        support_normal = np.einsum("nij,nj->ni", rotations, self._default_normals)
        support_normal[support_flex] = normals[vertices]
        node_normal = np.einsum("nij,nj->ni", data.xmat[self.body_ids].reshape(-1, 3, 3), self._default_normals)
        node_normal[node_flex] = normals[self.vertex_ids[node_flex]]
        displacement = node_pos - anchors - offset
        normal_component = np.sum(displacement * support_normal, axis=1)
        opening = np.maximum(normal_component, 0)
        slip = displacement - normal_component[:, None] * support_normal
        alignment = np.sum(node_normal * support_normal, axis=1)
        return {"opening": opening, "slip": slip, "alignment": alignment,
                "anchors": anchors, "rotations": rotations, "normal": support_normal}

    def initialize(self, data: mujoco.MjData) -> None:
        """Reset episode history at the builder's initialized tape pose."""
        self._refresh_positions(data)
        self._face_signs[:] = 1
        if len(self._triangles):
            pts = data.flexvert_xpos[self._triangles]
            face_normals = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
            for flex in np.unique(self._face_groups):
                mask = self._face_groups == flex
                vertices = np.unique(self._triangles[mask])
                nodes = np.isin(self.vertex_ids, vertices)
                expected = self._default_normals[nodes].sum(axis=0) if np.any(nodes) else np.array([0, 0, 1])
                if np.dot(face_normals[mask].sum(axis=0), expected) < 0:
                    self._face_signs[mask] = -1
        _, self._reference_frames = self._surface_frames(data)
        self._reference_support_rotations = data.xmat[self.support_ids].reshape(-1, 3, 3).copy()
        self.anchors_local_m = self._initial_anchors.copy()
        self.damage = self._initial_damage.copy()
        self.max_separation_ratio = np.where(
            self.damage > 0, self._onset_ratio + (1 - self._onset_ratio) * self.damage, 0,
        )
        self.wear = np.zeros(self.count)
        self.contamination = self._initial_contamination.copy()
        self.bond_age_s = np.full(self.count, self.params["initial_dwell_s"])
        self.rebond_dwell_s = np.zeros(self.count)
        self.separation_speed_m_s = np.zeros(self.count)
        self.slip_speed_m_s = np.zeros(self.count)
        self.damage_energy_proxy_j = 0.0
        self.contact_work_proxy_j = 0.0
        self.rebond_count = 0
        self.reach_limited = np.zeros(self.count, dtype=bool)
        self.orientation_released = np.zeros(self.count, dtype=bool)
        self._last_time = float(data.time)
        state = self._geometry(data)
        self._last_opening = state["opening"].copy()
        self._last_slip = state["slip"].copy()
        self._enabled = self._orientation_enabled(state)
        self._last_pressure_pa = np.zeros(self.count)
        self.initialized = True
        self.before_step(data)

    def _orientation_enabled(self, state: Mapping[str, np.ndarray]) -> np.ndarray:
        return state["alignment"] > self.params["normal_alignment_min"]

    def _separation_damage(self, state: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        ratio = np.hypot(state["opening"] / self.failure_opening_m,
                         self.params["shear_weight"] * np.linalg.norm(state["slip"], axis=1)
                         / self.failure_slip_m)
        history = np.maximum(self.max_separation_ratio, ratio)
        damage = np.maximum(self.damage, np.clip(
            (history - self._onset_ratio) / (1 - self._onset_ratio), 0, 1,
        ))
        return history, damage

    def _strength_factor(self) -> np.ndarray:
        p = self.params
        rate = np.minimum(p["rate_strength_max_factor"], 1 + p["rate_strength_coefficient"]
                          * np.log1p(self.separation_speed_m_s / p["reference_speed_m_s"]))
        if p["dwell_time_constant_s"] > 0:
            dwell = 1 - (1 - p["dwell_min_strength_factor"]) * np.exp(
                -self.bond_age_s / p["dwell_time_constant_s"],
            )
        else:
            dwell = np.ones(self.count)
        return rate * dwell * (1 - self.contamination) * (1 - self.wear)

    def _write_parameters(self, damage: np.ndarray, enabled: np.ndarray) -> None:
        factor = self._strength_factor()
        alive = enabled & (damage < 1)
        capacity = self.params["peak_traction_pa"] * self.areas_m2 * (1 - damage) * factor * alive
        self.model.pair_adhesion[self.pair_ids] = capacity
        # The two sliding entries are dimensionless. No extra torsion or rolling
        # at a quadrature point; the distribution of point forces supplies torque.
        friction = np.zeros((self.count, 5))
        friction[:, :2] = (self.params["shear_strength_pa"] / self.params["peak_traction_pa"]
                           * (capacity > 0))[:, None]
        self.model.pair_friction[self.pair_ids] = friction
        self.model.pair_gap[self.pair_ids] = np.where(alive, self.interaction_gap_m, 0.0)

    def before_step(self, data: mujoco.MjData) -> None:
        """Apply capacities without advancing damage, dwell, wear or time."""
        if not self.initialized:
            self.initialize(data)
            return
        state = self._geometry(data)
        _, prospective_damage = self._separation_damage(state)
        enabled = self._orientation_enabled(state) & (state["opening"] < self.interaction_gap_m)
        self._write_parameters(prospective_damage, enabled)

    def _contact_diagnostics(self, data: mujoco.MjData,
                             opening_increment: np.ndarray, slip_increment: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pressure = np.zeros(self.count)
        work = np.zeros(self.count)
        shear_work = np.zeros(self.count)
        force = np.zeros(6)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            key = tuple(sorted((int(contact.geom[0]), int(contact.geom[1]))))
            index = self._pair_lookup.get(key)
            if index is None or contact.efc_address < 0:
                continue
            mujoco.mj_contactForce(self.model, data, contact_index, force)
            pressure[index] += max(float(force[0]), 0) / self.areas_m2[index]
            increment = np.linalg.norm(force[1:3]) * np.linalg.norm(slip_increment[index])
            shear_work[index] += increment
            work[index] += max(-float(force[0]), 0) * max(opening_increment[index], 0) + increment
        return pressure, work, shear_work

    def after_step(self, data: mujoco.MjData) -> None:
        """Commit one accepted step and record explicitly approximate work."""
        if not self.initialized:
            raise RuntimeError("initialize must precede after_step")
        dt = float(data.time) - self._last_time
        if dt < -1e-12:
            raise RuntimeError("Simulation time moved backwards; call initialize after reset")
        if dt <= 1e-12:
            return
        state = self._geometry(data)
        opening_increment = state["opening"] - self._last_opening
        slip_increment = state["slip"] - self._last_slip
        self.slip_speed_m_s = np.linalg.norm(slip_increment, axis=1) / dt
        self.separation_speed_m_s = np.hypot(np.maximum(opening_increment, 0) / dt,
                                            self.slip_speed_m_s)
        pressure, work, shear_work = self._contact_diagnostics(data, opening_increment, slip_increment)
        self._last_pressure_pa = pressure
        self.contact_work_proxy_j += float(work.sum())
        self.wear = np.clip(self.wear + self.params["wear_rate_per_j_m2"]
                            * shear_work / self.areas_m2, 0, 1)
        old_damage = self.damage.copy()
        self.max_separation_ratio, self.damage = self._separation_damage(state)
        orientation = self._orientation_enabled(state)
        out_of_range = state["opening"] >= self.interaction_gap_m
        self.reach_limited |= out_of_range & (self.interaction_gap_m < self.failure_opening_m)
        self.orientation_released |= ~orientation
        self.damage[~orientation | out_of_range] = 1
        self.damage_energy_proxy_j += float(np.sum(
            np.maximum(self.damage - old_damage, 0) * self.params["fracture_energy_j_m2"] * self.areas_m2,
        ))
        seated = orientation & (state["opening"] <= self.params["capture_gap_m"])
        self.bond_age_s += dt * (seated & (self.damage < 1))
        if self.params["allow_rebond"]:
            eligible = ((self.damage >= 1) & orientation
                        & (state["opening"] <= self.params["rebond_gap_m"])
                        & (pressure >= self.params["rebond_pressure_pa"])
                        & (self.slip_speed_m_s <= self.params["rebond_max_slip_speed_m_s"]))
            self.rebond_dwell_s = np.where(eligible, self.rebond_dwell_s + dt, 0)
            rebond = self.rebond_dwell_s >= self.params["rebond_dwell_s"]
            if np.any(rebond):
                # Reattach where the surface was pressed down, retaining wear.
                local_slip = np.einsum("nji,nj->ni", state["rotations"], state["slip"])
                self.anchors_local_m[rebond] += local_slip[rebond]
                self.damage[rebond] = self.params["rebond_damage"]
                self.max_separation_ratio[rebond] = (self._onset_ratio
                    + (1 - self._onset_ratio) * self.params["rebond_damage"])
                self.bond_age_s[rebond] = 0
                self.rebond_dwell_s[rebond] = 0
                self.rebond_count += int(rebond.sum())
                state = self._geometry(data)
        self._last_opening = state["opening"].copy()
        self._last_slip = state["slip"].copy()
        self._last_time = float(data.time)
        self._enabled = self._orientation_enabled(state) & (state["opening"] < self.interaction_gap_m)
        self._write_parameters(self.damage, self._enabled)

    def metrics(self) -> dict[str, float | int]:
        """Return area-weighted bond health and labeled diagnostic proxies."""
        if not self.initialized:
            raise RuntimeError("initialize must precede metrics")
        area = float(self.areas_m2.sum())
        mean = lambda value: float(np.dot(value, self.areas_m2) / area) if area else 0.0
        return {
            "tape_bond_count": self.count,
            "tape_bonded_count": int(np.count_nonzero((self.damage < 1) & self._enabled)),
            "tape_bonded_area_m2": float(self.areas_m2[(self.damage < 1) & self._enabled].sum()),
            "tape_damage_fraction": mean(self.damage),
            "tape_wear_fraction": mean(self.wear),
            "tape_max_opening_m": float(self._last_opening.max()) if self.count else 0.0,
            "tape_max_slip_m": float(np.linalg.norm(self._last_slip, axis=1).max()) if self.count else 0.0,
            "tape_damage_energy_proxy_j": self.damage_energy_proxy_j,
            "tape_contact_work_proxy_j": self.contact_work_proxy_j,
            "tape_reach_limited_releases": int(self.reach_limited.sum()),
            "tape_orientation_releases": int(self.orientation_released.sum()),
            "tape_rebond_count": self.rebond_count,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return JSON-compatible controller history; serialize MjData separately."""
        if not self.initialized:
            raise RuntimeError("initialize must precede state_dict")
        names = ("damage", "max_separation_ratio", "wear", "contamination", "bond_age_s",
                 "rebond_dwell_s", "separation_speed_m_s", "slip_speed_m_s", "anchors_local_m",
                 "reach_limited", "orientation_released", "_last_opening", "_last_slip", "_enabled",
                 "_last_pressure_pa", "_reference_frames", "_reference_support_rotations", "_face_signs")
        return {"version": 1, "pair_names": self.pair_names.copy(), "params": self.params.copy(),
                "arrays": {name: getattr(self, name).tolist() for name in names},
                "last_time": self._last_time, "damage_energy_proxy_j": self.damage_energy_proxy_j,
                "contact_work_proxy_j": self.contact_work_proxy_j, "rebond_count": self.rebond_count}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore controller history to a matching model and parameter set."""
        if state.get("version") != 1 or list(state.get("pair_names", [])) != self.pair_names:
            raise ValueError("Tape state belongs to a different bond model or version")
        if dict(state.get("params", {})) != self.params:
            raise ValueError("Tape state parameters do not match this controller")
        shapes = {name: (self.count,) for name in (
            "damage", "max_separation_ratio", "wear", "contamination", "bond_age_s",
            "rebond_dwell_s", "separation_speed_m_s", "slip_speed_m_s", "reach_limited",
            "orientation_released", "_last_opening", "_enabled", "_last_pressure_pa")}
        shapes.update({"anchors_local_m": (self.count, 3), "_last_slip": (self.count, 3),
                       "_reference_frames": (self.model.nflexvert, 3, 3),
                       "_reference_support_rotations": (self.count, 3, 3),
                       "_face_signs": (len(self._triangles),)})
        arrays = {}
        for name, shape in shapes.items():
            boolean = name in {"reach_limited", "orientation_released", "_enabled"}
            value = np.asarray(state["arrays"][name], dtype=bool if boolean else float)
            # JSON turns empty multidimensional arrays into []; restore known shape.
            if value.size == 0 and np.prod(shape) == 0:
                value = value.reshape(shape)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid tape state array: {name}")
            arrays[name] = value.copy()
        for name in ("damage", "wear", "contamination"):
            if np.any((arrays[name] < 0) | (arrays[name] > 1)):
                raise ValueError(f"Tape state {name} is outside [0,1]")
        scalars = {name: float(state[name]) for name in (
            "last_time", "damage_energy_proxy_j", "contact_work_proxy_j", "rebond_count")}
        if not all(np.isfinite(value) and value >= 0 for value in scalars.values()):
            raise ValueError("Tape state scalar values must be finite and nonnegative")
        for name, value in arrays.items():
            setattr(self, name, value)
        self._last_time = scalars["last_time"]
        self.damage_energy_proxy_j = scalars["damage_energy_proxy_j"]
        self.contact_work_proxy_j = scalars["contact_work_proxy_j"]
        self.rebond_count = int(scalars["rebond_count"])
        self.initialized = True
        self._write_parameters(self.damage, self._enabled)
