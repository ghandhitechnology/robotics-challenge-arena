"""Finite-range electropermanent docking forces for native MuJoCo modules.

The four shoulder ports are a proposed hardware addition. Their 0.15 N peak
force and 3 mm face-gap reach are design assumptions, not prototype measurements.
Gilpin et al., Robot Pebbles (ICRA 2010), demonstrated 12 mm EPM modules with
2.06--3.18 N normal holding force and 4.31 mm two-sided pull-in:
https://cba.mit.edu/docs/papers/10.05.knaian.ICRA.pdf

Each occupied port has one partner. Forces derive from a compact potential,
U = -depth * (1 - r**2 / R**2)**3 * smoothstep(opposition), inside R.
Here r joins recessed magnetic centers, and opposition is minus the outward
normal dot product. Both positional and angular gradients are applied. Contact
colliders provide compression support and friction. Separation needs no special
break impulse: attraction has a finite maximum and vanishes outside the well.
A disabled module requests a coordinated release at both ends of each link.
This models the EPM release handshake, not one powered magnet against bare iron.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from xml.etree.ElementTree import SubElement

import mujoco
import numpy as np


@dataclass(frozen=True)
class MagneticParameters:
    peak_force_n: float = .15
    capture_gap_m: float = .003
    center_inset_m: float = .0007
    contact_gap_m: float = .0006
    opposition_cosine: float = .5

    def __post_init__(self):
        vals = (self.peak_force_n, self.capture_gap_m, self.center_inset_m,
                self.contact_gap_m, self.opposition_cosine)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("Magnetic parameters must be finite")
        if min(vals[:3]) <= 0 or not 0 <= self.contact_gap_m < self.capture_gap_m:
            raise ValueError("Magnetic force, reach and inset must be positive")
        if not -1 < self.opposition_cosine < 1:
            raise ValueError("Opposition cosine must be between -1 and 1")

    @property
    def radius_m(self):
        return self.capture_gap_m + 2 * self.center_inset_m

    @property
    def depth_j(self):
        # Maximum of 6*s*(1-s*s)**2 is 96/(25*sqrt(5)).
        return self.peak_force_n * self.radius_m * 25 * math.sqrt(5) / 96


PARAMETERS = MagneticParameters()
DOCK_POSITIONS = ((-.012, -.014, .004), (-.012, .014, .004),
                  (.012, -.014, .004), (.012, .014, .004))


def add_magnetic_docks(root, specs):
    """Add four colliding shoulder housings and outward +Z sites per module.

    Mutates the scene and each robot spec. The housing mass is included in the
    module's existing 40 g design budget; detailed internal packaging is pending.
    """
    for spec in specs:
        body = root.find(f".//body[@name='{spec['name']}']")
        if body is None:
            raise ValueError(f"Module body missing: {spec['name']}")
        if spec.get("magnetic_sites"):
            raise ValueError(f"Magnetic docks already present: {spec['name']}")
        names = []
        for port, (x, y, z) in enumerate(DOCK_POSITIONS):
            side = -1 if x < 0 else 1
            name = f"{spec['name']}_magnet_{port}"
            housing = name + "_housing"
            SubElement(body, "geom", name=housing, type="box",
                       pos=f"{side*.0095} {y} {z}", size=".0025 .0015 .0015",
                       mass="0", contype="4", conaffinity="7", condim="3",
                       friction=".4 .00001 .000001", solref=".004 1",
                       solimp=".9 .99 .0001", priority="2", rgba=".22 .31 .32 1")
            SubElement(body, "site", name=name, type="box", size=".0014 .0014 .0001",
                       pos=f"{x} {y} {z}", quat=f".707106781187 0 {side*.707106781187} 0",
                       rgba=".23 .65 .65 1")
            names.append(name)
            spec.setdefault("colliders", []).append({"name": housing,
                "body": spec["name"], "material": "magnet_housing",
                "grip_pad": False, "robot_part": True})
        spec["magnetic_sites"] = names
        spec["magnetic_docking"] = {
            **asdict(PARAMETERS), "ports": len(names), "releasable": True,
            "positions_body_m": [list(p) for p in DOCK_POSITIONS],
            "max_partners_per_port": 1, "switching": "coordinated EPM handshake",
            "assumption": "Unbuilt shoulder EPMs included in 40 g total mass budget",
        }
    return specs


class MagneticCoupling:
    """One coupling instance per MjData/world, shared immutable model allowed.

    Call mj_step1(model, data), apply(data, enabled), mj_step2(model, data) each
    native physics substep. The scene must use Euler/implicit integration, not
    RK4. apply overwrites xfrc_applied ONLY on bodies that own docking sites;
    callers must add any other module-body loads after apply. All payload and
    other body entries are untouched. No qpos, qvel, equality or actuator edits.

    enabled accepts one bool per robot or a robot-by-port boolean array.
    graph() and contact_graph() report ports within contact_gap_m of alignment.
    This is a close-dock measurement, not a MuJoCo collision-contact assertion.
    interaction_graph() includes every port pair receiving magnetic attraction;
    load_bearing_graph() further requires a stated minimum net pair force.
    pair_force_vectors_n[i,j] is the world force on robot i from robot j;
    pair_force_magnitudes_n is its norm, summing port vectors before taking the
    norm. These arrays report the most recent apply(), with units of newtons.
    Call reset() alongside mj_resetData.
    """
    def __init__(self, model, specs, parameters=PARAMETERS):
        self.model, self.parameters = model, parameters
        self.robot_bodies = np.array([model.body(s["name"]).id for s in specs], dtype=int)
        if np.any(self.robot_bodies == 0) or len(np.unique(self.robot_bodies)) != len(specs):
            raise ValueError("Module roots must be distinct non-world bodies")
        roots = set(self.robot_bodies)
        for body in self.robot_bodies:
            ancestor = int(model.body_parentid[body])
            while ancestor:
                if ancestor in roots:
                    raise ValueError("Module root subtrees must not overlap")
                ancestor = int(model.body_parentid[ancestor])
        self.site_ids = np.array([[model.site(n).id for n in s["magnetic_sites"]]
                                  for s in specs], dtype=int)
        if self.site_ids.ndim != 2 or not self.site_ids.size:
            raise ValueError("Each robot must have the same nonzero number of docks")
        self.num_robots, self.ports = self.site_ids.shape
        self.owners = np.repeat(np.arange(self.num_robots), self.ports)
        self.flat_sites = self.site_ids.ravel()
        self.body_ids = np.asarray(model.site_bodyid[self.flat_sites], dtype=int)
        for body, owner in zip(self.body_ids, self.owners):
            ancestor = int(body)
            while ancestor and ancestor != self.robot_bodies[owner]:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor != self.robot_bodies[owner]:
                raise ValueError("Dock sites must belong to their module root or its descendants")
        self.force_bodies = np.unique(self.body_ids)
        self._articulated_ports = np.any(self.body_ids != self.robot_bodies[self.owners])
        self._pair_a, self._pair_b = np.triu_indices(len(self.flat_sites), k=1)
        keep = self.owners[self._pair_a] != self.owners[self._pair_b]
        self._pair_a, self._pair_b = self._pair_a[keep], self._pair_b[keep]
        self._body_a, self._body_b = np.triu_indices(self.num_robots, k=1)
        body_pair_id = np.zeros((self.num_robots, self.num_robots), dtype=int)
        body_pair_id[self._body_a, self._body_b] = np.arange(len(self._body_a))
        self._port_body_pair = body_pair_id[self.owners[self._pair_a], self.owners[self._pair_b]]
        # Root-mounted ports have a constant reach under arbitrary rotation.
        # Descendant ports use their current world transforms each substep,
        # so hinges and sliders cannot move a port outside the broadphase.
        port_reach = np.linalg.norm(model.site_pos[self.site_ids], axis=-1) + parameters.center_inset_m
        self._body_reach = port_reach.max(axis=1)
        self._body_pair_range2 = (self._body_reach[self._body_a] +
                                 self._body_reach[self._body_b] + parameters.radius_m + 1e-12)**2
        self.reset()

    def _candidate_ports(self, data):
        centers = data.xpos[self.robot_bodies]
        if self._articulated_ports:
            reach = (np.linalg.norm(data.site_xpos[self.site_ids] - centers[:, None], axis=-1)
                     + self.parameters.center_inset_m).max(axis=1)
            ranges2 = (reach[self._body_a] + reach[self._body_b]
                       + self.parameters.radius_m + 1e-12)**2
        else:
            ranges2 = self._body_pair_range2
        delta = centers[self._body_b] - centers[self._body_a]
        nearby = np.einsum("ij,ij->i", delta, delta) <= ranges2
        # Keep the original lexicographic port order, including equal-distance
        # tie breaks used when unoccupied ports acquire a partner.
        keep = nearby[self._port_body_pair]
        return self._pair_a[keep], self._pair_b[keep]

    def reset(self):
        self.interacting_pairs = set()
        self.links = []
        self.last_potential_j = 0.
        self.last_peak_force_n = 0.
        self.pair_force_vectors_n = np.zeros((self.num_robots, self.num_robots, 3))
        self.pair_force_magnitudes_n = np.zeros((self.num_robots, self.num_robots))

    def graph(self):
        adjacency = np.zeros((self.num_robots, self.num_robots), dtype=bool)
        for a, b in self.links:
            i, j = self.owners[a], self.owners[b]
            adjacency[i, j] = adjacency[j, i] = True
        return adjacency

    def contact_graph(self):
        """Close docking faces, within contact_gap_m, preserving graph() semantics."""
        return self.graph()

    def interaction_graph(self):
        """Physical port pairs inside the enabled, aligned magnetic potential."""
        adjacency = np.zeros((self.num_robots, self.num_robots), dtype=bool)
        for a, b in self.interacting_pairs:
            i, j = self.owners[a], self.owners[b]
            adjacency[i, j] = adjacency[j, i] = True
        return adjacency

    def load_bearing_graph(self, force_threshold_n=.002):
        """Interactions transmitting at least force_threshold_n net force per pair."""
        if not math.isfinite(force_threshold_n) or force_threshold_n < 0:
            raise ValueError("Force threshold must be finite and nonnegative")
        return self.interaction_graph() & (self.pair_force_magnitudes_n >= force_threshold_n)

    def apply(self, data, enabled):
        enabled = np.asarray(enabled)
        if enabled.shape == (self.num_robots,):
            enabled = np.repeat(enabled[:, None], self.ports, axis=1)
        if enabled.shape != self.site_ids.shape or not np.all(np.isfinite(enabled)):
            raise ValueError("enabled must be a finite robot or robot-by-port array")
        enabled = enabled.astype(bool).ravel()
        data.xfrc_applied[self.force_bodies] = 0.
        p = self.parameters
        normals = data.site_xmat[self.flat_sites].reshape(-1, 3, 3)[:, :, 2]
        face_positions = data.site_xpos[self.flat_sites]
        positions = face_positions - p.center_inset_m * normals
        a, b = self._candidate_ports(data)
        delta = positions[b] - positions[a]
        r2 = np.einsum("ij,ij->i", delta, delta)
        opposition = -np.einsum("ij,ij->i", normals[a], normals[b])
        eligible = ((r2 < p.radius_m**2) & (opposition > p.opposition_cosine)
                    & enabled[a] & enabled[b])
        candidates = np.flatnonzero(eligible)
        pairs = {(int(a[k]), int(b[k])): int(k) for k in candidates}
        # Preserve partners while they remain in the well. New partners fill
        # free physical ports in distance order. Switching a free port on
        # introduces negative field potential and never a velocity impulse.
        selected, occupied = [], set()
        for pair in sorted(self.interacting_pairs):
            if pair in pairs:
                selected.append(pairs[pair])
                occupied.update(pair)
        for k in candidates[np.argsort(r2[candidates], kind="stable")]:
            if int(a[k]) not in occupied and int(b[k]) not in occupied:
                selected.append(int(k))
                occupied.update((int(a[k]), int(b[k])))
        self.interacting_pairs = {(int(a[k]), int(b[k])) for k in selected}
        self.links = []
        self.last_potential_j = 0.
        self.last_peak_force_n = 0.
        self.pair_force_vectors_n.fill(0.)
        self.pair_force_magnitudes_n.fill(0.)
        if not selected:
            return self.graph()
        k = np.asarray(selected)
        ia, ib = a[k], b[k]
        radial = 1 - r2[k] / p.radius_m**2
        s = np.clip((opposition[k] - p.opposition_cosine) / (1-p.opposition_cosine), 0, 1)
        angular = s*s*(3-2*s)
        angular_derivative = 6*s*(1-s)/(1-p.opposition_cosine)
        force = (6*p.depth_j / p.radius_m**2 * radial**2 * angular)[:, None] * delta[k]
        ra, rb = self.owners[ia], self.owners[ib]
        np.add.at(self.pair_force_vectors_n, (ra, rb), force)
        np.add.at(self.pair_force_vectors_n, (rb, ra), -force)
        self.pair_force_magnitudes_n[:] = np.linalg.norm(self.pair_force_vectors_n, axis=-1)
        angular_torque = (-p.depth_j * radial**3 * angular_derivative)[:, None] * np.cross(normals[ia], normals[ib])
        # The angular gradient is an internal couple. Both positional forces
        # act at their recessed magnetic centers, measured from body COM.
        ba, bb = self.body_ids[ia], self.body_ids[ib]
        torque_a = np.cross(positions[ia] - data.xipos[ba], force) + angular_torque
        torque_b = np.cross(positions[ib] - data.xipos[bb], -force) - angular_torque
        np.add.at(data.xfrc_applied[:, :3], ba, force)
        np.add.at(data.xfrc_applied[:, :3], bb, -force)
        np.add.at(data.xfrc_applied[:, 3:], ba, torque_a)
        np.add.at(data.xfrc_applied[:, 3:], bb, torque_b)
        self.last_potential_j = float(np.sum(-p.depth_j * radial**3 * angular))
        self.last_peak_force_n = float(np.max(np.linalg.norm(force, axis=1)))
        face_distance = np.linalg.norm(face_positions[ib] - face_positions[ia], axis=1)
        self.links = [(int(i), int(j)) for i, j, dist in zip(ia, ib, face_distance)
                      if dist <= p.contact_gap_m]
        return self.graph()
