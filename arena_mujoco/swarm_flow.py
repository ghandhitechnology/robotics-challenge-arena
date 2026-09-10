"""Local motion demonstrations and measurements for a magnetic robot body.

These functions produce wheel/lift/magnet commands. They never alter poses or
apply forces. Magnetic connectivity must come from the physical coupler, not a
proximity graph. Local +Y is forward, matching ``swarm_robot.DESIGN``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .swarm_robot import DESIGN


@dataclass(frozen=True)
class FlowConfig:
    neighbor_radius: float = .105
    neighbor_count: int = 8
    speed: float = .040
    attraction_gain: float = .45
    alignment_gain: float = .35
    separation_gain: float = 1.3
    shape_gain: float = .20
    obstacle_margin: float = .055
    consensus_steps: int = 5
    heading_gain: float = 5.
    max_yaw_rate: float = 1.2
    steering_full_speed: float = .015
    release_heading: float = .55
    release_min_speed: float = .012
    release_shear_speed: float = .025


def compact_packing(count=40):
    """West-facing, shoulder-dock-aligned poses within the full start zone.

    Rows alternate by 28 mm along the body. Bounds for all forty module
    footprints are X=0.872..1.123 m and Y=0.758..1.0052 m.
    """
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 40:
        raise ValueError("count must be an integer from 1 to 40")
    row, column = np.divmod(np.arange(count), 4)
    return np.column_stack((.900 + .056 * column + .028 * (row % 2),
                            .770 + .0248 * row, np.full(count, .0072),
                            np.full(count, math.pi / 2)))


def _clip_length(vector, limit):
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector * np.minimum(1., limit / np.maximum(norm, 1e-12))


def _axes(yaw):
    return (np.column_stack((-np.sin(yaw), np.cos(yaw))),
            np.column_stack((np.cos(yaw), np.sin(yaw))))


def local_velocity_field(positions, velocities, yaw, waypoint, *, links=None,
                         shape_radii=(.15, .15), shape_yaw=0., obstacles=(),
                         arena_bounds=(0., 0., 1.143, 1.181), config=None):
    """Return desired world XY velocities from a shared goal and local state.

    ``obstacles`` contains axis-aligned (xmin, ymin, xmax, ymax) rectangles.
    ``shape_radii`` bounds the body around its measured centroid without
    assigning any module a slot. ``links`` is the actual NxN magnetic graph.
    The caller may batch this function over independent simulation worlds.
    """
    cfg = config or FlowConfig()
    pos, vel, yaw = np.asarray(positions, float), np.asarray(velocities, float), np.asarray(yaw, float)
    if pos.ndim != 2 or pos.shape[1] != 2 or vel.shape != pos.shape or yaw.shape != (len(pos),):
        raise ValueError("positions/velocities must be Nx2 and yaw must be N")
    if not len(pos) or not all(np.isfinite(a).all() for a in (pos, vel, yaw, waypoint)):
        raise ValueError("flow inputs must be finite and contain at least one robot")
    radii = np.asarray(shape_radii, float)
    if radii.shape != (2,) or np.any(radii <= 0):
        raise ValueError("shape_radii must contain two positive radii")
    n = len(pos)
    delta = pos[None, :, :] - pos[:, None, :]
    dist = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(dist, np.inf)
    graph = np.zeros((n, n), bool) if links is None else np.asarray(links, bool).copy()
    if graph.shape != (n, n):
        raise ValueError("links must have shape NxN")
    graph |= graph.T
    np.fill_diagonal(graph, False)
    count = min(cfg.neighbor_count, n - 1)
    if count:
        cutoff = np.partition(dist, count - 1, axis=-1)[:, count - 1]
        # Include equal-distance ties so relabeling robots cannot change motion.
        neighbors = (dist <= cutoff[:, None] + 1e-10) & (dist < cfg.neighbor_radius)
    else:
        neighbors = np.zeros((n, n), bool)
    neighbors |= graph
    denominator = np.maximum(neighbors.sum(-1, keepdims=True), 1)
    fwd, right = _axes(yaw)
    forward_gap = np.einsum('ijk,ik->ij', delta, fwd)
    right_gap = np.einsum('ijk,ik->ij', delta, right)
    # The elliptical spacing accommodates 28 x 24.8 mm staggered neighbors.
    elliptical = np.sqrt((forward_gap / .056) ** 2 + (right_gap / .028) ** 2)
    np.fill_diagonal(elliptical, np.inf)
    direction = delta / np.maximum(dist[..., None], 1e-9)
    direction = np.nan_to_num(direction)
    separation = -(direction * (.028 * np.maximum(.98 - elliptical, 0) * neighbors)[..., None]).sum(1)
    stretch = np.where(neighbors, np.maximum(dist - .058, 0), 0.)
    attraction = (direction * stretch[..., None]).sum(1) / denominator
    alignment = neighbors @ vel / denominator - vel
    alignment[neighbors.sum(-1) == 0] = 0
    center = pos.mean(0)
    goal_delta = np.asarray(waypoint, float) - center
    drift = _clip_length(.8 * goal_delta, cfg.speed)
    rotation = np.array([[math.cos(shape_yaw), -math.sin(shape_yaw)],
                         [math.sin(shape_yaw), math.cos(shape_yaw)]])
    local = (pos - center) @ rotation
    radius = np.linalg.norm(local / radii, axis=-1)
    boundary = -local * np.maximum(1. - 1. / np.maximum(radius, 1e-9), 0.)[:, None]
    shape = boundary @ rotation.T
    desired = (drift + cfg.attraction_gain * attraction + cfg.alignment_gain * alignment
               + cfg.separation_gain * separation + cfg.shape_gain * shape)
    # Steer along obstacle edges as well as away from them. The finite support
    # keeps distant objects from changing the field throughout the arena.
    margin = cfg.obstacle_margin
    for obstacle in obstacles:
        lower, upper = np.asarray(obstacle[:2], float), np.asarray(obstacle[2:], float)
        nearest_point = np.clip(pos, lower, upper)
        outward = pos - nearest_point
        gap = np.linalg.norm(outward, axis=-1)
        inside = gap < 1e-10
        if inside.any():
            face_gaps = np.column_stack((pos[:, 0] - lower[0], upper[0] - pos[:, 0],
                                         pos[:, 1] - lower[1], upper[1] - pos[:, 1]))
            normals = np.array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])
            outward[inside] = normals[np.argmin(face_gaps[inside], axis=-1)]
        normal = outward / np.maximum(np.linalg.norm(outward, axis=-1, keepdims=True), 1e-9)
        influence = np.clip((margin - gap) / margin, 0, 1)
        tangent = np.column_stack((-normal[:, 1], normal[:, 0]))
        tangent *= np.where(tangent @ goal_delta >= 0, 1., -1.)[:, None]
        inward_speed = np.minimum(np.sum(desired * normal, axis=-1), 0.)
        desired += influence[:, None] * (normal * (.025 - inward_speed)[:, None] + .018 * tangent)
    xmin, ymin, xmax, ymax = arena_bounds
    safe = .034
    desired[:, 0] += .8 * (np.maximum(xmin + safe - pos[:, 0], 0) - np.maximum(pos[:, 0] - xmax + safe, 0))
    desired[:, 1] += .8 * (np.maximum(ymin + safe - pos[:, 1], 0) - np.maximum(pos[:, 1] - ymax + safe, 0))
    desired = _clip_length(desired, cfg.speed)
    # Neighbor communication spreads obstacle and boundary steering through the
    # measured magnetic graph before wheel conversion. Five local exchanges
    # reduce differential steering that otherwise peels perimeter rows away.
    if cfg.consensus_steps and graph.any():
        weights = graph.astype(float)
        np.fill_diagonal(weights, 2.)
        weights /= weights.sum(axis=-1, keepdims=True)
        for _ in range(cfg.consensus_steps):
            desired = weights @ desired
    return desired


def velocity_actions(desired_velocity, yaw, *, links=None, lift=-1., config=None,
                     allow_reverse=True):
    """Convert world velocities into normalized left/right/lift/magnet actions.

    Positive wheel actions request forward travel; the runtime applies the
    negative native wheel-joint sign from DESIGN. Magnet +1 enables attraction,
    and -1 releases it. Lift is a normalized command in [-1, 1]. With
    ``allow_reverse``, steering chooses the closer equivalent heading and uses
    negative wheel speeds for backward travel. This prevents a local cohesion
    correction behind a docked module from requesting a half-turn.
    """
    cfg = config or FlowConfig()
    desired, yaw = np.asarray(desired_velocity, float), np.asarray(yaw, float)
    n = len(yaw)
    if desired.shape != (n, 2) or not np.isfinite(desired).all() or not np.isfinite(yaw).all():
        raise ValueError("desired_velocity must be finite Nx2 and yaw finite N")
    forward, _ = _axes(yaw)
    speed = np.linalg.norm(desired, axis=-1)
    target_yaw = np.arctan2(-desired[:, 0], desired[:, 1])
    error = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
    if allow_reverse:
        error = np.where(error > math.pi / 2, error - math.pi, error)
        error = np.where(error < -math.pi / 2, error + math.pi, error)
    error = np.where(speed > 1e-5, error, 0.)
    turn = np.clip(cfg.heading_gain * error, -cfg.max_yaw_rate, cfg.max_yaw_rate)
    # Near a shared waypoint, tiny lateral corrections have a noisy direction.
    # Scale their steering instead of requesting a full-rate turn in the pack.
    turn *= np.minimum(speed / cfg.steering_full_speed, 1.)
    along = np.sum(desired * forward, axis=-1)
    # Reduce translation during large turns so the long body can reorient.
    along *= np.maximum(np.cos(error), 0.)
    graph = np.zeros((n, n), bool) if links is None else np.asarray(links, bool).copy()
    if graph.shape != (n, n):
        raise ValueError("links must have shape NxN")
    graph |= graph.T
    np.fill_diagonal(graph, False)
    degree = graph.sum(-1)
    relative = desired[:, None, :] - desired[None, :, :]
    shear = np.max(np.linalg.norm(relative, axis=-1) * graph, axis=-1, initial=0.)
    release = ((np.abs(error) > cfg.release_heading) & (speed >= cfg.release_min_speed)
               & (degree <= 2) & (degree > 0))
    release |= (shear > cfg.release_shear_speed) & (degree > 1)
    scale = DESIGN['wheel_radius_m'] * DESIGN['max_wheel_speed_rad_s']
    differential = turn * DESIGN['wheel_track_m'] / 2
    wheels = np.column_stack((along - differential, along + differential)) / scale
    return np.column_stack((np.clip(wheels, -1, 1),
                            np.broadcast_to(np.clip(lift, -1, 1), (n,)),
                            np.where(release, -1., 1.)))


_DOCK_XY = np.array([[-.012, -.014], [-.012, .014], [.012, -.014], [.012, .014]])


def _clear_docking_poses(centers, headings, positions, yaw, ignored):
    """Conservative oriented-footprint clearance for proposed attachment poses."""
    cf, cr = _axes(headings)
    of, ort = _axes(yaw)
    # Footprint center is 0.5 mm ahead of the root reference.
    delta = (positions[None] + .0005*of[None]
             - centers[:, None] - .0005*cf[:, None])
    axes = (cr[:, None], cf[:, None], ort[None], of[None])
    separated = np.zeros((len(centers), len(positions)), dtype=bool)
    for axis in axes:
        distance = np.abs(np.sum(delta*axis, axis=-1))
        own = (.0121*np.abs(np.sum(cr[:, None]*axis, axis=-1))
               + .0276*np.abs(np.sum(cf[:, None]*axis, axis=-1)))
        other = (.0121*np.abs(np.sum(ort[None]*axis, axis=-1))
                 + .0276*np.abs(np.sum(of[None]*axis, axis=-1)))
        separated |= distance >= own+other
    separated[:, ignored] = True
    return separated.all(axis=-1)


def docking_targets(positions, yaw, links, *, obstacles=(),
                    arena_bounds=(0., 0., 1.143, 1.181), capture_distance=.0015,
                    search_distance=.24, reserved_ports=None):
    """Choose free shoulder ports for isolated modules to rejoin nearby bodies.

    Targets follow the current body geometry. No lattice slots or module IDs
    determine placement. The returned position and heading are also suitable
    policy observations. Connected modules retain their local flow command.
    """
    pos, yaw = np.asarray(positions, float), np.asarray(yaw, float)
    graph = np.asarray(links, bool)
    n = len(pos)
    if pos.shape != (n, 2) or yaw.shape != (n,) or graph.shape != (n, n):
        raise ValueError("docking needs Nx2 positions, N headings and NxN links")
    result = dict(positions=pos.copy(), yaw=yaw.copy(), active=np.zeros(n, bool),
                  neighbor=np.full(n, -1, int), port=np.full(n, -1, int),
                  own_port=np.full(n, -1, int), axis_flip=np.zeros(n, bool))
    if n < 2:
        return result
    graph = graph | graph.T
    forward, right = _axes(yaw)
    sites = (pos[:, None] + _DOCK_XY[None, :, :1]*right[:, None]
             + _DOCK_XY[None, :, 1:]*forward[:, None])
    normals = np.sign(_DOCK_XY[None, :, :1])*right[:, None]
    gap = np.linalg.norm(sites[:, :, None, None]-sites[None, None], axis=-1)
    opposition = -np.einsum('ipa,jqa->ipjq', normals, normals)
    other = ~np.eye(n, dtype=bool)[:, None, :, None]
    near = (gap <= capture_distance) & (opposition > .6) & other
    capture_possible = near.any(axis=(1, 2, 3))
    occupied = ((gap < .006) & (opposition > .5) & graph[:, None, :, None]).any(axis=(2, 3))
    labels, _ = connected_components(graph)
    counts = np.bincount(labels)
    largest = np.flatnonzero(counts == counts.max())
    # A geometric tie-break keeps the same anchors after robots are relabeled.
    center = pos.mean(0)
    anchor = min(largest, key=lambda label: (
        float(np.linalg.norm(pos[labels == label].mean(0)-center)),
        *pos[labels == label].mean(0).tolist()))
    anchors = np.flatnonzero(labels == anchor)
    isolated = np.flatnonzero((labels != anchor) & (counts[labels] == 1) & ~capture_possible)
    if not len(isolated):
        return result
    # Give the nearest arrivals first choice of an unoccupied physical port.
    closest = np.linalg.norm(pos[isolated, None]-pos[None, anchors], axis=-1).min(axis=-1)
    order = np.lexsort((pos[isolated, 1], pos[isolated, 0], closest))
    reserved = occupied.copy()
    if reserved_ports is not None:
        reserved |= np.asarray(reserved_ports, bool)
    virtual = pos.copy()
    virtual_yaw = yaw.copy()
    xmin, ymin, xmax, ymax = arena_bounds
    for robot in isolated[order]:
        distance = np.linalg.norm(pos[anchors]-pos[robot], axis=-1)
        local_anchors = anchors[np.argsort(distance)[:8]]
        local_anchors = local_anchors[np.linalg.norm(pos[local_anchors]-pos[robot], axis=-1) < search_distance]
        owners, ports = np.nonzero(~reserved[local_anchors])
        if not len(owners):
            continue
        owners = local_anchors[owners]
        # Two own shoulder positions can mate with each free neighbor port.
        owners, ports = np.repeat(owners, 2), np.repeat(ports, 2)
        target_yaw = yaw[owners].copy()
        difference = np.arctan2(np.sin(target_yaw-yaw[robot]), np.cos(target_yaw-yaw[robot]))
        target_yaw += np.where(np.abs(difference) > math.pi/2, math.pi, 0.)
        tf, tr = _axes(target_yaw)
        normal = normals[owners, ports]
        own_sign = -np.sign(np.sum(normal*tr, axis=-1))
        own_y = np.tile([-.014, .014], len(owners)//2)
        targets = sites[owners, ports] + .0008*normal - .012*own_sign[:, None]*tr - own_y[:, None]*tf
        valid = _clear_docking_poses(targets, target_yaw, virtual, virtual_yaw, robot)
        half_xy = .012*np.abs(tr)+.0275*np.abs(tf)
        center_xy = targets+.0005*tf
        valid &= np.all(center_xy-half_xy >= [xmin+.001, ymin+.001], axis=-1)
        valid &= np.all(center_xy+half_xy <= [xmax-.001, ymax-.001], axis=-1)
        # Only approach an exposed port from its exterior half-space.
        valid &= np.sum((pos[robot]-targets)*normal, axis=-1) >= -.012
        for obstacle in obstacles:
            lo, hi = np.asarray(obstacle[:2]), np.asarray(obstacle[2:])
            intersects = np.all(center_xy+half_xy > lo, axis=-1) & np.all(center_xy-half_xy < hi, axis=-1)
            valid &= ~intersects
        difference = np.arctan2(np.sin(target_yaw-yaw[robot]), np.cos(target_yaw-yaw[robot]))
        approach = targets-pos[robot]
        # A nearly sideways final approach cannot be driven by these wheels.
        # Prefer a diagonal shoulder match that leaves room to roll alongside.
        approach_angle = np.arctan2(np.abs(np.sum(approach*tr, axis=-1)),
                                    np.abs(np.sum(approach*tf, axis=-1)))
        cost = (np.linalg.norm(approach, axis=-1)+.015*np.abs(difference)
                + .030*approach_angle)
        cost = np.where(valid, cost, np.inf)
        best = int(np.argmin(cost))
        if not np.isfinite(cost[best]):
            continue
        result['positions'][robot] = targets[best]
        result['yaw'][robot] = target_yaw[best]
        result['active'][robot] = True
        result['neighbor'][robot] = owners[best]
        result['port'][robot] = ports[best]
        result['own_port'][robot] = int(2*(own_sign[best] > 0)+(own_y[best] > 0))
        result['axis_flip'][robot] = abs(target_yaw[best]-yaw[owners[best]]) > math.pi/2
        reserved[owners[best], ports[best]] = True
        virtual[robot], virtual_yaw[robot] = targets[best], target_yaw[best]
    return result


class DockingState:
    """Latched physical ports with an explicit once-per-step state update.

    ``update`` is the only mutating method. Observation and action construction
    may call ``guidance`` repeatedly without advancing an approach. Module IDs
    keep plans attached to the same robot when the active subset changes.
    """

    def __init__(self, count):
        self.count = int(count)
        self.reset()

    def reset(self):
        self.neighbor = np.full(self.count, -1, int)
        self.port = np.full(self.count, -1, int)
        self.own_port = np.full(self.count, -1, int)
        self.axis_flip = np.zeros(self.count, bool)
        self.stage = np.full(self.count, -1, int)
        self.offset = np.zeros(self.count)
        self.waiting = np.zeros(self.count, bool)
        self.release_steps = np.zeros(self.count, int)
        self.release_direction = np.ones(self.count)

    def _ids(self, n, module_ids):
        ids = np.arange(n) if module_ids is None else np.asarray(module_ids, int)
        if ids.shape != (n,) or len(np.unique(ids)) != n or np.any((ids < 0) | (ids >= self.count)):
            raise ValueError("module_ids must be unique valid global robot IDs")
        return ids

    def guidance(self, positions, yaw, *, module_ids=None):
        """Return current pose directives without changing docking state."""
        pos, yaw = np.asarray(positions, float), np.asarray(yaw, float)
        ids = self._ids(len(pos), module_ids)
        local = {int(robot): i for i, robot in enumerate(ids)}
        result = dict(positions=pos.copy(), yaw=yaw.copy(), active=self.waiting[ids].copy(),
                      stage=np.where(self.waiting[ids], -2, -1), final_positions=pos.copy(),
                      neighbor=np.full(len(pos), -1, int), port=self.port[ids].copy())
        releasing = self.release_steps[ids] > 0
        forward, _ = _axes(yaw)
        result['positions'][releasing] += (.025*self.release_direction[ids[releasing], None]
                                           *forward[releasing])
        result['active'][releasing], result['stage'][releasing] = True, -3
        result['release'] = releasing.copy()
        for i, robot in enumerate(ids):
            destination = local.get(int(self.neighbor[robot]))
            if self.stage[robot] < 0 or destination is None:
                continue
            target_yaw = yaw[destination]+math.pi*self.axis_flip[robot]
            forward, right = _axes(np.array([target_yaw]))
            af, ar = _axes(yaw[destination:destination+1])
            dock = _DOCK_XY[self.port[robot]]
            own = _DOCK_XY[self.own_port[robot]]
            normal = np.sign(dock[0])*ar[0]
            final = (pos[destination]+dock[0]*ar[0]+dock[1]*af[0]+.0008*normal
                     -own[0]*right[0]-own[1]*forward[0])
            stage = self.stage[robot]
            target = final+self.offset[robot]*forward[0] if stage == 0 else final
            if stage == 1:
                target = pos[i]
            result['positions'][i], result['yaw'][i] = target, target_yaw
            result['active'][i], result['stage'][i] = True, stage
            result['final_positions'][i], result['neighbor'][i] = final, destination
        return result

    def update(self, positions, yaw, links, *, module_ids=None, obstacles=(),
               arena_bounds=(0., 0., 1.143, 1.181)):
        """Advance once after physics, measured links, and membership updates."""
        pos, yaw, graph = np.asarray(positions, float), np.asarray(yaw, float), np.asarray(links, bool)
        ids = self._ids(len(pos), module_ids)
        if pos.shape != (len(ids), 2) or yaw.shape != (len(ids),) or graph.shape != (len(ids), len(ids)):
            raise ValueError("docking needs Nx2 positions, N headings and NxN links")
        graph = graph | graph.T
        local = {int(robot): i for i, robot in enumerate(ids)}
        present = np.zeros(self.count, bool)
        present[ids] = True
        self.stage[~present] = -1
        self.release_steps[~present] = 0
        self.release_steps = np.maximum(self.release_steps-1, 0)
        self.waiting[:] = False
        labels, _ = connected_components(graph)
        counts = np.bincount(labels)
        largest = int(np.argmax(counts)) if len(counts) else -1
        detached_group = (labels != largest) & (counts[labels] > 1)
        forward, right = _axes(yaw)
        for label in np.unique(labels[detached_group]):
            group = labels == label
            delta = pos[group]-pos[group].mean(axis=0)
            along = np.sum(delta*forward[group], axis=-1)
            side = np.sum(delta*right[group], axis=-1)
            direction = np.where(np.abs(along) > .001, along, side)
            self.release_direction[ids[group]] = np.where(direction >= 0., 1., -1.)
            self.release_steps[ids[group]] = 70
        for i, robot in enumerate(ids):
            destination = local.get(int(self.neighbor[robot]))
            if (graph[i].any() or destination is None or labels[destination] != largest
                    or self.release_steps[robot] > 0):
                self.stage[robot] = -1
                self.neighbor[robot] = -1
        guidance = self.guidance(pos, yaw, module_ids=ids)
        for i, robot in enumerate(ids):
            stage = self.stage[robot]
            if stage == 0 and np.linalg.norm(guidance['positions'][i]-pos[i]) < .003:
                self.stage[robot] = 1
            elif stage == 1:
                error = guidance['yaw'][i]-yaw[i]
                if abs(math.atan2(math.sin(error), math.cos(error))) < .035:
                    self.stage[robot] = 2
        reserved = np.zeros((len(ids), 4), bool)
        normals = []
        for robot in ids[self.stage[ids] >= 0]:
            destination = local[int(self.neighbor[robot])]
            reserved[destination, self.port[robot]] = True
            _, right = _axes(yaw[destination:destination+1])
            normals.append(np.sign(_DOCK_XY[self.port[robot], 0])*right[0])
        choices = docking_targets(pos, yaw, graph, obstacles=obstacles,
                                  arena_bounds=arena_bounds, reserved_ports=reserved)
        labels, _ = connected_components(graph)
        for i in np.flatnonzero(choices['active']):
            robot = ids[i]
            if self.stage[robot] >= 0 or self.release_steps[robot] > 0:
                continue
            destination = choices['neighbor'][i]
            _, ar = _axes(yaw[destination:destination+1])
            normal = np.sign(_DOCK_XY[choices['port'][i], 0])*ar[0]
            # At most one arrival per exterior side keeps the rolling lanes clear.
            if len(normals) >= 2 or any(normal@other > -.5 for other in normals):
                self.waiting[robot] = True
                continue
            target, target_yaw = choices['positions'][i], choices['yaw'][i]
            forward, right = _axes(np.array([target_yaw]))
            component = pos[labels == labels[destination]]
            extent = (component-target)@forward[0]
            offsets = np.array([extent.min()-.070, extent.max()+.070])
            staging = target+offsets[:, None]*forward[0]
            headings = np.full(2, target_yaw)
            valid = _clear_docking_poses(staging, headings, pos, yaw, i)
            half = .012*np.abs(right[0])+.028*np.abs(forward[0])
            xmin, ymin, xmax, ymax = arena_bounds
            valid &= np.all(staging-half >= [xmin+.002, ymin+.002], axis=-1)
            valid &= np.all(staging+half <= [xmax-.002, ymax-.002], axis=-1)
            for obstacle in obstacles:
                valid &= ~(np.all(staging+half > obstacle[:2], axis=-1)
                           & np.all(staging-half < obstacle[2:], axis=-1))
            cost = np.where(valid, np.linalg.norm(staging-pos[i], axis=-1), np.inf)
            if not np.isfinite(cost).any():
                continue
            self.neighbor[robot], self.port[robot] = ids[destination], choices['port'][i]
            self.own_port[robot], self.axis_flip[robot] = choices['own_port'][i], choices['axis_flip'][i]
            self.offset[robot] = offsets[int(np.argmin(cost))]
            self.stage[robot] = 0
            error = math.atan2(math.sin(target_yaw-yaw[i]), math.cos(target_yaw-yaw[i]))
            delta = target-pos[i]
            if abs(delta@right[0]) < .0015 and abs(error) < .1 and np.linalg.norm(delta) < .06:
                self.stage[robot] = 2
            normals.append(normal)


def _staged_docking_actions(positions, yaw, guidance):
    """Approach, align in open space, then roll parallel to a shoulder port."""
    actions = _docking_actions(positions, yaw, guidance['positions'], guidance['yaw'])
    stage = guidance['stage']
    actions[stage == -3, 3] = -1.
    target_yaw = guidance['yaw']
    heading = np.arctan2(np.sin(target_yaw-yaw), np.cos(target_yaw-yaw))
    forward, right = _axes(target_yaw)
    delta = guidance['positions']-positions
    along, lateral = np.sum(delta*forward, axis=-1), np.sum(delta*right, axis=-1)
    speed = np.clip(along, -.012, .012)
    turn = np.clip(6.*heading-100.*lateral*np.sign(speed), -1., 1.)
    align = stage == 1
    speed = np.where(align, 0., speed)
    turn = np.where(align, np.clip(6.*heading, -2.5, 2.5), turn)
    stop = (stage == -2) | ((stage == 2) & (np.linalg.norm(delta, axis=-1) < .001))
    speed, turn = np.where(stop, 0., speed), np.where(stop, 0., turn)
    mask = (stage == 1) | (stage == 2) | (stage == -2)
    scale = DESIGN['wheel_radius_m']*DESIGN['max_wheel_speed_rad_s']
    differential = turn*DESIGN['wheel_track_m']/2
    actions[mask, :2] = np.clip(np.column_stack((speed-differential, speed+differential))[mask]/scale, -1., 1.)
    return actions


def _docking_actions(positions, yaw, targets, target_yaw, speed_limit=.030):
    """Wheel-only bidirectional pose servo for the last approach to a dock."""
    delta = targets-positions
    rho = np.linalg.norm(delta, axis=-1)
    bearing = np.arctan2(-delta[:, 0], delta[:, 1])
    alpha = np.arctan2(np.sin(bearing-yaw), np.cos(bearing-yaw))
    reverse = np.abs(alpha) > math.pi/2
    virtual_yaw = yaw+reverse*math.pi
    alpha = np.arctan2(np.sin(bearing-virtual_yaw), np.cos(bearing-virtual_yaw))
    beta = np.arctan2(np.sin(target_yaw+reverse*math.pi-bearing),
                      np.cos(target_yaw+reverse*math.pi-bearing))
    speed = np.minimum(.8*rho, speed_limit)*np.maximum(np.cos(alpha), 0.)
    speed *= np.where(reverse, -1., 1.)
    # A weak final-heading term lets the short-range magnetic torque finish
    # alignment instead of wedging the long chassis against its neighbor.
    turn = np.clip(4.*alpha-.2*beta, -2.5, 2.5)
    heading = np.arctan2(np.sin(target_yaw-yaw), np.cos(target_yaw-yaw))
    near = rho < .0015
    speed = np.where(near, 0., speed)
    turn = np.where(near, np.clip(6.*heading, -2.5, 2.5), turn)
    differential = turn*DESIGN['wheel_track_m']/2
    scale = DESIGN['wheel_radius_m']*DESIGN['max_wheel_speed_rad_s']
    wheels = np.column_stack((speed-differential, speed+differential))/scale
    return np.column_stack((np.clip(wheels, -1., 1.), np.full(len(yaw), -1.),
                            np.ones(len(yaw))))


def flow_actions(positions, velocities, yaw, waypoint, **kwargs):
    """Return actions and world velocities, optionally with observed dock poses.

    ``docking=True`` adds local free-port attachment for isolated modules.
    ``return_guidance=True`` adds the docking target dictionary as a third
    return value so callers can expose those same poses to a learned policy.
    """
    lift = kwargs.pop('lift', -1.)
    allow_reverse = kwargs.pop('allow_reverse', True)
    docking = kwargs.pop('docking', False)
    docking_state = kwargs.pop('docking_state', None)
    module_ids = kwargs.pop('module_ids', None)
    return_guidance = kwargs.pop('return_guidance', False)
    desired = local_velocity_field(positions, velocities, yaw, waypoint, **kwargs)
    actions = velocity_actions(desired, yaw, links=kwargs.get('links'), lift=lift,
                               config=kwargs.get('config'), allow_reverse=allow_reverse)
    pos, heading = np.asarray(positions, float), np.asarray(yaw, float)
    guidance = dict(positions=pos.copy(), yaw=heading.copy(), active=np.zeros(len(pos), bool))
    if docking and kwargs.get('links') is not None:
        guidance = (docking_state.guidance(pos, heading, module_ids=module_ids)
                    if docking_state is not None else docking_targets(pos, heading, kwargs['links'],
                                   obstacles=kwargs.get('obstacles', ()),
                                   arena_bounds=kwargs.get('arena_bounds', (0., 0., 1.143, 1.181))))
        if docking_state is not None and len(kwargs.get('obstacles', ())):
            # Only approach routing bends around payloads. Alignment and final
            # rolling retain the physical port axis. Expose the same deflected
            # pose to the policy that the wheel controller receives.
            for robot in np.flatnonzero(guidance['active'] & (guidance['stage'] == 0)):
                navigation = local_velocity_field(
                    pos[robot:robot+1], np.zeros((1, 2)), heading[robot:robot+1],
                    guidance['positions'][robot], obstacles=kwargs['obstacles'],
                    arena_bounds=kwargs.get('arena_bounds', (0., 0., 1.143, 1.181)),
                    config=FlowConfig(speed=.030, obstacle_margin=.020))[0]
                guidance['positions'][robot] = pos[robot]+navigation/.8
                if np.linalg.norm(navigation) > 1e-5:
                    angle = math.atan2(-navigation[0], navigation[1])
                    difference = math.atan2(math.sin(angle-heading[robot]), math.cos(angle-heading[robot]))
                    guidance['yaw'][robot] = angle+(math.pi if abs(difference) > math.pi/2 else 0.)
        mask = guidance['active']
        if mask.any():
            if docking_state is None:
                actions[mask] = _docking_actions(pos[mask], heading[mask], guidance['positions'][mask], guidance['yaw'][mask])
            else:
                actions[mask] = _staged_docking_actions(pos, heading, guidance)[mask]
            desired[mask] = _clip_length(guidance['positions'][mask]-pos[mask], .030)
    return (actions, desired, guidance) if return_guidance else (actions, desired)


def connected_components(links):
    """Return component labels and descending sizes for an undirected graph."""
    graph = np.asarray(links, bool)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("links must be square")
    graph = graph | graph.T
    labels = np.full(len(graph), -1, dtype=int)
    sizes = []
    for root in range(len(graph)):
        if labels[root] >= 0:
            continue
        stack = [root]
        labels[root] = len(sizes)
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in np.flatnonzero(graph[node]):
                if labels[neighbor] < 0:
                    labels[neighbor] = labels[root]
                    stack.append(int(neighbor))
        sizes.append(size)
    return labels, sorted(sizes, reverse=True)


def body_telemetry(positions, velocities, links, *, link_forces=None,
                   initial_positions=None, previous_links=None, force_threshold=.002):
    """Measure shape, motion and actual force-bearing magnetic connectivity.

    Pass NxN link force magnitudes in newtons to distinguish enabled docking
    from load-bearing connections. Without forces, only the supplied physical
    link graph is measured; this function never infers links from distances.
    """
    pos, vel = np.asarray(positions, float), np.asarray(velocities, float)
    n = len(pos)
    graph = np.asarray(links, bool).copy()
    if not n or pos.shape != (n, 2) or vel.shape != pos.shape or graph.shape != (n, n):
        raise ValueError("telemetry needs nonempty Nx2 positions/velocities and NxN links")
    graph |= graph.T
    np.fill_diagonal(graph, False)
    labels, sizes = connected_components(graph)
    center = pos.mean(0)
    eigenvalues = np.maximum(np.linalg.eigvalsh((pos - center).T @ (pos - center) / n), 0.)
    speed = np.linalg.norm(vel, axis=-1)
    result = dict(component_count=len(sizes), component_sizes=sizes,
                  largest_component_fraction=sizes[0] / n,
                  isolated_count=int(np.sum(graph.sum(-1) == 0)),
                  edge_count=int(np.triu(graph, 1).sum()), centroid_xy=center.tolist(),
                  span_xy=np.ptp(pos, axis=0).tolist(),
                  radius_of_gyration=float(np.sqrt(eigenvalues.sum())),
                  shape_aspect_ratio=float(np.sqrt((eigenvalues[1] + 1e-12) / (eigenvalues[0] + 1e-12))),
                  moving_fraction=float(np.mean(speed > .003)),
                  mean_speed=float(speed.mean()), centroid_speed=float(np.linalg.norm(vel.mean(0))))
    if initial_positions is not None:
        displacement = np.linalg.norm(pos - np.asarray(initial_positions, float), axis=-1)
        result.update(minimum_displacement=float(displacement.min()),
                      displaced_fraction=float(np.mean(displacement > .10)))
    if previous_links is not None:
        previous = np.asarray(previous_links, bool)
        previous = previous | previous.T
        result.update(formed_edges=int(np.triu(graph & ~previous, 1).sum()),
                      released_edges=int(np.triu(~graph & previous, 1).sum()))
    if link_forces is not None:
        forces = np.asarray(link_forces, float)
        if forces.shape != (n, n) or not np.isfinite(forces).all() or np.any(forces < 0):
            raise ValueError("link_forces must be finite nonnegative NxN magnitudes")
        forces = np.maximum(forces, forces.T)
        bearing = graph & (forces >= force_threshold)
        _, bearing_sizes = connected_components(bearing)
        result.update(force_bearing_edge_count=int(np.triu(bearing, 1).sum()),
                      force_bearing_largest_fraction=bearing_sizes[0] / n,
                      peak_link_force=float(forces.max()),
                      summed_link_force=float(np.triu(forces * graph, 1).sum()))
    return result
