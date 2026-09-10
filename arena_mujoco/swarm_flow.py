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
    heading_gain: float = 5.
    max_yaw_rate: float = 1.2
    release_heading: float = .55
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
    margin = .055
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
    return _clip_length(desired, cfg.speed)


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
    release = ((np.abs(error) > cfg.release_heading) & (degree <= 2) & (degree > 0))
    release |= (shear > cfg.release_shear_speed) & (degree > 1)
    scale = DESIGN['wheel_radius_m'] * DESIGN['max_wheel_speed_rad_s']
    differential = turn * DESIGN['wheel_track_m'] / 2
    wheels = np.column_stack((along - differential, along + differential)) / scale
    return np.column_stack((np.clip(wheels, -1, 1),
                            np.broadcast_to(np.clip(lift, -1, 1), (n,)),
                            np.where(release, -1., 1.)))


def flow_actions(positions, velocities, yaw, waypoint, **kwargs):
    """Convenience wrapper returning actions and the teacher's world velocities."""
    lift = kwargs.pop('lift', -1.)
    allow_reverse = kwargs.pop('allow_reverse', True)
    desired = local_velocity_field(positions, velocities, yaw, waypoint, **kwargs)
    actions = velocity_actions(desired, yaw, links=kwargs.get('links'), lift=lift,
                               config=kwargs.get('config'), allow_reverse=allow_reverse)
    return actions, desired


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
