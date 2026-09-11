"""Departure candidates and stateful start sequencing for the five-robot fleet.

These paths avoid parking on the shared left corridor and keep the original
start cell order. Native trajectory checks must establish their collision
clearance; this module does not override contacts or teleport a robot.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


NORTHBOUND_LEFT_X = .225
SOUTHBOUND_LEFT_X = .370
RIGHT_X = .830
CROSSING_Y = .5905
DEPARTURE_ORDER = ("lab", "red", "yellow", "kit", "green")


@dataclass(frozen=True)
class Waypoint:
    xy: tuple[float, float]
    heading: float | None = None


def departure_waypoints(key: str, start_xy: Iterable[float]) -> tuple[Waypoint, ...]:
    """Return geometry candidates using +X east, +Y north and heading radians.

    Endpoints for patient robots are off the transit lane, facing their first
    payload. KIT already carries its four cubes. The new preload arrangement
    leaves the upper-left cell clear for GREEN's westward departure.
    """
    x, y = map(float, start_xy)
    north, west, east = math.pi / 2, math.pi, 0.
    paths = {
        "lab": (
            Waypoint((.933, .640), north), Waypoint((.780, .640), west),
            Waypoint((RIGHT_X, .300)), Waypoint((RIGHT_X, .145)),
            Waypoint((.867, .145), east),
        ),
        "red": (
            Waypoint((.780, y), west), Waypoint((.780, CROSSING_Y)),
            Waypoint((NORTHBOUND_LEFT_X, CROSSING_Y), west),
            Waypoint((NORTHBOUND_LEFT_X, .831), north), Waypoint((.385, .831), east),
        ),
        "yellow": (
            Waypoint((.933, .750), north), Waypoint((.780, .750), west),
            Waypoint((.780, CROSSING_Y)), Waypoint((NORTHBOUND_LEFT_X, CROSSING_Y), west),
            Waypoint((NORTHBOUND_LEFT_X, 1.031), north), Waypoint((.385, 1.031), east),
        ),
        "kit": (
            Waypoint((x, y - .020), north), Waypoint((.780, y - .020), west),
            Waypoint((RIGHT_X, CROSSING_Y)), Waypoint((.550, CROSSING_Y), west),
        ),
        "green": (
            Waypoint((.780, y), west), Waypoint((RIGHT_X, .931)),
            Waypoint((.715, .931), west),
        ),
    }
    if key not in paths:
        raise ValueError(f"Unknown fleet role: {key}")
    return paths[key]


class DepartureSequencer:
    """Release the next packed cell once the preceding body clears the start.

    The robots continue to their first pickup concurrently. ``completed`` records
    arrival at those endpoints; ``cleared`` controls the next start-cell release.
    The west lane stays reserved until patient robots finish their off-lane
    parking moves, including any controller retries.
    All moves use the shared native simulation clock and motor controller.
    """
    def __init__(self, keys: Iterable[str]):
        keys = tuple(keys)
        unknown = set(keys) - set(DEPARTURE_ORDER)
        if unknown:
            raise ValueError(f"Unknown fleet roles: {sorted(unknown)}")
        self.order = tuple(key for key in DEPARTURE_ORDER if key in keys)
        self.completed: dict[str, float] = {}
        self.cleared: list[str] = []
        self.failure: tuple[str, str] | None = None
        self.active: str | None = None

    @property
    def ready(self) -> bool:
        return len(self.completed) == len(self.order) and self.failure is None

    def _check_failure(self):
        if self.failure:
            key, detail = self.failure
            raise RuntimeError(f"Departure stopped after {key}: {detail}")

    def depart(self, driver):
        driver.phase = "waiting for departure"
        while self.order[len(self.cleared)] != driver.key:
            self._check_failure()
            yield from driver.tick()
        self._check_failure()
        self.active = driver.key
        driver.phase = "deploy"
        try:
            for index, waypoint in enumerate(departure_waypoints(driver.key, driver.sim.pose(driver.key)[0])):
                # RED must finish turning off the west lane before YELLOW heads
                # north. KIT waits on the east side of the crossing until both
                # patient robots are parked, so it cannot catch a slow deployment
                # or enter the space used by a departure controller's retry.
                predecessors = {
                    ("yellow", 4): ("red",),
                    ("kit", 3): ("red", "yellow"),
                }.get((driver.key, index), ())
                while any(key in self.order and key not in self.completed for key in predecessors):
                    self._check_failure()
                    driver.phase = "wait for west lane departure clearance"
                    yield from driver.tick()
                driver.phase = "deploy"
                yield from driver.line(waypoint.xy, heading=waypoint.heading)
                clear_index = {"lab": 1, "red": 0, "yellow": 1, "kit": 1, "green": 0}[driver.key]
                if index == clear_index:
                    self.cleared.append(driver.key)
            if driver.key in ("red", "yellow"):
                driver.side = "left"
            self.completed[driver.key] = float(driver.sim.data.time)
        except (RuntimeError, FloatingPointError) as error:
            self.failure = (driver.key, str(error))
            raise
        finally:
            self.active = None

    def wait_until_ready(self, driver):
        driver.phase = "waiting for fleet departure"
        while not self.ready:
            self._check_failure()
            yield from driver.tick()


def left_lane(current_y: float, target_y: float) -> float:
    """Select a directional lane; turns and lane changes still need reservation.

    The centers are 145 mm apart, leaving 20 mm between 125 mm transverse envelopes.
    The inner lane's right edge is 432.5 mm, 7.5 mm before the first cylinders.
    """
    return NORTHBOUND_LEFT_X if target_y >= current_y else SOUTHBOUND_LEFT_X
