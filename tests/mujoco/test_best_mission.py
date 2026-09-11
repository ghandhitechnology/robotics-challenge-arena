import math
import unittest

import mujoco
import numpy as np

from arena_mujoco.best_fleet import FleetSimulation, wrap
from arena_mujoco.best_mission import Driver


class FleetLineRecoveryTests(unittest.TestCase):
    def test_reaches_waypoint_from_pure_cross_track_error(self):
        sim = FleetSimulation(only=["green"], drive_limits=(.48, 4.))
        target = np.array([.225, .300])
        # Initialize in the open west corridor, abreast of the destination.
        # All subsequent motion uses the native wheel motors and contacts.
        address = sim.model.joint("fleet_green_competition_robot_free").qposadr[0]
        sim.data.qpos[address:address + 7] = [.241, .300, .025, 1., 0., 0., 0.]
        mujoco.mj_forward(sim.model, sim.data)
        driver = Driver(sim, "green")
        driver.phase = "cross-track recovery"

        for _ in driver.line(target, heading=math.pi / 2):
            sim.step()

        position, heading = sim.pose("green")
        self.assertLess(float(np.max(np.abs(position - target))), .005)
        self.assertLess(abs(wrap(heading - math.pi / 2)), .012)


if __name__ == "__main__":
    unittest.main()
