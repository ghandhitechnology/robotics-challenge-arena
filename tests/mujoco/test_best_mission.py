import math
import unittest

import mujoco
import numpy as np

from arena_mujoco.best_fleet import FleetSimulation, wrap
from arena_mujoco.best_mission import Driver


class FleetLineRecoveryTests(unittest.TestCase):
    def test_unloaded_deployment_recovers_large_cross_track_error(self):
        sim = FleetSimulation(only=["green"], drive_limits=(.48, 4.))
        target = np.array([.225, .300])
        address = sim.model.joint("fleet_green_competition_robot_free").qposadr[0]
        sim.data.qpos[address:address + 7] = [.300, .300, .025, 1., 0., 0., 0.]
        mujoco.mj_forward(sim.model, sim.data)
        driver = Driver(sim, "green")
        driver.phase = "deploy"
        names = tuple(sim.model.geom(i).name for i in range(sim.model.ngeom))
        robot = np.array([name.startswith("fleet_green_") for name in names])
        floor = np.array([name == "floor" or name.startswith("Tape_") for name in names])

        for _ in driver.line(target, heading=math.pi / 2):
            sim.step()
            pairs = sim.data.contact.geom
            external = np.any(robot[pairs], axis=1) & ~np.all(robot[pairs], axis=1)
            obstacles = external & ~np.any(floor[pairs], axis=1)
            self.assertFalse(np.any(obstacles), "Deployment recovery hit an obstacle")

        position, heading = sim.pose("green")
        self.assertLess(float(np.max(np.abs(position - target))), .005)
        self.assertLess(abs(wrap(heading - math.pi / 2)), .012)

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
