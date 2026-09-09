"""Measurement formulas and the boundary between estimates and fitted inputs."""
import copy
import math
import unittest

from scripts.calibrate_mujoco import TEMPLATE, calibrate, volume_m3


class CalibrationTests(unittest.TestCase):
    def test_friction_density_and_linear_modulus(self):
        sample = {"schema_version": 1,
                  "contacts": [{"pair": "wood_paper", "static_incline_deg": 45,
                                "kinetic_pull": {"mass_kg": 0.1, "force_n": 0.4905},
                                "sliding_coast": {"initial_speed_m_s": 0.9904544411531507,
                                                  "distance_m": 0.1}}],
                  "densities": [{"material": "wood", "mass_kg": 0.006,
                                 "geometry": {"shape": "box", "size_m": [0.01, 0.02, 0.05]}}],
                  "tape": {"tensile": {"width_m": 0.02, "thickness_m": 0.00015,
                                       "gauge_length_m": 0.1,
                                       "samples": [{"extension_m": 0, "force_n": 0.2},
                                                   {"extension_m": 0.001, "force_n": 0.5},
                                                   {"extension_m": 0.002, "force_n": 0.8}]},
                           "curl_radius_m": 0.05, "curl_sign": -1}}
        patch, report = calibrate(sample)
        self.assertAlmostEqual(patch["materials"]["wood_paper"][0], 1)
        self.assertAlmostEqual(patch["materials"]["wood_paper"][1], 0.5)
        self.assertAlmostEqual(patch["wood_density_kg_m3"], 600)
        self.assertAlmostEqual(patch["tape"]["young_pa"], 10_000_000)
        self.assertEqual(patch["tape"]["rest_curvature_m_inv"], -20)
        self.assertTrue(report["observations"])

    def test_lab_holes_and_peel_angle_units(self):
        geometry = {"shape": "perforated_plate", "size_m": [0.15, 0.345, 0.003],
                    "holes": [{"diameter_m": 0.06, "count": 3}]}
        self.assertAlmostEqual(volume_m3(geometry), (0.15 * 0.345 - 3 * math.pi * 0.03**2) * 0.003)
        patch, report = calibrate({"schema_version": 1,
                                   "drops": [{"drop_height_m": 0.4, "rebound_height_m": 0.1}],
                                   "tape": {"peel": [{"angle_deg": angle, "width_m": 0.02,
                                                      "force_n": 3.6} for angle in (90, 180)]}})
        self.assertEqual(patch, {})
        self.assertAlmostEqual(report["observations"][0]["restitution_estimate"], 0.5)
        peel = report["observations"][1:]
        self.assertAlmostEqual(peel[0]["resistance_n_per_m"], 180)
        self.assertAlmostEqual(peel[0]["ideal_energy_estimate_j_m2"], 180)
        self.assertAlmostEqual(peel[1]["ideal_energy_estimate_j_m2"], 360)

    def test_blank_template_retains_base_and_rejects_invalid_measurements(self):
        template = copy.deepcopy(TEMPLATE)
        patch, report = calibrate(template, {"tape": {"density_kg_m3": 1400}})
        self.assertEqual(patch, {"tape": {"density_kg_m3": 1400}})
        self.assertTrue(report["skipped_incomplete_measurements"])
        invalid = {"schema_version": 1, "contacts": [{"pair": "wood_paper", "static_incline_deg": 10,
                   "kinetic_pull": {"mass_kg": 0.1, "force_n": 1}}]}
        with self.assertRaisesRegex(ValueError, "kinetic friction exceeds static"):
            calibrate(invalid)
        with self.assertRaisesRegex(ValueError, "Rebound height exceeds"):
            calibrate({"schema_version": 1, "drops": [{"drop_height_m": 0.1, "rebound_height_m": 0.2}]})


if __name__ == "__main__":
    unittest.main()
