# Best-design assets and evidence

Open `best_design.blend` for the editable seven-scene CAD design. The source is `../../scripts/build_best_design_blender.py`; dimensions are in `../../best_design.json`.

| Asset | Contents |
| --- | --- |
| `fleet_arena.png` | Five robot roles on the senior preliminary arena. |
| `start_packing.png` | Starting fit, including KIT's four preloaded cubes. |
| `shared_mechanism.png` | Shared wheel drive, lift, jaws and component envelopes. |
| `lab_extension.png` | LAB's 40 mm travel stage. |
| `disc_clearance.png` | Jaw pads, sample edge and 3 mm laboratory plate. |
| `kit_magazine.png` | Three gravity doors for four kits. |
| `lab_support_camera.png` | Passive front support and moving inspection camera. |
| `cad_geometry_audit.json`, `visual_qa.json` | Dimensions, starting fit, source and asset hashes. |
| `chassis_fit_mockup_mm.stl` | Chassis fit mockup in millimeters. |
| `final_frame_160_top.mp4`, `final_frame_160_oblique.mp4` | Native simulation playback of 160 points at 79.60 seconds, followed by the five-second hold. |
| `final_frame_nominal/` | Saved native scene, metadata, state trajectory and mission report for those videos. |
| `drive_time_penalty_comparison.png` | Matched PPO comparison of 0.05/s and 0.5/s time penalties. |
| `search/` | Earlier six-speed comparison before the final support, camera and routing changes. |
| `search_final/` | Final nine-profile comparison, selected settings and separate held-out evaluation. |
| `fleet_speed_comparison.png`, `fleet_mission_timeline.png` | Measured profile outcomes and role phases during the nominal mission. |

The adjacent `../best_drive/model/weights.npz` contains the trained drive network. `../best_vision/refined/model.ts` contains the selected CNN. Their directories preserve training histories, validation selection and independent final tests. `../best_vision/onboard_sample_camera_final_frame/` contains the actual onboard camera coupon and accepted-view error measurements.

The mission uses simulator poses and a geometric controller. The drive network and RGB modules have separate measured benchmarks. The video JSON manifests identify each recording's source hashes and exact playback method. Saved-state playback does not rerun contact physics.

See [mechanical design](../../docs/best_design_mechanics.md) and [training and timing](../../docs/best_design_training.md) for the operating plan, G4 commands, comparisons and physical assumptions.
