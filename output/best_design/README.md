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
| `validated_fleet_top.mp4`, `validated_fleet_oblique.mp4` | Native simulation playback of the reserved-lane controller: 160 points at 81.10 seconds, followed by the five-second hold. |
| `traffic_reservation_development/nominal/` | Saved scene, metadata, state trajectory and mission report for those videos. |
| `final_frame_160_top.mp4`, `final_frame_160_oblique.mp4`, `final_frame_nominal/` | Earlier 79.60-second run before departure reservations and settled-release confirmation. |
| `final_drive_time_penalty_comparison.png` | Final 474 g retrain's matched PPO validation comparison of 0.05/s and 0.5/s time penalties. |
| `drive_time_penalty_comparison.png` | Historical 450 g drive comparison. |
| `final_learned_fleet_trial/` | Hybrid learned-coarse/geometric-precision mission report and trajectory: 160 points at 116.30 seconds. |
| `search/` | Earlier six-speed comparison before the final support, camera and routing changes. |
| `search_final/` | Nine-profile comparison and initial independent test: 97/100 full successes. |
| `search_verified/` | Evaluation after the departure and settling corrections, with a new seed schedule. |
| `fleet_speed_comparison.png`, `fleet_mission_timeline.png` | Measured profile outcomes and role phases during the nominal mission. |

The adjacent `../best_drive_final/model/weights.npz` contains the selected 474 g drive network. Its matched 0.5/s checkpoint remains beside it as `ppo_requested.npz`. The earlier 450 g weights remain under `../best_drive/model/` as historical evidence. `../best_vision/refined/model.ts` contains the selected CNN. Their directories preserve training histories, validation selection and independent final tests. `../best_vision/onboard_sample_camera_final_frame/` contains the actual onboard camera coupon and accepted-view error measurements.

The mission uses simulator poses and a geometric controller. The drive network and RGB modules have separate measured benchmarks. The video JSON manifests identify each recording's source hashes and exact playback method. Saved-state playback does not rerun contact physics.

See [mechanical design](../../docs/best_design_mechanics.md) and [training and timing](../../docs/best_design_training.md) for the operating plan, G4 commands, comparisons and physical assumptions.
