# LAB onboard sample camera

The LAB inspection camera is fixed to `fleet_lab_comp_lift_body`, so it follows both the 0–40 mm extension and 0–45 mm lift. Its optical center is carriage-local `[0, 0, 0.0841]` m and looks along local `-Z` with a 58-degree vertical field of view. The gripper site is 100 mm below the camera. Four collision-enabled frame walls surround a 12×12 mm clear aperture and weigh 4 g total. A 2 g, 25×24×1.6 mm sensor board sits behind the optics at carriage z=85.3 mm. The full assembly reaches 106.1 mm at the starting lift position, inside our 110 mm design budget. The integrated LAB mass is 474 g and the adopted front caster remains active in the coupon. At the frame's lower face the full 58-degree optical cone has a 3.94 mm half-width inside the 6 mm aperture half-width.

The RGB pipeline uses the frozen CNN for sample identity and a calibrated plane-circle fit for precision. Development views selected the visible top rim: floor samples use world z=5.02 mm, while held samples use a plane 99.1 mm down the camera axis, 0.9 mm above the gripper site. The refiner returns `camera_lateral_xy_m`, which supports local tool alignment without a world robot pose. Camera calibration and the floor or gripper plane can be expressed in the robot/body frame using the fixed mount and IMU attitude.

The final-frame native coupon used 48 floor trials, 48 physical held-disc trials, and 36 yellow-cylinder negatives without changing the selected thresholds or seed. All 48 grasp attempts lifted the sample. Floor views accepted 45/48, with 0.076 mm median, 0.182 mm p95, and 0.374 mm maximum camera-lateral error. Held views accepted 46/48, with 0.083 mm median, 0.306 mm p95, and 0.562 mm maximum. All 36 yellow-cylinder views were rejected. Rejected sample views disagreed with the CNN coarse center and should trigger another observation rather than a docking move.

```bash
python scripts/select_best_onboard_sample_plane.py
python scripts/test_best_onboard_sample_camera.py \
  --model output/best_vision/refined/model.ts \
  --output output/best_vision/onboard_sample_camera_final_frame
```

The final report and review sheet are `output/best_vision/onboard_sample_camera_final_frame/test.json` and `onboard_examples.png`. The coupon measures visibility, grasp retention, and local RGB geometry. Packaging tolerance and closed-loop sample seating remain physical integration tasks.
