# RGB sample docking refiner

`BestSampleGeometryRefiner` refines a CNN sample proposal from a 224×224 RGB crop. It segments the yellow sample rim, maps the visible rim pixels through calibrated camera rays onto the sample-center plane, and robustly fits the known 28 mm radius there. The inference path reads RGB, the CNN class/center/confidence, crop origin, camera calibration, and plane geometry. It does not read segmentation IDs, depth, object pose, or simulator state.

The caller must provide full-image pinhole intrinsics and the camera-to-world rotation and position. Pixel coordinates use left/top as the origin. MuJoCo cameras can pass `data.cam_xmat[camera_id].reshape(3, 3)` directly as the camera-to-world rotation. The fitted yellow silhouette is the top rim, so a resting 5 mm disc uses the horizontal plane 5.02 mm above the board in this model. A 32-view onboard development comparison measured 0.186 mm p95 camera-lateral error with that plane versus 0.394 mm at the 2.52 mm center plane.

```python
from arena_mujoco.best_vision import BestVisionPredictor
from arena_mujoco.best_vision_refiner import (
    BestSampleGeometryRefiner, PinholeCamera, Plane, RefinerThresholds,
)

predict = BestVisionPredictor("output/best_vision/refined/model.ts", device="cuda")
coarse = predict(crop_rgb_224)
camera = PinholeCamera(
    fx, fy, cx, cy,
    tuple(camera_position_world),
    tuple(map(tuple, rotation_camera_to_world)),
)
result = BestSampleGeometryRefiner(RefinerThresholds()).refine(
    crop_rgb_224,
    coarse,
    camera,
    Plane((0.0, 0.0, sample_center_height_m)),
    crop_origin_px=(crop_left, crop_top),
)
if result["accepted"]:
    sample_xy_m = result["world_point_m"][:2]
    local_alignment_xy_m = result["camera_lateral_xy_m"]
else:
    reason = result["rejection_reason"]
```

The independent native-render evaluation uses 160 development views to select quality thresholds, then 320 disjoint final-test views. Camera height spans 100–180 mm, vertical field of view spans 48–68 degrees, tilt spans 0–8 degrees, and native lights, materials, clutter, sensor noise, and zero to two physical RGB occluders vary per view. The frozen CNN gate remains 0.995 confidence.

The general crop test accepted 300 of 320 views (93.75%). Accepted position error was 0.232 mm median, 0.493 mm p95, and 1.079 mm maximum. Its original evaluator intersected at the disc center plane; the later onboard test uses the selected visible-rim plane. The integrated carriage-camera coupon reports the practical mounted-camera result separately.

Run the evaluation with:

```bash
python scripts/test_best_sample_refiner.py \
  --model output/best_vision/refined/model.ts \
  --output output/best_vision/sample_refiner
```

The report is `output/best_vision/sample_refiner/test.json`; the fitted and rejected RGB examples are in `test_examples.png`.
