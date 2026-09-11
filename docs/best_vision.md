# Best-design crop perception

This pipeline trains a six-class CNN on native MuJoCo renders of the senior preliminary arena. The classes are `background`, `red`, `yellow`, `green`, `kit`, and `sample`. A second head predicts the target body's projected center in normalized crop coordinates.

The tested camera contract is a 72 by 72 perspective crop from a downward inspection camera 100 to 180 mm above the playing surface. Vertical field of view varies from 48 to 68 degrees, and camera tilt varies up to 8 degrees. Each foreground crop contains one proposed target in its central region. Other arena objects can cross the crop edge, and physical MuJoCo boxes provide partial occlusion. This is an ROI classifier and refiner, so a full-frame proposal or tracking stage must supply the crop.

The generator starts from `build_arena("senior_preliminary")`. It renders the existing paper, tape, laboratory, cylinders, medical kits, and biological samples. It randomizes fixed-camera geometry, lights, material colors, object free-joint poses, edge clutter, occluders, exposure, and sensor noise. Train, validation, and test arrays use separate seeds and frozen file hashes.

Run a local end-to-end check with small counts:

```bash
.venv/bin/python scripts/train_best_vision.py --device cpu --cpu-smoke \
  --train-count 96 --validation-count 36 --test-count 36 \
  --image-size 48 --channels 16 --settle-steps 1 --epochs 2 --patience 2 \
  --batch-size 24 --workers 0 --dataset /tmp/best-vision-data \
  --output /tmp/best-vision-model
.venv/bin/python scripts/test_best_vision.py --device cpu \
  --dataset /tmp/best-vision-data --model /tmp/best-vision-model \
  --batch-size 36 --workers 0 --latency-iterations 20
```

Use an EGL-backed Colab GPU for the real run. The default creates 24,000 training crops and 3,000 crops for each held-out split. CUDA is mandatory unless `--cpu-smoke` is present.

```bash
MUJOCO_GL=egl python scripts/train_best_vision.py --render-backend egl \
  --device cuda --compile --output output/best_vision/model
python scripts/test_best_vision.py --device cuda \
  --model output/best_vision/model --dataset output/best_vision/dataset
```

Training selects a checkpoint from validation loss and fits the confidence temperature on validation logits. It never reads test images. `test_best_vision.py` verifies the frozen hashes, runs the final split once, writes classification, calibration, center error, confusion, and latency metrics, and reports the actual GPU model.

Load the exported inference API with:

```python
from arena_mujoco.best_vision import BestVisionPredictor

predict = BestVisionPredictor("output/best_vision/model", device="cuda")
result = predict(rgb_crop)  # H x W x 3 uint8 RGB
```

The result has `class_name`, `class_index`, calibrated `confidence`, `probabilities`, `center_normalized_xy`, and `center_pixel_xy`. Center values are meaningful for foreground predictions. Crop metrics do not establish robot or fleet task success.
