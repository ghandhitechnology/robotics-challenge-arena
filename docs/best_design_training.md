# Training and timing evidence

The objective is to finish all 16 scoring deliveries within 120 seconds. Training rewards elapsed time more strongly, while checkpoint selection keeps successful completion ahead of speed.

## Time cost

The native drive comparison starts two PPO candidates from the same selected behavior-cloning and DAgger weights. Both use the same episode seeds, physical parameter distributions and action noise. The original candidate loses 0.05 reward per simulated second. The requested candidate loses 0.5 per second, ten times as much. At the 20 ms control interval, their time costs are 0.001 and 0.01 per step.

Each primitive also receives progress shaping, a 10-point success bonus, a 20-point timeout penalty and a 25-point physical-failure penalty. Reward uses simulated elapsed time, so GPU throughput cannot improve the reward. The export is selected by validation reach rate, then successful completion time, then position and heading error. The best pre-PPO candidate remains eligible. Failed trials remain in the denominator.

The mission report records a separate task objective, `10 * official_score - 0.5 * declaration_seconds`. A 10-point delivery adds 100 objective units; the entire 120-second time cost is 60. Skipping a delivery cannot improve that task objective solely by saving time. This report objective does not turn the geometric mission planner into a learned policy.

## G4 runtime

The connected Google Colab G4 runtime reports an NVIDIA RTX PRO 6000 Blackwell Server Edition with 97,887 MiB of VRAM. The observed software is PyTorch 2.11.0 with CUDA 12.8 and MuJoCo 3.12.0. Native EGL rendering identifies the NVIDIA GPU. The runtime has 48 logical CPUs, which run independent native wheel-contact episodes while the GPU trains the networks.

Runtime details are saved in `output/best_design/g4_runtime.json`. Source commits and SHA-256 hashes accompany the model and dataset artifacts.

## Crop perception

The six-class CNN predicts background, red, yellow, green, kit or sample, plus a projected object center. Each 72 by 72 image comes from the native arena with randomized camera position, lighting, materials, object poses, clutter and physical occlusion. The frozen split contains 24,000 training images, 3,000 validation images and 3,000 test images.

| Matched run | Epochs executed | Selected epoch | Validation accuracy | Foreground center p95 |
| --- | ---: | ---: | ---: | ---: |
| Control, learning rate 0.003 | 33 | 21 | 99.6667% | 2.2201 px |
| Refined, learning rate 0.001 | 91 | 66 | 99.7667% | 2.0433 px |

The refined checkpoint improved both validation classification accuracy and center p95. Its selected model hash is `7532f12a04597ea9b8388e399dad7c15a52e746b602bdb4c0c8abf598b942266`. The matched dataset manifest hash is `62a3e7d61b827ea9dc82b8c5ea8406e65e0848eb3aec2224d1dfa73ec7315fd8`.

A 0.995 confidence threshold was selected before the final test. On validation foreground crops it accepts 98.36%, with 99.9187% classification accuracy and 1.9368 px center p95. The sample class alone accepts 99.2%, with 1.7468 px center p95 and a 4.0363 px maximum. Classification confidence does not bound localization error, so the sample mechanism still requires a calibrated precision alignment step for its 2 mm radial clearance.

The independent final test contains 3,000 images. The frozen refined checkpoint classifies 2,991 correctly, for 99.70% accuracy. Foreground center error averages 0.8864 px, with 1.9770 px p95 and a 9.1671 px maximum. G4 batch-one end-to-end inference measures 0.4984 ms median and 0.5253 ms p95 across 200 iterations. These latency measurements use the Colab GPU. The compute board in the robot still needs its own latency measurement. The full report is `output/best_vision/refined/test.json`.

The initial baseline used a different image-array hash. A JSON tuple/list comparison incorrectly rejected dataset reuse, so a matched control was retrained on the exact regenerated arrays used by the refinement. The initial artifacts remain under `baseline/`; `comparison.json` compares only the matched runs. Dataset reuse now normalizes JSON configuration and verifies all frozen file hashes.

See `docs/best_vision.md` for the inference API and camera contract. The existing geometric mission uses simulator poses; its scores must be reported separately from RGB perception metrics.

## Native drive training

The exported network has eight observations: local goal position, final heading error, both wheel speeds and measured forward/yaw velocity. It outputs forward and yaw commands. The controller applies those commands through torque-limited wheels, with native floor contact at a 1 ms physics interval.

Training includes forward lines, reverse lines and turns. Each episode varies wheel friction, motor strength and no-load speed. Two DAgger rounds label states reached by the current learner. PPO then compares the two time costs using actual native rollouts. The precision gate is 2 mm position error and 0.015 rad heading error within 12 seconds. This benchmark covers one unloaded LAB base in the center corridor.

The G4 source snapshot is commit `9ce704984e89316e8d1acb29c69047f743feeeb5`:

```bash
python scripts/train_best_drive.py --device cuda --compile --workers 24 \
  --train-episodes 720 --validation-episodes 120 --test-episodes 120 \
  --dagger-rounds 2 --dagger-episodes 180 \
  --closed-loop-validation-episodes 40 \
  --ppo-updates 20 --ppo-episodes-per-update 24 \
  --time-penalty-per-second .5 --comparison-time-penalty-per-second .05
```

The final drive test uses frozen episode seeds and runs the selected learner, the geometric teacher and zero commands under matched physical variation:

```bash
python scripts/test_best_drive.py --workers 24 \
  --min-reach-rate .95 --max-final-distance-m .002
```

The executed run took 442.1 seconds. Its initial dataset contains 73,002 native training transitions from 720 episodes. DAgger added 56,862 and 49,102 transitions across two 180-episode rounds. The three supervised phases ran 200, 58 and 34 epochs. PPO executed 20 updates per time-cost candidate, with 24 native episodes per update. The network contains 26,050 parameters.

| Validation candidate | Targets reached | Mean episode duration, including timeouts | Position p95 |
| --- | ---: | ---: | ---: |
| PPO, time cost 0.05/s | 29/40 | 5.5305 s | 11.7902 mm |
| PPO, time cost 0.5/s | 40/40 | 2.6340 s | 1.9921 mm |

The selected 0.5/s weights have SHA-256 `851484bad87154380fae57ee4a85717004ed4dc9b8cba1c153446314ecb8e047`. The independent final test reaches 119 of 120 targets, with a 95.43% Wilson lower confidence bound, 1.9864 mm position p95 and no terminal physical failures. Successful episodes average 2.8513 seconds. The teacher reaches 107 of the same 120 targets; its successful episodes average 1.6510 seconds. The learner improves precision reliability in this benchmark, while the teacher completes its successful trials faster.

The first full-fleet hybrid trial, using learned coarse drive and geometric fine docking, scored 70 points. Its changed arrival times disrupted corridor sharing. The geometric fleet remains the validated mission controller. Both implementations and their results are retained for comparison.

## RGB sample refinement

The separate refiner maps visible RGB rim pixels onto a calibrated plane and fits the known 28 mm sample radius. The frozen CNN supplies the proposal and confidence. It uses no depth image or segmentation IDs. Thresholds were selected on 160 development views, then tested on 320 disjoint views. It accepted 300 of 320 views, with 0.232 mm median, 0.493 mm p95 and 1.079 mm maximum accepted center error. Rejected views require another observation before placement. See `docs/best_sample_refiner.md` for the camera and error assumptions.

The final camera coupon uses the actual moving LAB camera and its open frame, with the 474 g robot and passive support. At unchanged thresholds it accepts 45 of 48 floor samples, with 0.182 mm p95 and 0.374 mm maximum lateral error. All 48 held-disc trials grasp successfully; 46 views are accepted, with 0.306 mm p95 and 0.562 mm maximum. It rejects all 36 yellow-cylinder negatives. These measurements concern accepted views. The mission must request another observation when the detector rejects a frame. See `docs/best_onboard_sample_camera.md` for calibration, output coordinates and the final source hashes.

## Full mission timing

The original five-robot geometric fleet completed all deliveries in 115.14 seconds. The first G4 comparison evaluated six common speed limits across six matched physics seeds, 36 missions total. Faster common settings caused LAB sample failures. The next revision gives LAB its own limits, adds its passive front support, takes GREEN's second upper cylinder before waiting for LAB, and stops once all released objects score 160. It then checks the score throughout another five seconds. Robots do not need to finish unused parking movements before declaration.

The source-stable nominal run at commit `bbc54c3` scores 160 in 79.60 seconds, with all 50 hold checks retaining 160. LAB uses limits of 0.35 m/s and 2.5 rad/s; the other robots use 0.48 m/s and 4 rad/s. The full recording is under `output/best_design/final_frame_nominal/`. This is a 30.9% reduction from the original 115.14-second run, across mechanical and routing changes as well as speed settings.

```bash
python scripts/run_best_fleet.py --drive-limits .48 4 \
  --lab-drive-limits .35 2.5 --green-upper-first \
  --output output/best_design/reproduction
```

The final parameter comparison uses nine profiles and six matched tuning seeds. Selection prioritizes full success rate, then mean official score, then the time-penalized objective. A separate 100-seed test evaluates only the frozen selected profile. Material variation includes wood density by ±15%, material friction by ±20%, wheel friction scaled by 0.75–1.15, and individual motor strength by 0.8–1.0. Arena geometry and task object marks stay fixed. These are unmeasured engineering distributions.

The 54 final tuning missions took 498.8 seconds on G4, using 12 native workers. All source hashes match the frozen `bbc54c3` snapshot, and no episode raised an exception. Four profiles completed all six tuning missions. The selected mixed-speed profile averaged 81.20 seconds, with 89.40 seconds p95. Its six individual times were 85.2, 74.7, 76.2, 80.7, 79.6 and 90.8 seconds. Its maximum fleet tilt was 7.282 degrees.

| Final geometry, speed profile | Full success | Mean successful time | Mean official score |
| --- | ---: | ---: | ---: |
| All 0.35 m/s, 2.5 rad/s | 6/6 | 114.1 s | 160.0 |
| All 0.40 m/s, 2.75 rad/s | 6/6 | 103.6 s | 160.0 |
| All 0.42 m/s, 3.0 rad/s | 5/6 | 95.8 s | 143.3 |
| All 0.42 m/s, 3.5 rad/s | 3/6 | 85.1 s | 118.3 |
| All 0.45 m/s, 3.5 rad/s | 2/6 | 83.0 s | 76.7 |
| All 0.48 m/s, 4.0 rad/s | 2/6 | 77.1 s | 93.3 |
| Couriers 0.40 m/s, 2.75 rad/s; LAB 0.35/2.5 | 3/6 | 104.1 s | 125.0 |
| Couriers 0.45 m/s, 3.5 rad/s; LAB 0.35/2.5 | 6/6 | 85.3 s | 160.0 |
| Couriers 0.48 m/s, 4.0 rad/s; LAB 0.35/2.5 | 6/6 | 81.2 s | 160.0 |

Mixed-speed profiles also use GREEN's upper-first route. Candidate speed and route changes therefore belong to one profile comparison. Successful-time means exclude failed episodes; the success and score columns retain every trial. Selection and complete episode records are in `output/best_design/search_final/`.

## Mechanical evidence and simulation limits

The isolated LAB mechanism seated and released three samples for 30 points in 59.8 simulated seconds. The KIT gravity magazine released all four kits for 40 points in 39.44 seconds. Both recordings include a further five-second stable view. Their scenes retain the original arena objects, with only the tested robot present. Shared-fleet timing is measured separately.

`scripts/capture_best_fleet.py` records saved native states at real-time speed and writes source hashes alongside the MP4. It does not interpolate robot or object poses. The clips in `output/best_design` show simulation playback.

The dynamic model uses simplified collision shapes, a 450 g base mass budget and a 474 g LAB budget including its support and camera. Some mast and servo details in Blender are absent from the dynamic model. Tape is rigid, and motor/friction distributions are engineering estimates. Hardware measurement, camera calibration, loaded mechanism tests and repeated physical full-field runs are required to establish real competition performance. The finite searches and early stopping establish measured comparisons; they cannot establish that every possible design or optimization has been exhausted.
