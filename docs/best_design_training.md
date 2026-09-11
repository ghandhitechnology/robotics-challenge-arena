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

Training includes forward lines, reverse lines and turns. Each episode varies wheel friction, motor strength and no-load speed. Two DAgger rounds label states reached by the current learner. PPO then compares the two time costs using actual native rollouts. The precision gate is 2 mm position error and 0.015 rad heading error within 12 seconds. This first benchmark covers one unloaded 450 g LAB base in the center corridor, before the passive support and inspection camera additions.

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

### Final 474 g drive retrain

The final drive model was retrained after the passive support and inspection camera raised LAB to 474 g. The frozen native dataset has 72,562 transitions from 720 episodes. Two DAgger rounds added 44,834 and 29,098 learner-visited transitions. The three supervised phases ran for 200, 200 and 134 epochs. Each PPO candidate then ran 20 updates with 24 native episodes per update. Dataset generation and training took 637.1 seconds on G4.

| Final validation candidate | Targets reached | Successful mean | Mean including timeouts | Position p95 |
| --- | ---: | ---: | ---: | ---: |
| PPO, time cost 0.05/s | 40/40 | 1.9245 s | 1.9245 s | 1.9838 mm |
| PPO, time cost 0.5/s | 39/40 | 1.8821 s | 2.1350 s | 1.8999 mm |

Selection keeps reach rate ahead of completion time, so the final export uses the 0.05/s candidate. Its SHA-256 is `dbe14271f520a306e2bf11f2338f70ef72accf28d34fc8650a5371ef2a3cef8e`. The requested 0.5/s checkpoint remains at `output/best_drive_final/model/ppo_requested.npz`; the selected checkpoint is also stored as `ppo_baseline.npz` and `weights.npz`. The matched comparison plot is `output/best_design/final_drive_time_penalty_comparison.png`.

On the independent 120-episode test, the selected learner reached 119 targets. Its successful episodes averaged 2.020840 seconds and its final position error p95 was 1.979758 mm. The geometric teacher reached 118 targets and averaged 1.788644 seconds across its successful episodes. Neither controller had a terminal physical failure.

Full-fleet learned-drive integration remains experimental. The historical `a7e420d` nominal trial scored 160 at 116.30 seconds, and the later `f506bc2` nominal trial scored 160 at 117.90 seconds. Under the frozen `e2be32d` source, the hybrid trial reaches only 140 points at the 120-second limit. LAB times out on the line to its third sample placement, GREEN remains unfinished, and all 50 hold checks stay at 140. The report records 6,098 learned calls and 16,356 geometric calls under `output/best_design/deployment_point_recovery_development/learned_nominal/`.

A separate nominal experiment moved the learned-to-geometric handoff distance from 30 mm to 50 mm and 80 mm. The three runs scored 140, 140 and 120. The larger handoffs were rejected, their source patch was removed, and the reports remain under `output/best_design/learned_handoff_development/`. These mission trials do not change the independent 119/120 primitive result or the selection of the geometric controller for the full mission.

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

The first independent 100-mission test achieved 97 full successes. Successful runs averaged 79.912 seconds, with 90.84 seconds p95. The Wilson lower 95% confidence bound was 91.548%. Two runs stopped at 60 points after YELLOW/KIT traffic conflicts. A third reached 160 but briefly fell to 150 during the final hold. No numerical exception occurred. The report's 158-point mean uses declaration/final endpoint scores; the mean using the minimum score across the hold is 157.9. The complete original test remains in `search_final/test.json`.

Those failures motivated two changes. YELLOW now waits for RED to finish turning off the west lane, and KIT waits east of the crossing until both patient robots finish deployment. Early declaration before parking requires all 16 release events and six successful score checks at 0.1-second intervals, spanning half a second while the robots continue retreating. Normal program completion can also end the mission. Every termination retains the separate five-second hold with 50 score checks. The revised nominal run finishes at 81.10 seconds, with a 16.22-second deployment and 7.161-degree peak fleet tilt. Both traffic development cases finish with 160, at 80.30 and 81.10 seconds. The separate transient-hold development case passes at 83.00 seconds.

The repaired scheduler is commit `a7e420d099ac78404725ab21154304d4775c98cc`. Its evaluation uses a 12-seed check of the already-selected profile, followed by a new 100-seed test. Generator seeds are 202609118 and 202609117 respectively. The new test seeds have no overlap with the previous 100. Results belong under `output/best_design/search_verified/`; the previous test becomes development evidence for this revision.

The 12 validation missions all passed, averaging 81.142 seconds. The following 100-seed test passed only 88 missions, averaging 82.713 seconds among successes, with 90.375 seconds p95. All 12 failures scored 120 after LAB dropped its first sample in transit. Source hashes matched all 14 frozen files, and no numerical exception occurred. The complete result remains in `search_verified/test.json`.

The G4 replay of seed 1300600412 identifies contact between KIT's chute front and LAB's upper left gripper pad at 18.74 seconds. LAB loses its sample at 18.80 seconds, after grasping at 15.42 seconds and before any release command. The same scene and randomized parameters pass locally because the arrival timing differs. This is a crossing-clearance failure sensitive to simulation platform timing.

KIT had reached x=0.550 while heading west, but its route state still said right. Its first delivery therefore returned to x=0.830 before crossing west again. Commit `f506bc2df95ef7b2ebcefee8f07c1d234d0235ef` sets KIT's route state to left after departure, removing 0.56 m of backtracking while retaining the departure reservations. Its historical local nominal mission finishes at 80.80 seconds. Three development seeds finish at 77.00, 76.40 and 76.70 seconds; all four keep 160 throughout the hold. Contact evidence and reports are in `lab_transit_development/`.

The complete `f506bc2` G4 test passes 297 of 300 independent missions. Successful runs average 80.948 seconds, with 91.50 seconds p95; the Wilson lower 95% confidence bound is 97.102%. Two GREEN deployment timeouts score 130 each. A YELLOW deployment timeout blocks KIT and LAB, leaving 30 points. All three timeouts occur at retreat waypoints with remaining cross-track error. No sample-drop recurrence or numerical exception appears in this test. All 14 source hashes match the frozen commit. The full report is `search_direct_west/test.json`; it becomes development evidence for the recovery correction.

Commit `f16f2f2e87f037f5ba6380662b541931c06637a7` fixes the geometric driver's cross-track recovery. The old recovery waypoint was 25 mm behind the target, which could leave a differential-drive robot with the same lateral error on its retry. The new waypoint is 25 mm behind the robot's actual position along the approach direction. The focused native regression times out under the old calculation and passes under the new calculation.

Four local full missions check the recovery correction. The nominal run finishes at 80.80 seconds. GREEN seed 1092317225 finishes at 86.30 seconds, LAB seed 1300600412 at 77.00 seconds and YELLOW seed 1567586323 at 78.10 seconds. Each run scores 160 and retains it through all 50 hold checks. Reports are in `current_pose_recovery_development/`.

The `f16f2f2` G4 validation passes all 24 missions, averaging 81.363 seconds with 90.70 seconds p95. The known GREEN seed 1092317225 passes at 86.20 seconds, but YELLOW seed 1567586323 still fails at 30 points and GREEN seed 1433951863 at 130. Two short retreats cannot reliably remove their 20 mm and 73 mm cross-track errors. Those results remain in `current_pose_recovery_development/g4_*_verified/`.

Commit `e2be32d42e5dd0b8486881919f37c2d4a13cc419` adds bounded point guidance only for unloaded RED, YELLOW and GREEN during deployment. After a stall, the robot steers toward the waypoint, stops within half the requested tolerance and restores the required approach heading. The original final position and heading checks still apply. Each recovery has an eight-second timeout and consumes an existing retry. KIT, LAB and loaded approaches keep their approach geometry. The new 75 mm native regression fails under the prior controller and passes with this change without sampled obstacle contacts.

The e2 local nominal mission scores 160 at 80.80 seconds and retains 160 through all 50 hold checks. Its state and command arrays equal the f16 nominal recording. The e2 hybrid repeats the 140-point LAB timeout with 6,098 learned calls and 16,356 geometric calls. Both source-stable reports and trajectories are under `deployment_point_recovery_development/`, and the current geometric and experimental hybrid videos use those recordings.

The three targeted e2 G4 replays now pass: seed 1092317225 finishes at 86.30 seconds, seed 1433951863 at 82.40 seconds and seed 1567586323 at 79.50 seconds. Each scores 160 and retains it through all 50 hold checks. Their reports are under `deployment_recovery_development/g4_*/`.

The fresh e2 G4 validation passes all 24 missions generated from seed 202609124. Successful runs average 81.679 seconds, with 89.85 seconds p95 and no exceptions. All 14 source hashes match `e2be32d`.

The predeclared final test uses generator seed 202609123 and contains 300 unique missions with no seed overlap against the previous 500 test missions. It succeeds in 299 of 300 missions, a 99.667% success rate with a 98.136% Wilson lower 95% confidence bound. Successful missions average 80.684 seconds and have an 89.52-second p95. Mean score is 159.967, minimum score is 150, maximum fleet tilt is 12.353 degrees and no episode raises an exception.

The only failure is seed 1495332060. GREEN times out while delivering `Cylinder_Green_04`, at pose `[0.8242562, 0.76016917]` for goal `[0.83, 0.76]`. The run declares at 106.12 seconds with 150 points, and all 50 subsequent hold samples remain at 150. The protocol, complete episode rows and independent artifact check are in `search_deployment_final/evaluation_protocol.json`, `test.json` and `artifact_verification.json`. The final comparison plot is `output/best_design/final_fleet_evaluation.png`.

### Exit-counter correction after the frozen benchmark

The G4 benchmark retains its `e2be32d` source hashes. Commit `a4f4f474ab986d44178704d86db24fbaadecb75c` subsequently corrects the scorer so each robot's fully-outside transition is counted independently. Previously, a second robot leaving while another remained outside received no additional penalty. The new native regression covers overlapping exits, re-entry, simultaneous exits and the resulting −10 points per occurrence; all 14 focused checks pass.

For valid starts, both counters are zero until the first exit and permanently positive afterward. Full-score success requires zero penalties, and motion does not consume the counter. Qualification and early-declaration decisions therefore remain unchanged. The frozen 300-mission test has no exit penalties, so its scores are unchanged under the corrected counter. Failed runs with exits may receive lower scores and objectives. The corrected-source nominal run still scores 160 at 80.80 seconds with all 50 hold checks passed; every saved trajectory array matches the frozen benchmark. CI run 34615070409 passes 52 unit tests, two native fleet checks and the legacy checks. Details are in `output/best_design/scoring_review.json` and `corrected_scoring_nominal/`.

## Mechanical evidence and simulation limits

The isolated LAB mechanism seated and released three samples for 30 points in 59.8 simulated seconds. The KIT gravity magazine released all four kits for 40 points in 39.44 seconds. Both recordings include a further five-second stable view. Their scenes retain the original arena objects, with only the tested robot present. Shared-fleet timing is measured separately.

`scripts/capture_best_fleet.py` records saved native states at real-time speed and writes source hashes alongside the MP4. It does not interpolate robot or object poses. The clips in `output/best_design` show simulation playback.

The dynamic model uses simplified collision shapes, a 450 g base mass budget and a 474 g LAB budget including its support and camera. Some mast and servo details in Blender are absent from the dynamic model. Tape is rigid, motor/friction distributions are engineering estimates, and the full mission does not include communication or camera-processing delays. Hardware measurement, camera calibration, loaded mechanism tests and repeated physical full-field runs are required to establish real competition performance. The finite searches and early stopping establish measured comparisons; they cannot establish that every possible design or optimization has been exhausted.
