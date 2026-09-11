# Final LAB drive training

This run uses the 474 g LAB geometry, including the passive support and moving camera. G4 trained both time costs from the same selected DAgger checkpoint. Validation chose the 0.05/s candidate for its 40/40 reach rate; the requested 0.5/s candidate reached 39/40 and remains available as `model/ppo_requested.npz`.

The 637.1-second run collected 720 initial episodes and two 180-episode DAgger rounds, then executed 960 native PPO episodes across the two time costs.

`model/weights.npz` is the selected export. Its independent test reaches 119/120 targets, with 1.980 mm position p95 and 2.021 seconds mean successful duration. The benchmark uses unloaded line and turn primitives. It does not establish full-fleet reliability.

Earlier experimental hybrid missions reached 160 at 116.30 seconds under `a7e420d` and 117.90 seconds under `f506bc2`. The frozen `e2be32d` hybrid run fails at 140 points at the 120-second limit after LAB's third-sample line times out and GREEN remains unfinished. Its 6,098 learned calls and 16,356 geometric calls leave all 50 hold checks at 140. The 30, 50 and 80 mm learned handoff settings scored 140, 140 and 120, so the larger handoffs were rejected and archived. The selected geometric mission scores 160 at 80.80 seconds nominally and succeeds in 299 of its 300 frozen randomized tests.

| Files | Purpose |
| --- | --- |
| `model/training.json`, `model/history.jsonl` | Executed epochs, DAgger and PPO rollouts, validation selection, source hashes. |
| `model/weights.npz`, `model/config.json` | Selected NumPy inference export and observation/action contract. |
| `model/ppo_baseline.npz`, `model/ppo_requested.npz` | Best validation checkpoints for 0.05/s and 0.5/s time cost. |
| `model/test.json` | Independent test, matched teacher/zero baselines and measured inference latency. |
| `dataset/manifest.json`, `dataset/*.npz` | All frozen training, validation and test arrays with hashes and episode seeds. |
| `launch.json`, `runtime.json`, `train.log` | G4 launch command, device record and training log. |

Reproduce the selected drive test from the repository root:

```bash
python scripts/test_best_drive.py --model output/best_drive_final/model \
  --dataset output/best_drive_final/dataset --workers 12 \
  --min-reach-rate .95 --max-final-distance-m .002
```

Run the experimental full-fleet hybrid with learned coarse motion and geometric fine docking:

```bash
python scripts/run_best_fleet.py --drive-policy output/best_drive_final/model \
  --drive-limits .48 4 --lab-drive-limits .35 2.5 --green-upper-first \
  --output output/best_design/learned_reproduction
```

The full mission still observes simulator poses. The CNN and camera alignment modules have separate RGB benchmarks. See [training and timing](../../docs/best_design_training.md) for both model generations, the current hybrid failure and the geometric mission results.
