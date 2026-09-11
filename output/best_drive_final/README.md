# Final LAB drive training

This run uses the 474 g LAB geometry, including the passive support and moving camera. G4 trained both time costs from the same selected DAgger checkpoint. Validation chose the 0.05/s candidate for its 40/40 reach rate; the requested 0.5/s candidate reached 39/40 and remains available as `model/ppo_requested.npz`.

`model/weights.npz` is the selected export. Its independent test reaches 119/120 targets, with 1.980 mm position p95 and 2.021 seconds mean successful duration. The benchmark uses unloaded line and turn primitives. The full fleet hybrid trial scores 160 at 116.30 seconds; the selected geometric mission finishes at 81.10 seconds.

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

Run the full fleet with learned coarse motion and geometric fine docking:

```bash
python scripts/run_best_fleet.py --drive-policy output/best_drive_final/model \
  --drive-limits .48 4 --lab-drive-limits .35 2.5 --green-upper-first \
  --output output/best_design/learned_reproduction
```

The full mission still observes simulator poses. The CNN and camera alignment modules have separate RGB benchmarks. See [training and timing](../../docs/best_design_training.md) for both model generations and the mission comparisons.
