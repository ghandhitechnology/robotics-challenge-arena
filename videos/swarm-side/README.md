# Swarm side proof video

This project replays the trained 40-robot native MuJoCo recording from a fixed low oblique camera. Every frame uses a saved qpos sample; the renderer does not step physics or interpolate poses.

From the repository root, after `output/swarm/proof/` and the accepted actor export exist:

```sh
.venv/bin/python scripts/render_swarm_video.py --render
```

The one command renders both requested views. This project writes:

- `renders/swarm-side.mp4`: 1920 × 1080 master.
- `renders/swarm-side-pr.mp4`: 1280 × 720 attachment below 9.5 MB.
- `render-manifest.json`: source, camera, frame-selection, policy, and output hashes.

Use `--reuse-source --render` after composition-only edits. Reuse fails if any proof hash, camera setting, replay option, frame index, or native MP4 checksum changed.

For a temporary physical-teacher pipeline check, use an actual completed teacher recording with `--allow-teacher-test --max-seconds 3 --native-only`. The resulting composition is explicitly labeled as a test excerpt and cannot be used for a full teacher render.

The renderer first runs `scripts/verify_swarm_proof.py`, which replays every motor command through native physics and recomputes every deployed neural action. The final path requires a successful neural report, 40 active robots, four payloads spanning all three grasp geometries, a valid five-second final hold, accepted training, and matching checksums across the report, training record, and actor weights. The evidence rail reports the actual role split: eight payload carriers and 32 formation modules. It identifies the four-payload cooperative transport benchmark.
