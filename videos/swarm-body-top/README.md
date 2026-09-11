# Magnetic swarm-body overhead proof video

This project replays the 40-module native MuJoCo magnetic-body recording from a fixed overhead camera. Every frame uses one saved qpos sample; the renderer never steps physics or interpolates poses.

From the repository root, after `output/swarm/body_proof/` and the accepted `output/swarm/body_policy/` export exist:

```sh
.venv/bin/python scripts/render_swarm_body_video.py --render
```

The command renders both requested views. This project writes:

- `renders/swarm-body-top.mp4`: 1920 × 1080 master.
- `renders/swarm-body-top-pr.mp4`: 1280 × 720 attachment below 9.5 MB.
- `render-manifest.json`: proof, verifier, camera, frame-selection, magnetic telemetry, and output hashes.

Use `--reuse-source --render` after composition-only edits. Reuse fails if the proof, verifier, camera, selected frames, or native MP4 changed.

For a short pipeline check, point `--input` at a verifier-valid teacher recording and use `--allow-teacher-diagnostic --max-seconds 3 --native-only`. The result is labeled as a temporary recording diagnostic and cannot take the final render path.

The renderer runs `scripts/verify_swarm_body.py` before video work. Final output requires an accepted neural body policy, all 40 modules, two delivered payloads, every recorded four-action inference, deliberate link release and re-formation, the five-second hold, and a complete native physics replay. The frame HUD reads actual saved ≥2 mN connectivity and payload phases; it draws no synthetic magnetic links.
