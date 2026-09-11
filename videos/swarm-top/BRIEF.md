---
workflow: general-video
flow: automation
storyboard: no
message: "Show the actual trained 40-robot swarm completing cooperative transport."
destination: github-pr
aspect: 1920x1080
language: en
audience: pull-request reviewers
length: full trajectory at 2x playback plus a two-second final hold
---

## Intent

Replay the saved native MuJoCo trajectory from a fixed overhead camera. A narrow evidence rail reports the verified transport result without applying the official competition score.

## Assets

- ../../output/swarm/proof/scene.xml — exact recorded model.
- ../../output/swarm/proof/trajectory.npz — saved simulation times, qpos, and phases.
- ../../output/swarm/proof/report.json — completion, object count, hold, simulator, and timing.
- ../../output/swarm/proof/metadata.json — robot and payload definitions.
- ../../output/swarm/proof/policy_trace.npz — recorded actions and observations.
- ../../output/swarm/policy/training.json and weights.npz — accepted training record and actor checksum.
- assets/fonts/ — bundled OFL-licensed Barlow and IBM Plex Mono fonts.

## Customizations

- Native MuJoCo rendering through the saved `overview` camera.
- Fixed overhead framing that keeps the whole field visible.
- Exact saved simulation clock and phase in the strip below the scene.
- Forty active robots, the eight-carrier/32-formation role split, delivered payloads, completion time, and five-second hold from the saved traces and report.
- 1080p master plus a 720p MP4 below 9.5 MB for direct PR attachment.

## Notes

- Finished video rendering is authorized after source validation.
- Test excerpts are labeled Pipeline test; a teacher run also says Temporary physical teacher.
- Every rendered pose is selected directly from saved qpos. No pose interpolation, physics stepping, camera motion, cuts, narration, or music.
- This is a cooperative transport benchmark, with no official competition score claim.
