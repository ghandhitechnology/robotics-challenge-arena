---
workflow: general-video
flow: automation
storyboard: no
message: "Show the actual trained 40-module magnetic body transporting two payloads."
destination: github-pr
aspect: 1920x1080
language: en
audience: pull-request reviewers
length: full trajectory at 2x playback plus a two-second final hold
---

## Intent

Replay the saved native MuJoCo magnetic-body trajectory from a fixed overhead camera. A compact evidence rail and the frame HUD report only verifier-backed body connectivity and transport measurements.

## Assets

- ../../output/swarm/body_proof/scene.xml — exact recorded magnetic-body model.
- ../../output/swarm/body_proof/trajectory.npz — saved poses, payload phases, magnet states, and force-bearing graph.
- ../../output/swarm/body_proof/report.json and metadata.json — task, timing, module, payload, and magnetic-port definitions.
- ../../output/swarm/body_proof/policy_trace.npz — every four-action policy call and observation.
- ../../output/swarm/body_policy/training.json and weights.npz — accepted training record and actor checksum.
- assets/fonts/ — bundled OFL-licensed Barlow and IBM Plex Mono fonts.

## Customizations

- Fixed overhead body-focused framing that keeps all 40 modules and both payloads visible.
- Every displayed pose is selected directly from saved qpos with no interpolation.
- The HUD derives delivered payloads, enabled magnets, ≥2 mN link count, and largest load-bearing component from the same saved frame.
- Four carrier modules and 36 other body modules are derived from the two-payload assignment contract.
- 1080p master plus a 720p MP4 below 9.5 MB for direct PR attachment.

## Notes

- Full output requires the strict verifier's accepted neural magnetic-body proof and full native replay.
- Teacher input is restricted to a clearly labeled, time-limited recording diagnostic.
- The renderer adds no attraction animation or invented connection geometry.
- This is a magnetic-body transport benchmark and makes no official competition-score claim.
