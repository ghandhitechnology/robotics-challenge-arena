---
workflow: general-video
flow: automation
storyboard: no
message: "Show the recorded competition run and its measured result."
destination: github-pr
aspect: 1920x1080
language: en
audience: pull-request reviewers
length: trajectory duration at 3x playback plus a two-second final hold
---

## Intent

Replay the actual MuJoCo trajectory as one continuous view. A narrow results panel summarizes the same run. This is technical evidence for the competition-robot pull request.

## Assets

- ../../output/competition/proof/scene.xml — the exact recorded model.
- ../../output/competition/proof/trajectory.npz — recorded simulation times, qpos, and task phases.
- ../../output/competition/proof/report.json — measured score, completion status, timing, and diagnostics.
- assets/fonts/ — bundled OFL-licensed Barlow and IBM Plex Mono fonts.

## Customizations

- Native MuJoCo rendering with the saved overview camera.
- Exact recorded simulation clock and phase, burned into a separate strip below the scene.
- Roughly three times real-time playback, normally one 10 Hz recorded pose per 30 fps video frame.
- Final score and official time limit taken from report.json.
- A compact MP4 for direct GitHub pull-request attachment.

## Notes

- The user authorized the full cycle without further questions, including the finished video. Rendering is already authorized after validation.
- Test excerpts are labeled Pipeline test. Full rendering waits for the final trajectory.
- No invented robot motion, interpolated poses, cuts, narration, music, or decorative animation.
- The source footage supplies all motion. Keep the camera fixed and all scene geometry visible.

