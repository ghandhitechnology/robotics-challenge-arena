---
workflow: general-video
flow: automation
storyboard: no
message: "Show the complete verified run through the robot's body-mounted camera."
destination: github-pr
aspect: 1920x1080
language: en
audience: pull-request reviewers
length: full trajectory at 3x playback plus a two-second final hold
---

## Intent

Add a Robot POV video to pull request #2. Use the same successful MuJoCo recording as the overview video and a camera rigidly attached to the robot body. Preserve body pitch, roll, and yaw in the view.

## Assets

- ../../output/competition/proof/scene.xml
- ../../output/competition/proof/trajectory.npz
- ../../output/competition/proof/report.json
- ../../output/competition/policy/training.json
- assets/fonts/ contains the same licensed fonts as the overview.

## Presentation

Use one continuous native view with the recorded simulation clock and phase beneath it. The panel says Robot POV and Body-mounted camera, and shows the verified score, task counts, recorded time, official time limit, and A100 training time. The report identifies this as a feasibility run.

## Execution

The user requested the additional finished POV video on the existing PR. Full rendering and attachment are authorized. Preserve the overview project and its artifacts. Use the recorded qpos without stepping physics, interpolating poses, or stabilizing the body-mounted camera. Keep audio silent and surrounding graphics static.
