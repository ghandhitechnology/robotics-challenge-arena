# Competition proof video

The full recorded run scores 160/160, completes all 16 tasks, and passes the five-second final view. Completion was declared at 738.76 simulation seconds; the recording continues to 743.76 seconds. The video labels this a feasibility run and shows the official 120-second limit.

The 4:09.93 replay uses exact saved MuJoCo poses at 3× playback plus a two-second final hold. Its panel shows the verified 12.8-second A100 training run. Mission, training, and weight-file SHA-256 values match.

- [Final attachment frame](attachment-final.png) and [sample placement](attachment-sample-release.png) show the encoded 720p result.
- [Native contact sheet](native-contact-sheet.png) shows the beginning, middle, and end of the recorded trajectory.
- [Render manifest](render-manifest.json) records source hashes and replay timing; [video metadata](video-metadata.json) records export hashes, sizes, dimensions, and inspected timestamps.

The 1080p master is `videos/competition-proof/renders/competition-proof.mp4` (26,724,870 bytes). The inline PR attachment is `videos/competition-proof/renders/competition-proof-pr.mp4` (8,866,080 bytes). Both are H.264, 30 fps, and 249.933333 seconds. MP4 files remain ignored in Git and are attached directly to the pull request.

HyperFrames checks pass. Encoded sample placement and final states were visually checked in both exports. The dedicated training GPU runtime was released after preserving the notebook outputs and local policy artifacts.

Rebuild from the repository root with `.venv/bin/python scripts/render_competition_video.py --render`.
