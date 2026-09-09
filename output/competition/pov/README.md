# Robot POV proof video

The body-mounted camera replays the same verified 160/160 competition recording as the overview. Its position is `[0, 0.095, 0.150]` meters in the robot body, with a fixed 65-degree downward pitch and a 90-degree vertical field of view. The view preserves the recorded pitch, roll, and yaw.

The scene, trajectory, and report hashes match the overview. The full replay is 249.933333 seconds at 30 fps, using 3× playback and a two-second final hold. The panel shows Robot POV, Body-mounted camera, the verified score, the 743.8-second recording, the official 120-second limit, and 12.8 seconds of A100 training.

- [Sample pickup](attachment-sample-pickup.png), [sample placement](attachment-sample-release.png), and [final view](attachment-final.png) are frames from the encoded PR attachment.
- [Native contact sheet](native-contact-sheet.png) shows the beginning, middle, and end of the recorded view.
- [Render manifest](render-manifest.json) records source hashes and the exact camera mount. [Video metadata](video-metadata.json) records export hashes, dimensions, sizes, and inspected timestamps.

The 1080p master is `videos/competition-pov/renders/competition-pov.mp4`, 134,095,499 bytes. The 720p PR attachment is `videos/competition-pov/renders/competition-pov-pr.mp4`, 8,827,884 bytes. Both use H.264. MP4 files remain ignored in Git and are attached directly to pull request #2.

HyperFrames checks pass with no findings and all 72 contrast checks passing. Pickup, placement, and the final state were visually checked in both encoded exports. The overview project and its artifacts were preserved.

Rebuild from the repository root:

```sh
.venv/bin/python scripts/render_competition_video.py --camera robot-pov \
  --project videos/competition-pov --render
```
