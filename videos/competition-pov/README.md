# Robot POV video

This project presents the same verified competition run through a camera attached to the robot body. Its mount is at `[0, 0.095, 0.150]` meters in robot-body coordinates, with a fixed 65-degree downward pitch and a 90-degree vertical field of view. The view follows the recorded body orientation, including pitch and roll.

The panel identifies Robot POV and Body-mounted camera and uses the same verified report and training measurements as the overview. The full run plays at 3× speed with a two-second final hold.

From the repository root:

```sh
.venv/bin/python scripts/render_competition_video.py --camera robot-pov \
  --project videos/competition-pov --render
```

Add `--reuse-source` to reuse a native video whose input hashes and camera settings match. The script checks the composition before rendering, uses the pinned HyperFrames 0.8.33 with four workers, and creates `renders/competition-pov.mp4` plus a `renders/competition-pov-pr.mp4` attachment below 9.5 MB.

Native footage and MP4 exports stay ignored in Git. Encoded QA frames and export metadata are saved under `output/competition/pov/`. The overview project and its artifacts remain separate.
