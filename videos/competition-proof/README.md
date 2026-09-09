# Competition proof video

The source is the recorded MuJoCo model, trajectory, and score report. The renderer replays saved qpos without stepping physics or interpolating poses, adds the recorded clock and phase below the scene, and assembles one continuous HyperFrames view.

From the repository root, after the final proof files exist:

```sh
.venv/bin/python scripts/render_competition_video.py --render
```

This renders `output/competition/proof/` at 3× playback, runs the HyperFrames checks, and writes:

- `videos/competition-proof/renders/competition-proof.mp4` — 1920 × 1080 master.
- `videos/competition-proof/renders/competition-proof-pr.mp4` — 1280 × 720 attachment below 9.5 MB.
- `videos/competition-proof/render-manifest.json` — input/output hashes and exact replay timing.

The script requires the repository's MuJoCo Python environment with Pillow, FFmpeg, Node.js, and npm. HyperFrames is pinned in `package.json`; GSAP and the OFL fonts are bundled locally. Use `--reuse-source --render` when only the surrounding composition changed. The cache is checked against the input hashes and replay options.

For a short pipeline test, copy this scaffold under `tmp/`, then pass `--project tmp/competition-video-test --input tmp/mission_first --max-seconds 3`. Test clips remain outside the final proof directory.

The final report supplies the score, task counts, physical tape mode, simulation time, and official time limit. The video identifies simulator state feedback. A feasibility result beyond the official limit remains labeled as a feasibility run.

The installed GitHub CLI supports direct video attachment:

```sh
gh pr create --head feat/competition-robot --title "..." --body-file /tmp/pr-body.md \
  --attach videos/competition-proof/renders/competition-proof-pr.mp4
```

`--attach` renders video as an inline player. The 9.5 MB encode fits the documented 10 MB Free-plan video limit. See [GitHub's CLI attachment announcement](https://github.blog/changelog/2026-09-01-github-cli-media-in-issues-pull-requests-and-comments/).

