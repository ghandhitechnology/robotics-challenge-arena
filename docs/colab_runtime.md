# Colab runtime

A dedicated [Robotics arena MuJoCo runtime checks notebook](https://colab.research.google.com/drive/1zmPS7IvWuNpNUs-WjvSIcPDIHr1Qgtqm) was created on September 9, 2026. It completed all 21 native MuJoCo tests, a short arena simulation, and an arena render on the A100 runtime. The local `colab_runtime_diagnostics.ipynb` preserves the executed code and observed text reports.

| Item | Observed value |
| --- | --- |
| GPU | NVIDIA A100-SXM4-40GB |
| GPU memory | 40,960 MiB |
| NVIDIA driver | 580.82.07 |
| Operating system | Linux |
| Python | 3.13.15 |
| JAX / JAXlib | 0.11.1 / 0.11.1 |
| NumPy | 2.1.3 |
| MuJoCo / MuJoCo MJX | 3.12.0 / 3.12.0, installed after the initial diagnostic |
| Gymnasium | 1.3.0 |
| Warp | 1.16.0 |

These are observations from that session. Run the diagnostic cell again after reconnecting. Native MuJoCo runs physics on the CPU. NVIDIA EGL supplies GPU rendering separately.

## Running project tests

The notebook's runtime type is set to A100 GPU. It checks out the public `feat/mujoco-physics` branch from `ghandhitechnology/robotics-challenge-arena` in an isolated `/content` directory. The final validation cell fetches and checks out the exact source commit before running. This uses ordinary notebook code cells and requires no local asset upload. Each run records the full Git commit before testing.

Browser execution has been verified through the notebook's code editor, **Run cell**, and the cell's output panel. The initial diagnostic output remains intact. A local file chooser upload was unavailable; the public repository checkout supplies the project files.

This session does not expose a connected Colab MCP tool or an authenticated Colab CLI. Code-cell execution is the tested route.

The official [Google Colab CLI](https://github.com/googlecolab/google-colab-cli) also supports A100 sessions, local script execution, uploads, and downloads. It requires its own supported authentication setup; browser sign-in alone does not establish CLI access. The CLI was researched but not installed or authenticated here.

## Reproducing the environment check

Open `colab_runtime_diagnostics.ipynb` from this directory in Colab and select **Runtime → Change runtime type → A100 GPU**. The first cell reads environment information. Later cells install pinned dependencies, test native adhesion, inspect Warp compatibility, configure NVIDIA rendering, and run the public branch's validation commands.

The initial browser connection needed one supported Chrome reconnection. Chrome, its extension, and the native messaging host were already present. After opening the selected Chrome profile through the plugin's documented recovery flow, Colab became reachable and was already signed in.

## Rendering setup

The first EGL render selected Mesa `llvmpipe`. Colab already contained the matching NVIDIA EGL libraries under `/usr/lib64-nvidia`, but lacked NVIDIA's vendor registration. Following the [official MuJoCo 3.12 Colab tutorial](https://github.com/google-deepmind/mujoco/blob/3.12.0/python/tutorial.ipynb), the notebook added `10_nvidia.json`, selected that ICD, and added the library directory to `LD_LIBRARY_PATH`. No driver package was installed.

A fresh Python subprocess then rendered a nonblank 320 × 240 image with `GL_RENDERER` reporting `NVIDIA A100-SXM4-40GB/PCIe/SSE2`. Use a fresh subprocess for project renders because the notebook's earlier process had already initialized Mesa.

## Warp compatibility gate

MuJoCo 3.12 lists flex support for MJX-Warp and none for MJX-JAX in its [feature table](https://mujoco.readthedocs.io/en/3.12.0/mjx.html#feature-parity). The bundled Warp model nevertheless omits the `pair_adhesion` and `geom_adhesion` fields used by this arena. Its [versioned model definition](https://github.com/google-deepmind/mujoco/blob/3.12.0/mjx/mujoco/mjx/third_party/mujoco_warp/_src/types.py) and the installed package inspection agree. A nonzero-adhesion model was accepted by `put_model`, so conversion success alone does not establish physical equivalence.

A native CPU coupon applied a 0.05 N upward force to a 0.01 kg bead for 0.1 s. With zero adhesion, its center rose from 0.010 m to 0.035025 m. With 0.1 N contact adhesion, it stayed at 0.010000008 m. Both cases completed without warnings. The pinned Warp model lacks the field needed to preserve this difference. No Warp simulation steps were run after this failed compatibility check.

The runtime's Python bond damage, slip-dependent friction, and motor updates would also need an explicit device implementation and matching physical tests. The current full-physics execution path remains native MuJoCo CPU; GPU rendering is verified separately.

## Arena validation result

The following commands passed on final source commit `d8c20fd6d9a81cb34986356cd0b222109f94d0f5` from `feat/mujoco-physics`:

```sh
python -m unittest discover -s tests/mujoco -p 'test_*.py' -v
python -m arena_mujoco run --seconds .02 --output /content/native_report_final.json
python -m arena_mujoco render --seconds 0 --output /content/arena_final.png
```

The 0.02-second native run finished with 1,425 contacts, all 1,323 tape bonds intact, and zero numerical warnings. Maximum absolute generalized velocity was 0.000421602; maximum bond slip was 2.635 μm. Maximum tensile strain was 0.0000737403, and the material limit was not exceeded. Native stepping took 8.083 seconds of wall time. This short run checks initialization and immediate stability; long episodes and physical calibration require separate validation.

All 21 tests passed in 60.329 seconds. They covered calibration, flex stepping and checkpoints, the Gymnasium contract, seeded noisy/delayed action replay, task success and material limits, motor behavior, surface friction, and tape adhesion, peeling, rebonding, curvature, damage, and mesh refinement. The arena render command also returned successfully with the NVIDIA EGL environment. Its 1280 × 960 image had pixel standard deviation 123.675 and SHA-256 `93f4de348727444a3b362a1e6f40940eb5a8bd0cc02d7d42e51e2b2175c354b5`.

The live runtime retains `/content/native_report_final.json`, `/content/arena_final.png`, and `/content/robotics_runtime_checks/final_report.json`. The latter includes the exact commit, command output, native metrics, NVIDIA EGL probe, and image checks. Earlier notebook cells retain the initial six-test validation on `dc77f2aeae639e3120ece2857d74bcfb3bec24d5` and the adhesion compatibility probe. Cell outputs preserve the observed results after the runtime disconnects.
