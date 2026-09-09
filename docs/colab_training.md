# Colab neural policy training

The [competition training notebook](https://colab.research.google.com/drive/17PG8rfQmQ_49RQ887bSj4YVkAIQjOhZq) ran two CUDA training jobs on September 10, 2026, Korea time. The earlier physics-validation notebook remains separate.

## Observed runtime

The first allocation fell back to an L4 because Colab reported that the selected GPU was unavailable. A second A100 request succeeded. The training runs below both report `NVIDIA A100-SXM4-80GB`.

| Item | Observed value |
| --- | --- |
| GPU | NVIDIA A100-SXM4-80GB |
| GPU memory | 81,920 MiB total; 81,153 MiB free at the initial check |
| NVIDIA driver | 580.82.07 |
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| Compute capability | 8.0 |
| NumPy | 2.1.3 |
| Gymnasium | 1.3.0 |
| CUDA verification | Identity matrix multiplication passed on `cuda:0` |
| Runtime check time | 2026-09-09 15:48:31 UTC |

The runtime check wrote `/content/robotics_training/runtime_report.json`. The previous validation runtime had disconnected after inactivity or its maximum duration. No remaining-lifetime estimate was recorded for this new session.

## Training and measured comparison

Each job checked out an exact public Git commit and ran:

```sh
python scripts/train_competition_policy.py --print-artifact
```

The policy learns four goal-error feedback outputs through supervised teacher distillation. The geometric task planner operates outside the network. Both jobs used 131,072 generated training examples, 16,384 validation examples, seed `20260910`, validation generator seed `20260911`, batch size 8,192, and fused AdamW. Data and minibatches stayed on the GPU. Precision was FP32 with TF32 matrix multiplication allowed.

The revised trainer mixes the training examples with one seeded GPU permutation and masks connections between independent control axes. It also increases the step cap and changes the learning-rate schedule. Both jobs reached their step caps before satisfying the early-stopping criteria.

| Measurement | Baseline | Revised policy |
| --- | ---: | ---: |
| Source commit | `fab65e0337338bfbadee588736cea852f4aab956` | `81964fee8aa4838cc89f19f02916cffada4fdc02` |
| Network dimensions | 4–48–48–4 | 4–48–48–4 |
| Stored parameters | 2,688 | 2,688 |
| Active parameters | 2,688 | 672 |
| Training steps | 4,500 | 10,000 |
| Preparation time | 5.207 s | 1.352 s |
| Training time | 5.833 s | 12.840 s |
| Validation mean absolute error | 0.01848765 | 0.00164116 |
| Validation 99th-percentile absolute error | 0.07858355 | 0.00920955 |
| Validation maximum absolute error | 0.09804130 | 0.00961894 |
| Maximum output at zero goal error | 0 | 0 |
| Compressed weights | 10,698 bytes | 4,044 bytes |

The revised run reduced mean validation error by 91.12% on the same validation set. These measurements evaluate the learned feedback function. Competition task completion is evaluated separately using the exported weights in native MuJoCo.

## Saved artifacts and transfer verification

The revised weights and complete training history are saved in `output/competition/policy/weights.npz` and `output/competition/policy/training.json`. The baseline files remain in `output/competition/policy/baseline/`.

The notebook printed a ZIP containing only `weights.npz` and `training.json` as base64. Its reported byte count and SHA-256 matched the locally decoded bytes. The NPZ checksum also matched the training report before the files were written into the repository. A small NPZ fixture had previously verified the same route, including its decoded array values.

| Artifact | SHA-256 |
| --- | --- |
| Baseline weights | `8964866a425fcd586a896a9b265f1c77955302dd82bca3b4eb6be1964bf2e296` |
| Revised weights | `af0075c0c61861826391cb019c963a996f3af79dfbc542e41c365ddc45b5fc4f` |
| Revised transfer ZIP, 8,604 bytes | `ba7658a39ae360df69e3c171034008e655a6498b9af373927beac278eb6d629b` |

The public checkout supplied the trainer, which generated the dataset on the GPU. No runtime GitHub credentials, Drive mounting, or local-file upload were used.
