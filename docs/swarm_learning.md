# Learning cooperative transport

The swarm controller uses one shared attention policy for every robot. It emits
normalized left wheel velocity, right wheel velocity, and lift position targets.
A geometric allocator assigns pairs and supplies grasp and transport goals. The
learned actor controls the motors at each physics control step. The allocator's
object selection and phase logic remain explicit code.

Training starts with noisy demonstrations collected by stepping the same physics
environment. Behavior cloning initializes the actor and briefly adapts it after
each curriculum promotion. MAPPO then updates the actor
and a centralized per-robot critic from its own sampled actions, measured rewards,
and MuJoCo transitions. `--bc-anchor-coef` weights a demonstration
retention loss during PPO to preserve contact precision. It samples the current
stage's physical demonstrations and logs its loss separately. The default is 0.
The exported policy contains the trained attention network and no teacher fallback.

## Research choices

| Source | Decision in this project |
| --- | --- |
| [MAPPO, Yu et al.](https://arxiv.org/abs/2103.01955) | Centralized critic, shared actor, clipped policy and value updates, few PPO epochs, masked advantage normalization, and KL stopping. |
| [Graph policy gradients, Khan et al.](https://arxiv.org/abs/1907.03822) | Reuse local aggregation weights as the number of robots grows. This actor uses masked neighbor attention instead of graph convolution. |
| [MDP homomorphic networks, van der Pol et al.](https://arxiv.org/abs/2006.16908) | Encode known state/action symmetry in the actor. Our implementation averages the actor over the identity and a left/right reflection. |
| [MAPush, Feng et al.](https://arxiv.org/abs/2411.07104) | Separate object-level subgoals from learned motor coordination. Our wheeled robots and contact model differ from the paper's quadrupeds. |
| [Cooperative transport allocation, Shibata et al.](https://arxiv.org/abs/2212.02692) | Keep task priorities and local coordination separate so several teams can transport different objects. Our task allocator is programmed. |
| [Residual reinforcement learning, Johannink et al.](https://arxiv.org/abs/1812.03201) | Use existing control knowledge to reduce contact exploration. Here demonstrations initialize a direct actor; runtime commands have no residual teacher contribution. |
| [DAgger, Ross et al.](https://proceedings.mlr.press/v15/ross11a.html) | Collect corrective labels on states visited by the learned controller, aggregate those examples, and refit before PPO. |
| [Difference rewards policy gradients, Castellini et al.](https://arxiv.org/abs/2012.11258) | Credit assignment matters. Our team reward and local shaping do not compute the counterfactual removals needed to claim difference rewards. |

This is a project-specific combination of established methods. Its quality is
determined by held-out physical transport results.

## Observation and action contract

`SwarmVectorEnv` returns PyTorch tensors on the requested device. Dimensions are
inferred from the first reset and saved with the checkpoint. The actor supports
any number of robots and neighbors with the same feature definitions.

| Key | Shape | Purpose |
| --- | --- | --- |
| `local` | E × N × F | Robot-local state, assigned object, goal, manipulation phase, and contact feedback. |
| `neighbors` | E × N × K × G | Relative teammate state in each robot's frame. |
| `neighbor_mask` | E × N × K | Valid neighbors; padding contributes exactly zero. |
| `active` | E × N | Robots participating in the current curriculum stage. |
| `phase`, optional | E × N | Integer task phase used to balance demonstration losses. |
| `learning_weight`, optional | E × N | Carrier emphasis during demonstration fitting. |
| `global` | E × S | Privileged scene information used only by the critic. |
| Action | E × N × 3 | Left wheel, right wheel, lift target, each in [-1, 1]. |

The actor embeds local and neighbor observations into 64 channels, applies
two-head dot-product attention, then fuses the local and pooled neighbor state.
It samples a diagonal Gaussian followed by tanh. PPO uses the exact transformed
log probability. Deterministic evaluation uses the tanh of the Gaussian mean.
Inactive robots output zero and do not contribute to losses. A separate critic
combines privileged global features with each robot's local features.

For the environment's 32-feature contract, new training runs enable reflection
averaging. If `f` is the shared actor, `M` reflects its input, and `P` exchanges
wheel outputs, the Gaussian mean is `0.5 * (f(obs) + P(f(M(obs))))`. Tanh follows
the average. This makes deterministic commands exchange left and right wheels
under reflection while preserving lift. It encodes the module's bilateral wheel
geometry and discourages a learned turning bias in a straight pinch.

Reflection negates local indices 0, 2, 4, 6, 8, 10, 25, and 28; it exchanges the
previous wheel commands at 22 and 23. Neighbor indices 0, 2, and 4 change sign.
Neighborhood masks, lift, task phases, and the centralized critic are unchanged.
The same two actor evaluations run in NumPy. `--disable-reflection` trains the
unconstrained ablation. Existing checkpoints without the `mirror` configuration
field retain their original behavior when loaded or resumed. Physical success
must still be measured for each run.

`step(action)` returns `(obs, reward, terminated, truncated, info)`. Reward is E × N;
termination flags are E. Returned observations must describe the terminal state
before reset. Time limits bootstrap from that state, while true terminations
bootstrap zero. GAE never crosses episode boundaries. `reset_done(mask)` resets
only finished worlds and returns all current observations.

`info` supplies `success`, `delivered`, and `reward_components`. Each reward
component is E × N or E. These are recorded individually in `history.jsonl`.
`teacher_action()` is used only to collect warmup trajectories. `set_curriculum`
accepts `num_active_robots`, `num_active_objects`, and `difficulty`.

## Rewards and curriculum

The environment computes rewards from object movement and contact state.
Transport progress should depend on the object being supported above its starting
height. Approach shaping uses improvement in grasp distance, with its weight
reduced after contact. Pickup and settled delivery bonuses are paid once per
object per episode. A supported transport arms a drop latch. Unplanned arena
contact or clearance falling to 1.5 mm triggers one penalty; supported regrasp
can arm it again. Planned lowering disarms it. Brief airborne pad-contact losses
use the existing contact-asymmetry cost and recovery logic.
Dropping a lifted object, unstable contact, actuator effort,
and abrupt commands incur costs. Exact implemented coefficients belong beside
the environment's reward calculation and appear in its configuration.

Avoid a positive per-step proximity bonus. A stationary team could accumulate it
without carrying anything. Delivery success must require a physical pickup,
transport, release, and settling at the goal. Pure pushing must not satisfy the
pickup-and-carry criterion. Every reported component should be traceable to the
simulator state; the actor's output is never evidence of success by itself.

The stages activate 2, 4, 8, then 40 robots and 1, 1, 2, then 4 objects. Difficulty
increases from 0 to 1. The same observation widths and model weights are retained.
Demonstrations cover 1800 control steps per stage, long enough for a 30-second
transport at 50 Hz. Only active robot examples are stored. Optional task phase
labels weight rare pickup and release phases inversely to their frequency, and
carrier weights prevent queued robots from dominating the cloning loss. These
weights do not alter the PPO objective. Promotion requires at least 20 PPO
updates at the current stage and at least 70%
success in a complete evaluation batch. Stages 0–2 use eight evaluation episodes;
stage 3 uses 32 by default. A stalled stage remains active and is
reported as incomplete training.

## Running and inspecting a job

Install `requirements-swarm.txt` in an A100 or H100 runtime. The current training
job uses native MuJoCo for contact simulation and CUDA for the neural networks.
MuJoCo Warp also provides parallel simulation on NVIDIA hardware and
interoperates with PyTorch. Its
[official documentation](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html)
describes graph capture, contact capacity, and solver settings that affect
throughput.

The full-stage A100 configuration is:

```bash
python scripts/train_swarm_policy.py \
  --backend native --num-envs 8 --native-workers 8 \
  --robots 40 --objects 4 --episode-seconds 40 --start-stage 3 \
  --demo-steps 1800 --demo-noise .01 --bc-updates 4000 \
  --bc-anchor-coef 30 --learning-rate .0001 --ppo-epochs 2 \
  --clip .1 --target-kl .01 --entropy-coef .0003 \
  --horizon 128 --updates 200 --eval-interval 40 \
  --eval-episodes 32 --output output/swarm/policy
```

This run starts directly with four transport pairs and 32 formation robots.
The curriculum described above is available by omitting `--start-stage 3`.
The coefficient 30 multiplies normalized motor-command MSE; the report records
both that coefficient and the raw retention loss so its effect can be inspected.

The command rejects GPUs outside A100/H100 unless `--allow-other-gpu` is explicit.
`--backend native --native-workers 8` advances independent worlds on persistent
CPU workers. Each uses the same wheel torque-speed limit and lift slew calculation
at every physics substep. The actor, critic, demonstration fitting, and PPO
optimization run on the GPU. The report records both devices and the effective
number of native workers. `--backend warp` runs physics and policy optimization
on CUDA.

An eight-world, eight-worker benchmark on the Colab A100 host completed 100 control
steps in 10.59 seconds with zero terminations, robot collisions, or overflows. A
local four-worker check ran 3.09 times faster than serial integration and produced
bit-identical positions, velocities, and lift targets.

The tested Warp configuration had 36 early episode terminations in a 100-step,
32-world benchmark. Separate launches also encountered compiled collision-kernel
metadata errors. Native simulation passed the corresponding contact stability
checks, which determined the backend used for this job.

`--cpu-smoke` selects the native backend and marks the result as a smoke check.
`--time-budget-seconds` stops after an update boundary. `--resume checkpoint.pt`
restores the model, optimizer, stage, and update counter. Set `--updates` to the
desired total count when resuming. Physics states are reset on resume.
An anchored resume collects fresh demonstrations for the restored stage.
`--skip-initial-eval` skips the baseline before the first PPO update, which can
save a repeated audit on resume. Scheduled validation and all final audits still run.
Two consecutive stage-3 validation passes with at least 32 episodes, 80% success,
and a 60% Wilson lower bound stop optimization early. The unseen final audit still
runs. `--eval-max-steps 0` automatically allows enough control steps to finish
every requested episode; a positive value imposes an explicit cap.

`python scripts/verify_swarm_policy.py` checks the reflection against named
physical features, exact wheel exchange for 2, 8, and 40 robots, finite gradients,
NumPy parity, and compatibility with older exports.

| Artifact | Contents |
| --- | --- |
| `progress.json` | Atomically replaced latest update, reward components, optimization metrics, and evaluation. |
| `history.jsonl` | One record per PPO update. |
| `checkpoint.pt` | Torch actor, critic, optimizer, architecture, and counters. |
| `warmstart.npz` | Actor before PPO, for an ablation comparison. |
| `weights.npz` | Latest trained actor for NumPy inference. |
| `best.npz` | Best validation success within the furthest evaluated stage. |
| `training.json` | Device, configuration, demonstration counts, PPO counts, held-out evaluation, and weights SHA-256. |

Validation uses seed offset 100000. Final evaluation uses offset 200000 and the
full deployment stage, plus a zero-action baseline. Evaluation assigns a fixed,
balanced episode quota to each world and waits for every quota, so fast successes
cannot replace slower failed episodes. The acceptance flag requires
40 robots, at least two active objects, at least 32 held-out episodes, at least
80% success, a 95% Wilson lower bound of 60%, and a success advantage above the
zero baseline of more than 20 percentage points. Policy optimization must use CUDA.

The final evaluation accompanies native MuJoCo playback, contact and pickup
evidence, and warmstart-versus-PPO comparisons.

## Correcting accumulated control error

The initial A100 cloning run completed no full trials out of 32 and averaged 1.5
deliveries out of four. In a separate native rollout with seed 20260911, all four
objects were lifted by five seconds, but repeated losses of contact left one
delivery at the 40-second limit. Formation robots reached their targets and later
drifted away. Wheel-command RMSE on the carrier's own transport states was
0.064/0.068, compared with 0.0095/0.0118 on a held-out teacher trajectory. These
measurements motivated collecting examples under the learner's own control.

The initial reward also charged every loss of two-pad support as a drop. In that
native trace, the first cylinder incurred 13 such events while none of those
frames touched the arena. The corrected reward tracks unplanned grounding with
the latch described above. Contact dynamics and task completion criteria stay
the same.

`--dagger-rounds` enables physical dataset aggregation before PPO. Each control
step labels the current observation with the demonstration controller's action.
A per-world draw chooses whether the teacher or learned actor advances physics.
`--dagger-teacher-prob` decreases linearly to zero in the last round; a single
round uses the learner throughout. Each round fits the aggregated dataset,
capped deterministically at two million active-agent examples. `dagger.json`
records the mixture, completed episodes, deliveries, and action errors by role
and carrier phase. Its collection outcomes are separate from held-out evaluation.

`--balance-demo-roles` balances phases within the carrier group and gives carriers
and formation robots equal total fitting weight. The original global phase
weighting assigned little weight to formation robots because they all use phase
zero. `--initial-action-std .01` sets exploration for fresh models; resumed models
retain their saved standard deviation. CUDA matrix multiplication uses full FP32.

For a fresh recovery run, add the following options to the full-stage command
above, replacing its BC-update count and anchor coefficient:

```bash
--bc-updates 6000 --bc-anchor-coef 100 --balance-demo-roles \
--initial-action-std .01 --dagger-rounds 4 --dagger-steps 2000 \
--dagger-updates 4000 --dagger-teacher-prob .6 --skip-initial-eval
```

`bc_initial.npz` preserves the initial cloned actor. `dagger.npz` and
`warmstart.npz` hold the actor after dataset aggregation, immediately before PPO.
The final warmstart comparison therefore measures the effect of PPO after the
same imitation training. Resuming with DAgger archives an existing warmstart
before replacing it and records every export's meaning in `training.json`.

`python scripts/test_swarm_imitation.py` checks learner-state labeling, the policy
mixture, role weights, dataset capping, bounded fitting diagnostics, and saved
exploration on resume.
`python scripts/test_swarm_rewards.py` checks delayed grounding, planned lowering,
regrasp, bounce suppression, and reset behavior.

## Verifying and packaging the result

`scripts/verify_swarm_proof.py` audits a saved native rollout. It validates source
and artifact hashes, free-body topology, motor limits, seed-derived initial poses
and goals, complete timestamps, physical pickup and supported transport, released
payload rest, and formation arrival. It reconstructs every contact frame and
recomputes every neural action in bounded batches. Pose-derived actor inputs are
checked against geometry in the quick audit.

`--replay-physics` rebuilds the episode and integrates every recorded motor command.
It compares every resulting pose, timestamp, contact event, and actor observation,
including velocity inputs. Successful replay must reproduce task completion and
the full five-second hold. It requires the recorded MuJoCo version and matching
source files. The video renderer runs this full replay before rendering either
camera view.

The default verifier requires accepted A100/H100 training and matching weights.
`--allow-teacher` permits a teacher pipeline fixture and labels it explicitly;
it cannot produce `final_neural_proof=true`. Run the tamper checks with
`python scripts/test_swarm_proof.py --fixture PATH_TO_TEACHER_PROOF --replay-physics`.

After accepted training, record and render the neural proof, then build its
download bundle:

```bash
python scripts/run_swarm.py --output output/swarm/proof
python scripts/render_swarm_video.py --render
python scripts/package_swarm.py
```

The bundle contains the native scene and trace, policy weights and checkpoint,
training logs, module CAD, both complete video projects, and both video sizes.
The packager checks the artifact links, CRC, and source imports from a clean
extraction. It writes `output/swarm/swarm_complete.zip`, a manifest, and SHA-256
checksums. A final bundle requires a successful full native replay.
