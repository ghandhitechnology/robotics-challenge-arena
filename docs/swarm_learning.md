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
and MuJoCo transitions. Optional `--bc-anchor-coef 0.1` adds a small demonstration
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
object per episode. Dropping a lifted object, unstable contact, actuator effort,
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

Install `requirements-swarm.txt` in an A100 or H100 runtime. MuJoCo Warp runs
parallel simulation on NVIDIA hardware and interoperates with PyTorch. Its
[official documentation](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html)
describes graph capture, contact capacity, and solver settings that affect
throughput. The native backend is useful for short implementation checks.

```bash
python scripts/train_swarm_policy.py --backend warp --num-envs 64 \
  --updates 1000 --horizon 128 --demo-steps 1800 --bc-updates 500 \
  --eval-episodes 32 --output output/swarm/policy
```

The command rejects GPUs outside A100/H100 unless `--allow-other-gpu` is explicit.
`--backend native` keeps physics on the CPU while training the actor on CUDA.
The report records the physics backend and both devices. `--backend warp` runs
physics and policy optimization on CUDA.
`--cpu-smoke` selects the native backend and marks the result as a smoke check.
`--time-budget-seconds` stops after an update boundary. `--resume checkpoint.pt`
restores the model, optimizer, stage, and update counter. Set `--updates` to the
desired total count when resuming. Physics states are reset on resume.
An anchored resume collects fresh demonstrations for the restored stage.
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
full deployment stage, plus a zero-action baseline. The acceptance flag requires
40 robots, at least two active objects, at least 32 held-out episodes, at least
80% success, a 95% Wilson lower bound of 60%, and a success advantage above the
zero baseline of more than 20 percentage points. CPU checks cannot pass it.

The final evaluation belongs alongside native MuJoCo playback, contact and pickup
evidence, and warmstart-versus-PPO comparisons. Training loss alone does not measure
transport. No run is claimed as successful until its saved results pass the gate.

`scripts/verify_swarm_proof.py` audits a saved native rollout. It validates source
and artifact hashes, free-body topology, motor limits, seed-derived initial poses
and goals, complete timestamps, physical pickup and supported transport, released
payload rest, and formation arrival. It reconstructs every contact frame and
recomputes every neural action in bounded batches. Pose-derived actor inputs are
checked against geometry; instantaneous velocity inputs cannot be reconstructed
exactly because the recording contains positions rather than velocities.

The default verifier requires accepted A100/H100 training and matching weights.
`--allow-teacher` permits a teacher pipeline fixture and labels it explicitly;
it cannot produce `final_neural_proof=true`. Run the tamper checks with
`python scripts/test_swarm_proof.py --fixture PATH_TO_TEACHER_PROOF`.
