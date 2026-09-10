# Learning a connected magnetic swarm

Forty independently driven modules start inside the competition start zone.
They travel as a connected body, release four boundary modules to lift two
payloads, carry them 200 mm, and reconnect after delivery. Every robot and payload
has a free root joint. Wheel torque, pad contact, collision friction, and bounded
magnetic forces produce the motion.

The native demonstration controller has lifted and transported both payloads in
an integrated forty-module run. Transit cohesion and complete regrouping are still
being validated. Accepted magnetic-body GPU weights, held-out results, and final
videos are pending. The earlier independent-pair experiments and commands are
preserved in [swarm_transport_baseline.md](swarm_transport_baseline.md).

## Physical control

`arena_mujoco.swarm_body_env:SwarmBodyEnv` uses native MuJoCo at a 2 ms physics step
and a 20 ms control step. Each module has four shoulder ports. A port can attract
one partner, with a 0.15 N peak force and a 3 mm capture gap. Either module can
release the connection. The force and its associated moment act on robot roots;
payloads receive no applied magnetic wrench. Collision housings prevent overlap.

A shared flow goal and local neighbor feedback guide the body. Boundary modules
leave in sequence, align on opposite sides of a payload, and wait until both
partners are ready to pinch. The two carry teams wait for both payloads to be
supported before moving. Detached modules choose available shoulder ports and
approach through open space before aligning and rolling into contact. Docking
plans advance once after each physics control step; observation reads are pure.
The assigned target position and heading are exposed to the policy.

Wheel speed feedback uses a common gain of 0.00045 during navigation and 0.00015
during contact transport. Cylinder carriers use differential gain 0.00015 during
final approach and 0.00045 during supported transport. All gains share the same
0.002 N m torque cap and torque-speed envelope. Lift targets move at at most
10 mm/s. Task phases and geometric subgoals are programmed; the shared attention
actor supplies wheel, lift, and magnet commands at every control step.

The magnetic hardware direction follows
[Robot Pebbles](https://cba.mit.edu/docs/papers/10.05.knaian.ICRA.pdf).
[Granulobot](https://arxiv.org/html/2304.03125v2) motivates detachable connections
that permit deformation, using different rotating hardware. Our force envelope
and the internal packaging within a 40 g module are design assumptions.
[The module design](swarm_robot_design.md) documents CAD and contact experiments.

## Learning method

The shared actor embeds robot-local state and nearby modules, pools messages with
two-head attention, and emits a diagonal Gaussian followed by tanh. Deterministic
evaluation uses tanh of the Gaussian mean. A separate centralized critic sees
pooled scene information during training. Exported NumPy inference uses only the
actor and has no demonstration-controller fallback.

| Input or output | Shape | Contents |
| --- | --- | --- |
| Local observation | E × N × 40 | Local motion, object and target geometry, phase, contact feedback, body goal, magnetic degree and component fraction. |
| Neighbors | E × N × K × 12 | Relative neighbor state and measured magnetic connection. |
| Neighbor mask | E × N × K | Valid neighbors. |
| Critic state | E × 88 | Pooled local observations and task state. |
| Action | E × N × 4 | Left wheel, right wheel, lift, magnet enable, each in [-1, 1]. |

Reflection averaging exchanges wheel outputs, preserves lift and magnet enable,
and reflects all robot-local lateral coordinates. Torch and NumPy implement the
same operation. Previous wheel commands are also exchanged under reflection.

| Research | Use in this implementation |
| --- | --- |
| [MAPPO](https://arxiv.org/abs/2103.01955) | Shared actor, centralized critic, clipped policy/value updates, masked advantages, and KL stopping. |
| [Graph policy gradients](https://arxiv.org/abs/1907.03822) | Reuse neighbor aggregation across the swarm. This implementation uses attention. |
| [MDP homomorphic networks](https://arxiv.org/abs/2006.16908) | Encode bilateral observation/action symmetry through reflection averaging. |
| [MAPush](https://arxiv.org/abs/2411.07104) | Separate object subgoals from learned motor coordination. |
| [DAgger](https://proceedings.mlr.press/v15/ross11a.html) | Label states visited by the learner, aggregate physical rollouts, and refit the actor. |

Training first collects noisy demonstrations in the same contact simulator and
fits the actor. DAgger collects further labels under a decreasing teacher/learner
mixture. Phase and role balancing prevent stationary core examples from
outweighing grasp and release examples. PPO then updates the actor and critic
from sampled actions, actual MuJoCo transitions, and measured rewards. An optional
cloning retention loss preserves demonstrated contact precision during PPO.

Body rewards share supported transport progress, pickup, delivery, and drop terms
across modules. Additional terms measure body-goal progress, magnetic edge changes,
disconnection, penetration beyond 0.5 mm, magnet switching, energy, action changes,
and travel outside the arena. Ordinary robot contact is expected. Every reward
component is logged; payload movement counts as carried progress only with
physical support above the arena.

## Training and acceptance

Install `requirements-swarm.txt` in an A100 or H100 runtime. Native MuJoCo runs on
CPU workers; actor, critic, imitation fitting, and PPO optimization run on CUDA.
The connected-body task requires these arguments:

```bash
python scripts/train_swarm_policy.py \
  --env-factory arena_mujoco.swarm_body_env:SwarmBodyEnv \
  --backend native --num-envs 8 --native-workers 8 \
  --robots 40 --objects 2 --episode-seconds 100 --start-stage 3 \
  --hidden 128 --demo-steps 5000 --demo-noise .003 \
  --bc-updates 6000 --balance-demo-roles --bc-anchor-coef 100 \
  --initial-action-std .01 --dagger-rounds 4 --dagger-steps 5000 \
  --dagger-updates 4000 --dagger-teacher-prob .6 \
  --learning-rate .0001 --ppo-epochs 2 --clip .1 --target-kl .01 \
  --entropy-coef .0003 --horizon 128 --updates 200 \
  --eval-interval 40 --eval-episodes 32 --final-success .8 \
  --output output/swarm/body_policy
```

This is the proposed body configuration, awaiting complete physical validation.
The final report records the exact executed command, devices, source revision,
architecture, imitation counts, and PPO parameter changes. It must replace this
proposal when the accepted run is available. The body task starts at full stage;
earlier transport curriculum stages do not express its all-forty connectivity goal.

`progress.json` is the latest atomic status, `history.jsonl` records PPO updates,
and `checkpoint.pt` preserves the actor, critic, optimizer, and counters.
`bc_initial.npz` precedes DAgger. `dagger.npz` and `warmstart.npz` preserve the actor
before PPO. `weights.npz` is the portable actor export. `training.json` records
held-out evaluation and artifact hashes. `--resume` restores optimization state
and resets physical worlds. `--cpu-smoke` explicitly marks a local smoke check.

Validation and final evaluation use distinct seed offsets. Each world has a fixed
episode quota, including slow failures. Acceptance requires at least 32 held-out
full-stage trials, at least 80% complete success, a 95% Wilson lower bound of 60%,
a success advantage above the zero-action baseline greater than 20 percentage
points, CUDA training on A100/H100, and actual PPO actor updates.

A complete body episode requires both payloads delivered and released, at least
200 mm net displacement by every robot, all forty force-connected for at least
one second before release, a component of at least 32 throughout at least 95% of
post-release control samples, a new neighbor connection, and a settled all-forty
regroup. The independent proof strengthens connectivity checks to every physics
substep and requires five further seconds of connected, released rest.

## Proof, video, and download bundle

Record with the exact training episode duration because elapsed time is an actor
input. The verifier checks hashes, free roots, motor limits, initial conditions,
physical support and release, all four neural actions, observations, magnetic
loads, and connectivity. Every verification integrates every recorded command and
compares poses, velocities, contacts, and policy inputs throughout the episode.

```bash
python scripts/run_swarm_body.py --episode-seconds 100 \
  --output output/swarm/body_proof
python scripts/verify_swarm_body.py output/swarm/body_proof
python scripts/render_swarm_body_video.py --render
python scripts/package_swarm.py --body
```

The renderer requires a verified neural proof before producing top and side views
at 1080p and 720p. It renders recorded physical states. Explicit teacher diagnostics
cannot pass final neural acceptance or enter the final download bundle.
`output/swarm/swarm_body_complete.zip` contains the replay, weights, checkpoint,
training logs, CAD, source code, and both video projects and resolutions.

Focused checks are `test_swarm_magnets.py`, `test_swarm_flow.py`,
`test_swarm_body.py`, `test_swarm_body_proof.py`, `verify_swarm_policy.py`,
`test_swarm_imitation.py`, and `test_swarm_evaluation.py` under `scripts/`.
