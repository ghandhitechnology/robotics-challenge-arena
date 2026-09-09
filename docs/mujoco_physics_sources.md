# MuJoCo material and tape physics sources

Reviewed 9 September 2026. This records the implemented model, its source evidence and remaining calibration work. The rulebook specifies geometry and wooden pieces; it does not identify the tape brand, paint, paper coating, wood density or measured contact coefficients. Values marked **prior** below are proposed exploration ranges, not measurements of the competition kit.

## Verified engine capabilities

The installed `.venv` package reports **MuJoCo 3.12.0**, matching the [official release of 20 August 2026](https://github.com/google-deepmind/mujoco/releases/tag/3.12.0). A local probe compiled and stepped a 3 × 3, `dim="2"`, `dof="full"` flex with `elastic2d="both"`, thickness 0.00015 m and collision radius 0.000075 m. It also compiled geom and pair adhesion, and confirmed that `flex/contact adhesion` is rejected by the schema. These probes establish availability, not full arena stability.

Use a native triangular flex shell for the tape. The XML API exposes Young's modulus in Pa, Poisson ratio below 0.5, shell thickness in metres and stiffness-proportional Rayleigh damping in seconds. Bending and stretching can both contribute. Keep full vertex DOFs for the initial model. Set `contact passive="false"`: the experimental passive path removes friction from static-floor contact. [Versioned flex reference](https://mujoco.readthedocs.io/en/3.12.0/XMLreference.html#flex-elasticity).

Native contact adhesion, added in 3.11, is a **force per contact in N**. It allows tension and raises the sliding budget to `mu * (normal_force + adhesion)`. Its total effect changes with contact count. It has no peel-history or irreversible damage state. Adhesion belongs to geom and geom-pair declarations; the 3.12 flex contact schema has no such field. Leave adhesion zero on the arena floor, blocks, wheels and tape top. [Contact adhesion computation](https://mujoco.readthedocs.io/en/3.12.0/computation/index.html#adhesion), [3.12 schema](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/xml/mjcf.schema).

Full flex bending has a curved-reference contribution. A naturally curved reference mesh, flattened in runtime vertex positions while bonded, can store the bending energy that curls a released edge. A flat reference mesh has no intrinsic curl. Verify the released-strip equilibrium and bending force before integrating it into the arena. [Native bending implementation](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_passive.c).

## Manufacturer evidence for 0.15 mm tape

[3M Temflex 1500 manufacturer datasheet](https://multimedia.3m.com/mws/media/1983306O/3m-temflex-vinyl-electrical-tape-1500-datasheet-en-eu.pdf) supplies a close physical analogue: PVC tape, total thickness 0.15 mm, tensile breaking load 20 N per 10 mm width, elongation at break 170%, and adhesion to steel and backing of 1.8 N per 10 mm. These are typical product values. The competition tape has not been identified as this product.

For a 20 mm strip, those width-normalized figures imply 40 N breaking load and 3.6 N peel-test force. Peel resistance is 180 N/m of peel-front width. It is not a tensile stress, a force per mesh node, or a universal support-bond strength. The test uses steel or tape backing; the competition map includes paper. The datasheet does not provide Young's modulus, backing/adhesive thickness split, density, shear-creep law, fracture energy, surface friction or residual curvature. Dividing breaking stress by breaking strain does not recover the small-strain elastic modulus.

[3M's application guidance](https://www.3m.com/3M/en_US/electrical-construction-maintenance-us/products/vinyl-electrical-tape/temflex/) connects excessive stretch during application with end lifting. Represent application strain, rest curvature and initial edge defects as explicit setup parameters.

## Implemented underside bond model

`arena_mujoco/tape.py` controls native geom-pair adhesion at area-weighted quadrature points attached to full-DOF tape flex vertices. The visible PVC top has dry friction. Each hidden spherical underside point bonds to its actual support. Floor bonds use the finite floor box. Per-node sphere contacts avoid the variable contact multiplicity of box/plane adhesion. The controller also has an experimental interlayer path connecting upper and lower tape nodes; full-arena overlap geometry did not pass the stability checks below.

The controller sets pair tensile capacity in newtons:

```text
capacity = peak_traction_pa * patch_area_m2 * (1 - damage) * strength_factor
normal_failure_length = 2 * fracture_energy_j_m2 / peak_traction_pa
shear_failure_length = 2 * shear_fracture_energy_j_m2 / shear_strength_pa
mixed_ratio = hypot(opening / normal_failure_length,
                    shear_weight * slip_length / shear_failure_length)
history = max(previous_history, mixed_ratio)
damage = max(previous_damage, clip((history - onset_ratio) / (1 - onset_ratio), 0, 1))
```

This produces a capacity plateau followed by linear softening. Native soft contact supplies the near-contact response; the controller adds no explicit stiff spring through `qfrc_applied`. The nominal fracture-energy parameter sets the separation scale. Actual work also depends on contact compliance, mixed loading and shell mechanics, so both reported energy quantities have `_proxy_j` names. They are diagnostic sums, not a verified thermodynamic balance.

The implementation enables the model's native adhesion flag when configuring dynamic pair capacities. Changing `pair_adhesion` alone after compiling zero-adhesion XML leaves the attractive branch disabled. This was checked with a finite-floor single-contact force test. The flex's collision masks exclude a second flex/floor contact at the same location; the underside points already supply that compression reaction. Duplicate floor constraints caused millimetre-scale drift and numerical warnings in an isolated native-shell probe.

The full arena uses 30 kPa peak traction, 120 J/m² separation-scale parameter and 0.05 mm damage onset. The isolated coupon controller defaults to 0.1 mm onset. The nominal opening failure length is 8 mm. Damage is irreversible and updates once per accepted physics step. Inverting the local underside normal or exceeding the configured interaction gap releases the bond. Geometry follows the local shell and deforming support. The controller serializes damage, reference frames, support anchors, rate history, contamination, wear and dwell with the episode state.

Rate strengthening, dwell, contamination and contact-work-driven wear have explicit coefficients. Their default rate/wear effects are zero. Optional rebonding requires sustained compressive contact pressure and sufficiently low slip speed; it is off by default. Shear resistance follows the native adhesive friction cone, while mixed separation controls damage. The implementation has no independently calibrated shear-creep law.

Curved native reference geometry stores bending energy when the builder flattens the laid tape. Released edges then lift through native shell elasticity. The native linear material law has no PVC plasticity or topology change. `tape_max_tensile_strain` reports the largest positive edge extension relative to the compiled reference mesh. `tape_material_limit_exceeded` flags a configurable `max_valid_strain`, initially 10%, to bound training to the intended small-strain material range. This is a model-validity event; it does not simulate a torn backing.

Fit peak traction, separation scale and rate coefficients against simulated 90° and 180° peel coupons at stated speeds. A single peel-force datum cannot identify all three. Even the elementary inextensible-peel estimate changes with peel angle; backing stretch and adhesive dissipation also change the result. Actual-map measurements are needed to calibrate the paper interface and residual curvature.

## Checked native tape behavior

`tests/mujoco/test_tape.py` checks native finite-floor hold/release, irreversible mixed separation, JSON state restoration, underside orientation, pressure/dwell rebonding, intact shell equilibrium, area-scaled total capacity, released-edge curl, controlled grip peeling and the small-strain limit. The refinement check compares a 20 × 40 mm coupon at 5 and 2.5 mm mesh spacing, using a native equality grip pulled upward at 0.2 m/s and a 50 µs timestep. It requires mean mid-peel force to differ by less than 15% and total pull work by less than 10%.

A 0.22 s refinement probe produced the following values. Mean force covers mean nodal damage between 0.2 and 0.8; pull work integrates the absolute vertical grip reaction over commanded upward travel.

| Mesh spacing | Timestep | Mean mid-peel force | Peak force | Pull work |
| --- | --- | --- | --- | --- |
| 10 mm | 100 µs | 1.689 N | 3.003 N | 0.03027 J |
| 5 mm | 100 µs | 2.129 N | 2.901 N | 0.03646 J |
| 5 mm | 50 µs | 2.127 N | 2.919 N | 0.03633 J |
| 2.5 mm | 50 µs | 2.295 N | 2.849 N | 0.03749 J |

These runs completed without native numerical warnings and show timestep convergence at 5 mm and a smaller mesh-refinement change in work. The 10 mm mesh underpredicts mean peel force relative to finer coupons. A single refinement pair does not establish asymptotic convergence of every output. The short curl check establishes released-edge motion; long-duration curl equilibrium remains a separate convergence target. The manufacturer peel value above uses a different test configuration and is not the calibration target of these checks.

A 0.3 s default full-arena run without the robot, at 200 µs, retained all 1,323 bonds without damage or numerical warnings. Maximum slip was 7.10 µm and maximum tensile edge strain was 0.0115%. This checks initial full-arena equilibrium; driven robot contact is covered separately.

Full-arena overlap trials failed. Aligning the raised transition with the lower-strip boundary removed the initial 148 µm mesh penetration, but deformation still diverged. Smaller interlayer proxies, 50 µs steps, reduced contact dimensions, softer contact regularization and selective flex contact exclusions did not establish stability. A separate trial replaced all 36 interlayer adhesive pairs with native connect equalities, initialized with zero constraint error; it failed at 0.0292 s. These results support the default butt-join layout and do not establish a usable layered-arena model.

## Dry contact and physical parameter ledger

MuJoCo geom friction is `[sliding, torsional, rolling]`. Sliding is dimensionless; torsional and rolling coefficients have units of metres because they bound torque relative to normal force. `condim=3` has sliding, 4 adds torsion, and 6 adds rolling. [Contact dimensions and units](https://mujoco.readthedocs.io/en/3.12.0/computation/index.html#contact).

By default, equal-priority contacting geoms use the component-wise maximum of their friction values. Use explicit pair calibration or a documented material-pair policy; assigning a large generic value to every block would hide the paper/PVC distinction. [Contact parameter mixing](https://mujoco.readthedocs.io/en/3.12.0/modeling.html#contact-parameters).

The native contact cone provides sticking and sliding under one coefficient; it does not expose separate static and kinetic surface coefficients. A measured static/kinetic transition needs a deliberate velocity/history-dependent contact extension. Motor LuGre friction is a separate motor mechanism, not a replacement for wheel-ground friction. Avoid claiming that arbitrary `solref` changes create a calibrated static/kinetic law.

| Parameter | Proposed prior or constraint | How to identify it |
| --- | --- | --- |
| Tape total thickness | 0.15 mm analogue; measure kit | Micrometer away from overlap; record application stretch. |
| Effective tape modulus | 2-100 MPa prior; 10 MPa initial coupon value | Low-strain force-extension curve; fit bending separately if one homogeneous layer is insufficient. |
| Poisson ratio | 0.35-0.49 prior | Width change under small longitudinal strain. |
| Tape effective density | 1000-1600 kg/m³ prior | Weigh known area; shell mass is area × thickness × density. |
| Rayleigh damping | 0.00001 s current numerical default; identify from tests | Released-strip decay at more than one frequency. |
| Peel resistance | 180 N/m manufacturer analogue; 20-400 N/m exploration prior for unknown map interface | Actual-map peel force at known width, speed, angle and application dwell. |
| Bond peak traction, stiffness, shear creep, mixed-mode toughness | Unidentified; no measured numeric defaults available | Pull-off, lap shear, creep and peel curves together. Numerical stiffness must be reported separately from a fitted material value. |
| Rest curvature/application strain | Measured setup parameter; zero is a deliberate no-curl assumption | Released strip radius and marked-length relaxation. |
| Painted wood density | 400-900 kg/m³ prior | Weigh each piece type; account for lab holes when computing volume/inertia. |
| Painted wood/paper or wood/PVC sliding friction | 0.15-0.9 prior, independently fitted | Incline onset for static resistance and horizontal pull/coast for sliding. |
| Wheel/paper or wheel/PVC sliding friction | 0.3-1.3 prior, independently fitted | Slip/traction tests with the actual wheel, normal load and surface. |
| Torsional/rolling friction | 0-0.003 m / 0-0.0005 m sensitivity ranges | Spin-down and rolling coast; do not copy dimensionless sliding coefficients into these fields. |
| Contact compliance and restitution | Fit separate force-depth and drop/impact observations | Check penetration at expected loads and rebound across impact speeds. `solref` is a solver parameter, not directly a Young's modulus or restitution coefficient. |

Use a task-scale rigid support and wooden bodies unless their flexure is observed to matter; retain exact collision shape, mass and inertia. Treat paper coating, paint, dust, humidity, application dwell and temperature as recorded episode conditions that modify fitted parameters. This recommendation assumes indoor conditions and does not include heat conduction, electrical breakdown or chemical ageing dynamics. Surface wear, paper delamination and PVC tearing need explicit state/model coverage when those outcomes enter the intended training regime.

## Numerical validation and RL backend

Keep shell thickness, collision radius, tape mass and bond strength independent. A larger collision radius changes the obstacle a wheel feels. Do not increase tape mass to cure instability. Custom bond forces passed through `qfrc_applied` are not automatically made implicit by `implicitfast`; test the fastest bond mode or integrate the bond response implicitly. Start with small timesteps and halve the timestep and mesh spacing until peel force, sliding distance and curl height converge within a declared tolerance. Confirm that damage dissipates energy and that a block placed on intact tape top experiences no tensile attachment.

CPU native MuJoCo is the initial reference backend. The official 3.12 feature table lists flex support in MJX-Warp and no flex support in MJX-JAX. A Colab A100 can be considered for Warp after the complete shell, contact and bond-update implementation passes the same tests; Python callbacks do not automatically run as device kernels. Rigid or texture-only tape used for faster RL is a separate approximation. [Versioned MJX feature table](https://mujoco.readthedocs.io/en/3.12.0/mjx.html#feature-parity).

Minimum physical checks: dry floor/PVC sliding comparison; top-side non-adhesion; underside hold and progressive peel; shear-slip/creep; released-edge curl; overlap force transfer; mass/inertia consistency; force-width and fracture-work/area scaling; timestep/mesh convergence; deterministic reset of every wear and bond state. These checks establish consistent simulated behavior. Agreement with the actual kit requires its measurements.
