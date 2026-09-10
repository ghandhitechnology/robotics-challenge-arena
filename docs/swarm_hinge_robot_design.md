# Hinge-driven swarm module

The hinge module keeps the two tapered shells and central spherical housing while removing the wheels, skids, and lift carriage. Three geared joints move the shells against the floor: front shoulder yaw, front pitch, and rear pitch. The module remains a free 40 g body and moves only through motor torque and passive MuJoCo contact.

## Geometry and body hierarchy

Local `+Y` is forward, `+X` is right, and `+Z` is up. The neutral module spans 20.000 × 54.000 × 16.000 mm. The magnetic variant adds two lateral housings to each lobe and spans 24.000 × 54.000 × 16.000 mm. These dimensions describe the compiled collision envelopes.

| Body | Parent | Mass | Native geometry |
| --- | --- | ---: | --- |
| `hinge_00` | world | 12 g | 8 mm-radius central housing |
| `hinge_00_front` | `hinge_00` | 14 g | front tapered convex lobe |
| `hinge_00_rear` | `hinge_00` | 14 g | rear tapered convex lobe |

The magnetic housings have zero additional simulation mass because the 14 g lobe budgets already include them. Their local port frames come from `HINGE_PORT_FRAMES`: two faces per lobe at `X = ±12 mm`, `Y = 8 mm`, and `Z = 1 mm`. Connector forces therefore act on the moving child bodies.

The exported STL and OBJ files are collision-envelope references. Motor cavities, shafts, bearings, transmissions, fasteners, PCB, battery, wiring, cable paths, compliant stops, and assembly clearances still require mechanical CAD.

## Joints and motor model

| Joint | Body-frame axis | Travel | Command |
| --- | --- | ---: | --- |
| `front_yaw` | `[0, 0, 1]` | ±30° | normalized position target |
| `front_pitch` | `[1, 0, 0]` | ±40° | normalized position target |
| `rear_pitch` | `[1, 0, 0]` | ±40° | normalized position target |

The planned module action order is `[front_yaw, front_pitch, rear_pitch, magnet_enable]`. The first three values map linearly from `[-1, 1]` to the joint limits. The fourth value controls the module's four magnetic ports.

The simulation uses three Pololu 2359-class 700:1 geared motors with these controller assumptions:

- operating torque cap: 8 mN·m;
- no-load speed: 45 RPM, or 4.712 rad/s;
- position gain: 0.04 N·m/rad;
- velocity gain: 0.0014 N·m·s/rad;
- reflected joint armature: `5e-6 kg·m²`.

The [Pololu 2359 product page](https://www.pololu.com/product/2359) lists a 6 mm × 21 mm, 1.3 g motor, 90 RPM free-run speed at 6 V, 12 oz-in stall torque, and a recommended instantaneous load below 3.5 oz-in. It also states that the 3 V values are approximately half the 6 V values. The model therefore uses the 3 V-equivalent 45 RPM speed and caps torque at about 1.13 oz-in. Three motors and their transmissions have not yet been packaged inside the exported envelopes.

## Swept envelopes

The exporter evaluates each body's influencing joints on a 41-point grid. It expands each sampled axis-aligned bound by `reach from joint axis × half grid step` for every influencing joint. A separate midpoint grid checks the space between all primary samples.

| Variant | Neutral X | Neutral Y | Neutral Z | Conservative swept X | Swept Y | Swept Z |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | ±10.000 mm | ±27.000 mm | -5.000 to 11.000 mm | ±15.433 mm | -28.525 to 29.020 mm | -16.543 to 20.245 mm |
| Magnetic | ±12.000 mm | ±27.000 mm | -5.000 to 11.000 mm | ±15.986 mm | -28.525 to 29.020 mm | -16.543 to 20.245 mm |

The maximum conservative expansion is 0.6823 mm. The primary calculation evaluates 1,723 body poses: one fixed central body pose, 1,681 front yaw/pitch poses, and 41 rear pitch poses. The independent midpoint check evaluates 1,641 poses and measured zero bound exceedance. These are full kinematic joint-limit envelopes; floor contact will make some extreme combinations unreachable during operation.

## Verified 1 ms motion coupon

The deterministic coupon runs each gait for six seconds at a 1 ms physics timestep. It applies the bounded motor model at every substep and records no external or generalized force.

| Case | Forward travel | Lateral travel | Yaw | Peak torque | Peak joint speed | Peak tilt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Forward | 19.181 mm | -0.000001 mm | 0.000001° | 4.472 mN·m | 1.706 rad/s | 7.617° |
| Reverse | -20.099 mm | -0.00000001 mm | 0.00000003° | 4.472 mN·m | 1.706 rad/s | 7.607° |
| Left | 19.343 mm | -1.649 mm | 7.120° | 4.452 mN·m | 1.597 rad/s | 7.570° |
| Right | 19.343 mm | 1.649 mm | -7.120° | 4.452 mN·m | 1.597 rad/s | 7.570° |

All four cases completed without MuJoCo warnings. Airborne fractions stayed between 0.05% and 0.10%, and every measured torque remained below the 8 mN·m cap. The steering cases use a ±9° front-yaw amplitude with a quarter-cycle steering phase.

## Export pipeline

Generate the base collision references and repeat the 1 ms motion coupon:

```sh
.venv/bin/python scripts/export_swarm_hinge_robot.py --validate
```

Generate the four-port lobe-mounted variant:

```sh
.venv/bin/python scripts/export_swarm_hinge_robot.py --magnetic
```

The default output directory is `output/swarm/hinge_robot`. The exporter writes:

- `robot.xml` with the isolated compiled module;
- `geometry.json` with compiled geometry and source-shape metadata;
- `cad_manifest.json` with body hierarchy, masses, joint axes and limits, actuator assumptions, hashes, and swept bounds;
- `neutral_assembly_mm.stl` and `neutral_assembly_mm.obj`;
- one body-local STL and OBJ pair under `parts/` for the central housing and each moving lobe;
- `motion_validation_1ms.json` when `--validate` is supplied.

The sphere uses a deterministic 48 × 24 display tessellation. The lobe vertices and triangles come directly from MuJoCo's compiled convex mesh. Box geometry in the magnetic export uses the exact compiled half-sizes and transforms.
