# Miniature magnetic swarm body module

The module keeps the reference photograph's two tapered gray lobes and central dark ball. It has two concealed drive wheels, a lifting rubber contact face, and four releasable magnetic shoulder ports. Forty identical modules can join, flow around obstacles, split, and reconnect as one mobile body. Each module remains a free body with three motors.

The photograph shows the exterior only. Dimensions, mass, transmission layout, tire compound, sensors, lift, and magnetic hardware are engineering proposals. The native tests validate the simulated geometry and force model. No magnetic prototype has been built or measured.

![Magnetic body module rendered in Blender](../output/swarm/body_robot/module_overview.png)

| Property | Proposed value |
| --- | --- |
| Ground footprint | 24 mm wide × 55 mm long |
| Local bounds | X −12…12 mm, Y −27…28 mm |
| Body reference height | 7 mm at level wheel contact |
| Height clearance budget | 23 mm including raised pad |
| Complete mass | 40 g including trim ballast |
| Drive | Two independent 14 × 3 mm tires, 21 mm track |
| Magnetic ports | Four shoulder ports at local X ±12 mm, Y ±14 mm, Z 4 mm |
| Magnetic force model | 0.15 N peak per port, 3 mm face-gap capture reach |
| Magnetic release | Coordinated electropermanent magnet handshake |
| Wheel command | −20…20 rad/s, approximately ±140 mm/s unloaded |
| Wheel torque limit | 2 mN m per wheel |
| Lift travel | 8 mm |
| Lift force and speed limits | 0.5 N, 10 mm/s |
| Low contact band | 3.4…4.8 mm above the floor at home |
| Upper contact band | 5…13 mm above the floor at home |
| Contact face | 10 mm wide, 28 mm forward of the wheel axle |

## Hardware and packaging

Two longitudinal 6 mm planetary gearmotors sit in opposite lobes. Custom bevel transfers drive separate half axles through the central housing. This arrangement preserves the narrow body; the two motors cannot simply sit end-to-end across a 24 mm chassis. A third longitudinal motor drives a guided vertical rack at the front. The ball is a fixed housing for the shafts and wiring. Its visible shape does not imply an actuated ball joint.

The candidate drive motor is the [Pololu 2358, 136:1 sub-micro planetary gearmotor](https://www.pololu.com/product/2358). Its body is 6 mm in diameter and about 18.7 mm long. Pololu specifies 500 rpm unloaded at 6 V and approximately half that at 3 V. The model uses 250 rpm as the no-load speed and a 2 mN m output limit. Those are distinct from the manufacturer's much larger short-duration stall figure.

The candidate lift motor is the [Pololu 2359, 700:1 sub-micro planetary gearmotor](https://www.pololu.com/product/2359). A proposed 3 mm pitch-radius pinion requires 1.5 mN m to produce the modeled 0.5 N rack force. Its approximately 45 rpm unloaded speed at 3 V gives 14 mm/s at that radius, leaving room for the 10 mm/s commanded limit. Motor diameters, shaft lengths, and gearhead lengths are in the [manufacturer's dimension drawing](https://www.pololu.com/file/0J1831/sub-micro-plastic-planetary-gearmotor-dimension-diagram.pdf). The transmission needs bearings and a lift position sensor; the simulation replaces its gear train with the equivalent force-limited slide.

The 40 g budget allocates 35 g to the body assembly, 2 g to the wheels, and 3 g to the lift carriage. The body allocation must now cover the three motors, four EPM units, a small 1S LiPo, control electronics, encoders, shell, bearings, wiring, and any trim ballast. The four housings fit inside the unchanged 24 × 55 mm footprint, but the exported solids do not contain cavities or routing. Detailed CAD and a measured mass rollup must establish whether all of this fits before ordering forty sets. A laboratory overhead camera can supply global pose and object tracking; motor encoders and the lift sensor supply local feedback. The source gearmotors do not include encoders.

Each port is modeled as a two-sided electropermanent magnetic connection. A proposed controller would pulse a reversible current through a port coil to switch its magnetic state, then coordinate release with the module at the other end of the link. This adds switching electronics and four magnetic ports, not four more motors. Coil construction, pulse voltage and current, switching energy, standby draw, driver topology, connector routing, heat, and battery capacity are still open hardware decisions. The policy currently sends one magnet-enable command per module, which applies to all four ports; the lower-level interface also accepts a separate enable bit for each port.

[Robot Pebbles](https://cba.mit.edu/docs/papers/10.05.knaian.ICRA.pdf) is the scale precedent for this proposal. Gilpin et al. report 12 mm modules with electropermanent connectors, 2.06 to 3.18 N normal holding force, and 4.31 mm two-sided pull-in. Those measurements support investigating EPMs at this size. They do not validate this module's 0.15 N force curve, 3 mm capture reach, packaging, or power budget.

## Contact and cooperation

The magnetic ports let neighboring modules transmit finite forces while each module keeps its own wheel drive and free joint. A port accepts one partner. Attraction acts only within the 3 mm capture region and falls to zero at its boundary. The connection has no weld constraint or break impulse. Disabling either endpoint removes its magnetic force, after which wheel motion can separate or shear the pair apart. Housing contact and friction carry compression when faces meet.

Opposing modules drive their pads inward to establish normal force, raise their pads together, and translate while keeping that pressure. When the pair lies along the direction of travel, one module drives forward and the other reverses. Both wheel commands still contain the inward pressure component. Lowering the pads returns the object to the floor before the modules back away.

The low pad grips the upper edge of the 5 mm sample disc. Its underside clears a 3 mm laboratory plate when the module remains level. The upper pad supports the 20 mm cylinder and kit sides. Sliding, dropping, tipping, and collisions remain possible because every object retains its free joint. Payload handling adds no weld, suction, magnetic attachment, external payload force, or commanded body translation.

A two-module team is sufficient for the tested cylinder, kit, and sample. Long beams need additional contact points and a suitable formation. Several teams can carry separate objects simultaneously. The exported model does not claim that forty modules have completed an arena mission or that a learned policy has acquired this behavior.

Robot geoms use collision bit 4 with affinity 7. That permits robot-to-robot and arena contact. MuJoCo filters rigidly attached geometry and direct parent-child bodies within each module. The remaining internal shapes do not intersect. Tire sliding friction is 0.85, rubber-pad friction is 0.7, and rounded skid friction is 0.04. These are tunable material assumptions. Per-geom priority 2 preserves those coefficients against the arena's lower-priority material geoms.

## Controller interface

`arena_mujoco.swarm_robot.add_swarm_robot(root, index, position, yaw)` adds one module. `add_swarm_robots` adds up to forty. `arena_mujoco.swarm_magnets.add_magnetic_docks(root, specs)` then adds four housings and four oriented sites to every supplied module. The returned metadata records the port names and force assumptions. Names use `swarm_00` through `swarm_39`. The default fleet poses form a standalone display grid; the arena environment supplies its own obstacle-aware starting poses.

Local +Y is forward, +X is right, and +Z is up. Positive wheel joint rotation moves the module backward. The magnetic-body policy issues four normalized controls per module: left wheel speed, right wheel speed, lift position, and magnet enable. Positive policy wheel controls request forward motion, so native joint targets equal −20 times the wheel control in rad/s. Map lift control −1…1 to 0…8 mm. Forty modules retain 120 motor actuator channels and add 40 logical magnet commands. Magnetic force acts between aligned robot ports only; it never attaches or applies force to a payload.

The contact-test wheel servo applies `clip(0.00015 * (target_speed - measured_speed), -0.002, 0.002)` N m. The body environment separates the common wheel-speed error from the differential steering error. Common gain is 0.00045 for navigation and 0.00015 while pinching or carrying. Cylinder carriers use differential gain 0.00015 during final approach and 0.00045 during supported transport to hold their headings. All modes retain the same 2 mN m torque limit and motor torque-speed envelope. Lift targets change at no more than 10 mm/s. MuJoCo also enforces actuator force limits.

A stationary opposed pair uses −3 rad/s on both wheels of both modules. At zero speed this requests 0.45 mN m per wheel, about 0.129 N combined forward force before skid losses. The ideal two-pad vertical friction budget at this force is roughly 0.18 N, or 18 g; the modeled objects weigh approximately 4…7.5 g at the test density of 600 kg/m³.

For a common world +Y speed of 14 mm/s, the lower module requests −5 rad/s and the upper module requests −1 rad/s. Yaw feedback holds headings 0 and π. The correction is `clip(8 * yaw_error, -2, 2)` rad/s, converted to differential wheel speed with `track / (2 * radius)`. The left wheel adds this correction and the right wheel subtracts it. The longer cylinder test loses its grasp without heading feedback, so the stabilizer is part of the demonstrated controller.

## Native validation

Run:

```sh
.venv/bin/python scripts/export_swarm_robot.py --magnetic --validate --render
```

The script exports the magnetic MJCF, geometry JSON, design JSON, and preview to `output/swarm/body_robot`. Its six contact coupons repeat the three object tests at both 1 ms and 2 ms timesteps with all four housings installed. Each lasts 29 simulated seconds: close, lift, translate, lower, back away, then leave the payload undisturbed for five seconds. Two more coupons check native magnetic joining, coordinated release with no pose edit, and wheel-driven separation at both timesteps.

| Object | Travel at 1 ms | Minimum floor clearance while carrying | Release |
| --- | --- | --- | --- |
| 20 × 20 mm cylinder | 210.8 mm | 7.45 mm | On floor |
| 25 × 25 × 20 mm kit | 209.2 mm | 7.18 mm | On floor |
| 56 × 5 mm disc | 209.2 mm | 6.64 mm | On floor |

All six tests maintained both modules' pad contacts throughout the measured carry interval, had zero payload-to-floor contacts during that interval, and emitted zero solver warnings. The report records the exact initial poses, controller timing, torque peak, and minimum clearance. These are deterministic mechanism tests from aligned starting poses. Learned navigation, crowded formations, object transfer onto the laboratory, and forty-module mission success require separate evaluation.

Both magnetic coupons joined through two aligned port pairs and released without changing `qpos` or leaving an applied force. Opposing wheel commands then produced 127.8 mm of relative shear at 1 ms and 120.7 mm at 2 ms, with no solver warnings.

## Magnetic Blender and STL assets

`output/swarm/body_robot/swarm_body_robot.blend` contains the exact native collision assembly, four visible dock-site markers, materials, and three review cameras. `assembly_reference_mm.stl` contains the complete solid assembly in millimeters. The `parts` directory has the nine legacy solids plus four magnetic housing STLs. `cad_manifest.json` records their dimensions, assembly coordinates, SHA-256 hashes, dock transforms, and force assumptions. `bundle_manifest.json` adds the byte count, CRC-32, and SHA-256 digest of every packaged file. The Blender scene uses meters with a millimeter display scale.

These solids provide a mechanical layout reference. Shell cavities, motor mounts, gears, bearing seats, fasteners, PCB layout, and manufacturing fits still require detailed engineering. The separate low and upper rubber contact bands remain separate parts in the export, matching the simulation.

Regenerate the CAD assets and review images after exporting the native geometry:

```sh
/Applications/Blender.app/Contents/MacOS/Blender --background --python scripts/render_swarm_robot.py -- --magnetic
```

`module_overview.png` shows the complete module. `contact_detail.png` frames one shoulder housing and its site face. `module_top.png` provides an overhead view of all four ports. The renderer checks that the assembly and every part STL have finite vertices and closed edges, then writes `output/swarm/swarm_body_robot_assets.zip` and verifies its ZIP CRC records.

## Arena integration

The finite floor box and forty-module starting layout were checked in the actual arena. The complete starting footprint spans X 872…1113 mm and Y 717…1108 mm, inside the 280 × 480 mm start zone. The relocated transport payloads leave that region empty.

The elliptic friction solver needs an adequate Newton line-search budget for these small bodies. At ten line-search iterations, the full arena became unstable during an unpowered settling test at both 1 ms and 2 ms timesteps. Increasing only `ls_iterations` to 50 stabilized the same finite-floor geometry for a one-second settling test at both timesteps. More main solver iterations alone did not fix the failure. The integration uses 50 main iterations and a tolerance of `1e-8` as well. This preserves the modeled floor and contact coefficients.
