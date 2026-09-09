# Preliminary robot design

The selected robot uses four motors: differential drive, a vertical rack lift and mechanically coupled parallel jaws. Its 180 × 200 mm footprint fits the 280 × 480 mm start zone. One pair of stepped jaws handles the preliminary cylinders, kits and samples. The physical model is in `arena_mujoco/competition_robot.py`; `robot_design.json` records dimensions and controller limits.

| Concept | Advantage | Decision |
| --- | --- | --- |
| Vertical lift with stepped contact jaws | One mechanism picks all three object types; the thin sample pads clear the laboratory plate. | Selected for the physical proof. |
| Scoop with patient pockets and a gravity kit cassette | Can reduce delivery trips when objects are already sorted and loaded. | Deferred; adds gates and singulation while still needing precise sample placement. |

The preliminary scope has no beam-rotation mechanism. Four preloaded medical kits are permitted by the supplied national rules, but this first mechanism can also collect them from the floor. The current proof does not depend on a magazine.

## Geometry and construction

All robot coordinates use +Y forward. The body origin is 20 mm above the floor. Bounds are X ±90 mm and Y −75..125 mm. The 40 mm drive wheels have a 170 mm track and an axle at Y +5 mm. A 12 mm-diameter ball caster sits at Y −55 mm. Keeping the caster behind the drive axle prevents it catching the 3 mm laboratory edge during placement.

The grasp center is 105 mm ahead of the body origin, or 100 mm ahead of the wheel axle. At zero lift, the sample contact band is 3.4..4.8 mm above the floor. The broad upper pads span 6..16 mm and remain parallel, avoiding fore-aft cam forces during closure. The sample pads grip the top 1.4 mm of the disk's edge; they need no scoop beneath the disk. Once seated, the 5 mm disk projects above the 3 mm plate, leaving room to open the jaws. The 56 mm disk has 2 mm radial clearance in a 60 mm hole. At the 75 mm-inset lab center, the wheel envelope remains 5 mm behind the plate edge.

Each jaw slides outward by 0..25 mm. The sample-pad aperture is 18..68 mm. A central 8 mm pitch-radius pinion drives two opposing racks, giving equal travel. The native model implements this gear relationship with a joint equality and one actuator; the equality never references a task object. Both pad levels share the same aperture. The controller closes under a force limit instead of inferring force from object width. Kits approach with a 36 mm opening to clear their neighboring column, whose center is only 40 mm away.

Use printed PETG or nylon for the frame, slider blocks and replaceable finger carriers. Use two straight guide rods for the 80 mm vertical travel and the rack lift. Add thin silicone or TPU gripping inserts on the supported lower tips and flat upper faces. The narrow lower pad is 1 mm thick, with a supported 1.4 mm-high contact band. These are construction dimensions; mounting-hole locations and production tolerances still need detailed fabrication drawings.

The modeled mass is 0.800 kg. The rear battery/power compartment represents 165 g; the moving carriage represents 60 g, and both jaws total 44 g. The computed initial center of mass is Y −3.37 mm, Z 33.88 mm above the floor. It is 8.37 mm behind the drive axle. Raising the carriage and jaws by 80 mm raises the center of mass to about 44.28 mm. A 7.5 g payload changes the fore-aft center by roughly 1 mm. Mass allocation and contact coefficients are engineering priors pending hardware measurement.

## Catalog hardware and load checks

| Part | Verified specification | Use |
| --- | --- | --- |
| [Pololu #5188 gearmotor](https://www.pololu.com/product/5188/specs) | 75.81:1 HPCB 6 V; 430 rpm no load; 12 CPR motor-shaft encoder; 10 × 12 × 32 mm excluding output shaft; 11 g | Two wheel motors. Extrapolated stall torque is 1.1 kg·cm, approximately 0.108 N·m. |
| [Pololu #1452 wheels](https://www.pololu.com/product/1452) | 40 × 7 mm silicone tires; 3 mm D-shaft fit | Two drive wheels. |
| [ROBOTIS XL330-M288-T](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/) | 5 V; 0.52 N·m stall; 103 rpm no load; 20 × 34 × 26 mm; 18 g; current-based position and multi-turn modes | One rack-lift servo and one jaw servo. |
| [Pololu DRV8835 carrier](https://www.pololu.com/product/2135/specs) | Two motor channels; 1.2 A continuous per channel under the maker's stated test conditions; 1.5 A peak | Wheel drive, with regulated motor supply. |
| [Raspberry Pi Camera Module 3](https://www.raspberrypi.com/products/camera-module-3/) | 25 × 24 × 11.5 mm standard module; autofocus | Optional onboard visual alignment for a hardware build. The state-based simulation proof does not validate a camera pipeline. |

The native wheel torque cap is 0.025 N·m, with a linear torque-speed envelope and a 20 rad/s command limit, corresponding to 0.4 m/s. For an assumed 0.6 m/s² acceleration and rolling resistance of 0.03 times weight, each wheel needs about 0.0072 N·m. The hardware torque values above are stall specifications; the simulation caps are design settings, not measured continuous ratings.

At 1 N normal force per jaw and an assumed rubber/wood friction coefficient of 0.3, total vertical holding capacity is 0.6 N. The heaviest preliminary object is approximately 0.074 N using the arena's 600 kg/m³ wood-density prior. An 8 mm pinion and 65% assumed transmission efficiency require about 0.025 N·m to produce those jaw forces. The 4 N lift limit corresponds to about 0.049 N·m at the same radius and efficiency. Rate-limit lift targets to 80 mm/s and each jaw to 50 mm/s.

## Physical proof

Run:

```sh
.venv/bin/python -m unittest discover -s tests/mujoco -p test_competition_robot.py -v
```

The five tests use native MuJoCo friction contacts. The basic grasp coupons run at the arena's 200 µs timestep with CG and pyramidal friction. They verify the four-actuator count, 0.800 kg mass, symmetric jaws, cylinder/kit/disk lifts above 35 mm and gravity release. A kit initially 10 mm forward of center survives a reverse drive and 90° turn at a 1 ms timestep. Another coupon grasps a disk on the floor, drives 200 mm with wheel torques, lowers it, and releases it into a 60 mm hole in a 3 mm-thick ring.

The thin-sample endurance test holds the disk for 30 seconds at a 1 ms timestep with Newton, an elliptic friction cone and `impratio=100`, requiring less than 0.2 mm of vertical creep. An arena fixture using these settings measured 0.054 mm of creep over 30 seconds, compared with 0.545 mm at `impratio=10`. Grip-pad contacts retain `solref="0.003 1"` and `solimp="0.90 0.99 0.0003"`; overriding them with the harder field contact settings caused kit ejection during closure. Task objects remain free bodies throughout. All five tests pass without native warnings. Full-route success and the 120-second score belong to the mission evaluation.
