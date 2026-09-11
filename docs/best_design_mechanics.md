# Five robots for the senior preliminary

I would build five robots with one shared drive and gripper. RED, YELLOW and GREEN each move three cylinders. KIT delivers four cubes, two to the hospital and one to each PCC. LAB moves three discs and adds a 40 mm horizontal extension. The fleet has 21 motors, onboard cameras and separate batteries.

The official KOSAC FAQ permits any robot count and motor performance when everything starts inside the zone. The national PDF allows preloaded kits, but this baseline preserves all four original floor positions. Its vacant upper-left packing cell holds them. The 125 × 150 mm robot envelopes occupy 265 × 465 mm of the 280 × 480 mm start, with 15 mm between columns, 7.5 mm between rows and 7.5 mm outer margins. LAB reverses out through the lower-left opening first. A packing drawing establishes static fit; deployment must also clear moving bodies.

## Why five

| Fleet | Engineering assessment |
| --- | --- |
| 3 | Easier coordination, but multiple task types share each robot and serialize transport. |
| 4 | Sensible lower-cost fallback. Kit deliveries must join patient routes. |
| 5 | One robot per task type, common replacement parts, and room for the original kits. Recommended starting point. |
| 6 | Can split a long route, but this packing consumes the kit cell and requires preloading or a different arrangement. |
| 40 magnetic modules | Forty batteries, controllers and localization pipelines, plus coupling alignment and load transfer. The field has sixteen scoring objects and no preliminary wall requiring a reconfigurable structure. |

Five is an engineering choice, not a measured global optimum. Compare completion-time distributions only after every candidate has a legal start, physical grasps, deployment and release. The next useful speed improvement is a tested four-kit preload feeder; its gates and gravity feed need their own dynamics.

## Mechanical contract

Each courier has two 50 mm wheels on a 110 mm track, a 105 × 90 mm deck, a 45 mm cable lift and mechanically coupled parallel jaws with an 18–68 mm opening. The 0.45 kg mass is a budget. The CAD reserves actual component space, including a 65 × 30 mm compute board, 54 × 32 × 15 mm battery, controller, camera and two 20 × 34 × 26 mm XL330 actuator envelopes. Fasteners, guide rods, bearings, encoder motors, straps and cable routes are separate objects.

The drive model limits each wheel to 0.025 N m. With 25 mm wheel radius, this gives 1 N per wheel before transmission losses. The 0.35 m/s cruise target needs 14 rad/s, below the 20 rad/s command ceiling. Encoder micro gearmotors provide the relevant package size; choose the final ratio and current limit from measured acceleration and loaded speed.

LAB adds a screw-driven horizontal stage. It retracts inside the starting envelope and extends the tool center from 65 to 105 mm ahead of the axle. Approaching the lab from the west places the axle at x=963 mm and the leading wheel edge at x=988 mm, 5 mm before the plate starts at x=993 mm. This prevents the 3 mm plate from lifting the robot and changing the jaw height.

## The disc is the difficult part

A 56 mm disc in a 60 mm hole has only 2 mm radial clearance. The lower jaw pads contact the disc at 3.4–4.8 mm above the floor, above the 3 mm plate. They grip its exposed upper edge. After the disc settles onto the hole floor, the jaws open sideways above the rim, then lift away. The fingers never enter the narrow annulus.

The nominal vertical gap is only 0.4 mm. At 105 mm tool reach, roughly 0.22 degrees of pitch consumes that gap. Measure plate thickness, wheel compression, jaw sag and floor flatness before choosing the physical pad height. Horizontal error must remain below 2 mm; target 1 mm combined placement error to leave useful margin. A vacuum head would avoid side contact but introduces wood leakage, pump and pressure-sensor requirements, so it is a separate prototype option.

## Files and verification

`best_design.json` owns fleet dimensions and roles. `scripts/build_best_design_blender.py` recreates five named scenes through Blender MCP or Blender's Python entry point. `output/best_design/best_design.blend` preserves the incoming scene and contains the arena, starting fit, shared mechanism, disc section and extended LAB. The exported chassis STL is a fit mockup in millimeters. It has mounting holes but still needs material, tolerances and fastener design before fabrication.

`cad_geometry_audit.json` checks mesh bounds inside the start. MuJoCo uses simplified collision solids and force limits rather than every screw and cable in the CAD. Pad heights, wheel track/radius, start envelopes and LAB reach are the shared mechanical contract. Simulation results belong in their generated reports; the renders do not establish a 160-point physical run.

## Sources

- [KOSAC application and robot FAQ](https://apply.kosac.re.kr/onlnRcpt/getBscInfo.do?pbancNo=2026-S729), checked 11 September 2026.
- [National preliminary guide](../reference/challenge_2026_full.pdf), printed pages 10–13 and 17–19. The guide specifies the 120-second round, 160-point tasks, starting fit, permitted kit preload and wooden game pieces.
- [Pololu 5188 specifications](https://www.pololu.com/product/5188/specs), 32 × 12 × 10 mm motor body plus shaft; encoder extends outside that cross-section.
- [ROBOTIS XL330-M288 specifications](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/), 20 × 34 × 26 mm, recommended 5 V operation.
- [Raspberry Pi Zero 2 W](https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/), onboard compute reference.
