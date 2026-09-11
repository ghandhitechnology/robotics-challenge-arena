# Five robots for the senior preliminary

I would build five robots with a shared drive, three patient grippers and two specialist mechanisms. RED, YELLOW and GREEN each move three cylinders. KIT preloads four cubes in three hinged chutes, two side by side for the hospital and one for each PCC. LAB moves three discs and adds a 40 mm horizontal extension. The fleet has 22 motors, onboard cameras and separate batteries.

The official KOSAC FAQ permits any robot count and motor performance when everything starts inside the zone. The national PDF allows preloaded kits. This design loads all four on KIT before the start and leaves the upper-left packing cell vacant. The 125 × 150 mm robot envelopes occupy 265 × 465 mm of the 280 × 480 mm start, with 15 mm between columns, 7.5 mm between rows and 7.5 mm outer margins. LAB reverses out through the lower-left opening first. A packing drawing establishes static fit; deployment must also clear moving bodies.

## Why five

| Fleet | Engineering assessment |
| --- | --- |
| 3 | Easier coordination, but multiple task types share each robot and serialize transport. |
| 4 | Sensible lower-cost fallback. Kit deliveries must join patient routes. |
| 5 | One robot per task type and common replacement parts. Recommended starting point. |
| 6 | Fits the vacant upper-left cell after kit preloading. A flexible courier can split both the lower YELLOW and GREEN queues; the fleet then has 26 motors and another departure to schedule. |
| 40 magnetic modules | Forty batteries, controllers and localization pipelines, plus coupling alignment and load transfer. The field has sixteen scoring objects and no preliminary wall requiring a reconfigurable structure. |

Five is the validated baseline. In `output/best_design/traffic_11/report.json`, the native fleet scores 160 at 115.14 seconds with no failed or unfinished roles. All 50 checks over the following five seconds retain 160 points. The sixth cell remains available because the four kits are preloaded; robot count has not been globally optimized. The separate sixth-courier experiment gives that robot two lower YELLOW cylinders and one lower GREEN cylinder, reducing both existing queues.

The adopted kit feeder removes repeated return trips to the start. Its doors and falling cubes use explicit contact dynamics. Each patient courier reaches the farther cylinder through a row it has already cleared. The west lane is reserved for RED before YELLOW descends, and completed robots park away from transit.

The six-robot factory variant passes initial placement checks. Its first two native schedules collided with KIT at the shared crossing or lower-PCC departure; neither completed the mission. Those runs do not establish a completion-time advantage. The five-robot result remains the measured baseline.

The original loaded LAB model placed its center of mass only 3.5 mm behind the wheel axle support line and reached 18.1 degrees of forward pitch. The adopted passive front ball sits 20 mm ahead of that axle with a 0.5 mm floor gap. Its 468 g prototype completed the full mission at 117.88 seconds, kept all 50 hold checks at 160, and limited LAB pitch to 2.44 degrees. The camera adds a further 6 g, giving the current LAB a 474 g mass budget. The current camera-frame, reserved-lane and deployment-recovery configuration in `output/best_design/corrected_scoring_nominal/report.json` completes the native mission at 80.80 seconds. KIT continues west after deployment without returning to the right lane. The couriers use 0.48 m/s and 4 rad/s, while LAB retains 0.35 m/s and 2.5 rad/s. Peak fleet tilt is 7.16 degrees. The frozen e2 benchmark succeeds in 299 of 300 randomized missions; its later scorer review is recorded separately so the benchmark retains its original source provenance. Full metrics are reported in `docs/best_design_training.md`.

The native model uses simplified package shapes. KIT servo volumes, wiring and fasteners are more detailed in CAD than their physics proxies. Measure the assembled center of mass, caster gap and wheel loading before hardware tuning.

## How the fleet works

Departures share the confined start cells, then roles execute concurrently on one clock. The scheduler releases a following robot when its required cell is vacant. It reserves the narrow west lane for KIT and RED before YELLOW descends. Patient pickup approaches use a cleared cylinder row, which prevents a carrying robot from driving through the next cylinder.

| Role | Work order | Mechanical action |
| --- | --- | --- |
| KIT | Hospital pair, upper PCC, lower PCC | Stop over each destination and open the corresponding gravity door. |
| RED | Upper west, lower west, lower east | Grip, lift, travel through the shared west corridor, and release wholly inside H. |
| YELLOW | Upper west, lower west, lower east | Deliver one cylinder to upper PCC, then two to lower PCC after RED clears. |
| GREEN | Upper east, upper west, lower east | Clear the two upper pickups first, then wait for LAB before taking the lower cylinder to recovery. |
| LAB | Sample 1, sample 2, sample 3 | Carry each disc to its own hole, extend the tool while wheels stay off the plate, lower, open and retract. |

All drive commands pass through finite motor torque and wheel contact. Payloads remain free bodies. Early declaration before parking requires all 16 release events and six successful score checks at 0.1-second intervals, spanning half a second. Every termination, including normal program completion, then holds the robots for five seconds and checks the score 50 times. A future onboard implementation needs the same completion decision from verified delivery observations. The current full-mission evaluator obtains those observations from simulator poses; the trained RGB proposal and precision-refinement modules are evaluated separately.

## Mechanical contract

Each patient courier has two 50 mm wheels on a 110 mm track, a 105 × 90 mm deck, a 45 mm cable lift and mechanically coupled parallel jaws with an 18–68 mm opening. The common base has a 0.45 kg mass budget; LAB adds 18 g of passive support and 6 g of inspection camera. The CAD reserves actual component space, including a 65 × 30 mm compute board, 54 × 32 × 15 mm battery, controller, camera and two 20 × 34 × 26 mm XL330 actuator envelopes. Fasteners, guide rods, bearings, encoder motors, straps and cable routes are separate objects.

The drive model limits each wheel to 0.025 N m. With 25 mm wheel radius, this gives 1 N per wheel before transmission losses. The 0.35 m/s cruise target needs 14 rad/s, below the 20 rad/s command ceiling. Encoder micro gearmotors provide the relevant package size; choose the final ratio and current limit from measured acceleration and loaded speed.

LAB adds a screw-driven horizontal stage. It retracts inside the starting envelope and extends the tool center from 65 to 105 mm ahead of the axle. Approaching the lab from the west places the axle at x=963 mm and the leading wheel edge at x=988 mm, 5 mm before the plate starts at x=993 mm. This prevents the 3 mm plate from lifting the robot and changing the jaw height.

The LAB inspection camera moves with the lift and horizontal stage. Its downward optical center is 104.1 mm above the floor at start. Four frame walls leave a 12 × 12 mm clear aperture; a 2 g sensor board sits behind the optical center and brings the module total to 6 g. The camera assembly reaches 106.1 mm, within the 110 mm design envelope. Thin outboard carbon posts keep the 58-degree view clear.

## Three gravity doors for KIT

KIT replaces the jaw and lift with three front chutes at y=51 mm. The 117 mm rack fits between the 125 mm side bounds. The hospital chute holds two cubes side by side in a 53 mm clear opening; each PCC chute holds one cube. All four upper faces remain visible. The hospital pair has 0.5 mm nominal outer clearance per side and needs a print tolerance check.

The 1 mm bottom doors have upper faces at z=42 mm. Each pivots around a hinge at y=35 mm and z=42 mm. Opening 90 degrees lowers its 32 mm tip to z=10 mm, clear of the floor. The cubes fall beyond the deck and wheels, whose forward edges stop at y=25 mm. Three gate servos replace the two gripper servos, giving KIT five motors and the fleet twenty-two. The camera moves to a rear mast so the bins remain visible.

## The disc is the difficult part

A 56 mm disc in a 60 mm hole has only 2 mm radial clearance. The lower jaw pads contact the disc at 3.4–4.8 mm above the floor, above the 3 mm plate. They grip its exposed upper edge. After the disc settles onto the hole floor, the jaws open sideways above the rim, then lift away. The fingers never enter the narrow annulus.

The nominal vertical gap is only 0.4 mm. At 105 mm tool reach, roughly 0.22 degrees of pitch consumes that gap. Measure plate thickness, wheel compression, jaw sag and floor flatness before choosing the physical pad height. Horizontal error must remain below 2 mm; target 1 mm combined placement error to leave useful margin. A vacuum head would avoid side contact but introduces wood leakage, pump and pressure-sensor requirements, so it is a separate prototype option.

## Files and verification

`best_design.json` owns fleet dimensions and roles. `scripts/build_best_design_blender.py` recreates seven named scenes through Blender MCP or Blender's Python entry point. `output/best_design/best_design.blend` preserves the incoming scene and contains the arena, starting fit, shared mechanism, disc section, extended LAB, KIT magazine and LAB support/camera section. The exported chassis STL is a fit mockup in millimeters. It has mounting holes but still needs material, tolerances and fastener design before fabrication.

`cad_geometry_audit.json` checks mesh bounds inside the start. MuJoCo uses simplified collision solids and force limits rather than every screw and cable in the CAD. Pad heights, wheel track/radius, start envelopes and LAB reach are the shared mechanical contract. Simulation results belong in their generated reports; the renders do not establish a 160-point physical run.

## Sources

- [KOSAC application and robot FAQ](https://apply.kosac.re.kr/onlnRcpt/getBscInfo.do?pbancNo=2026-S729), checked 11 September 2026.
- [National preliminary guide](../reference/challenge_2026_full.pdf), printed pages 10–13 and 17–19. The guide specifies the 120-second round, 160-point tasks, starting fit, permitted kit preload and wooden game pieces.
- [Pololu 5188 specifications](https://www.pololu.com/product/5188/specs), 32 × 12 × 10 mm motor body plus shaft; encoder extends outside that cross-section.
- [ROBOTIS XL330-M288 specifications](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/), 20 × 34 × 26 mm, recommended 5 V operation.
- [Raspberry Pi Zero 2 W](https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/), onboard compute reference.
