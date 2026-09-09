# Full document audit

Primary supplied source: `reference/challenge_2026_full.pdf`, 20 PDF pages. Printed page numbers run from 1 through 19 after the cover. This audit covers the extracted text of every page, the relevant page renders, every new embedded image at native resolution and the enlarged map image, plus comparison with all previously inspected dimensional JPEGs.

Supplement: the official [2026–2027 international rulebook](https://robotics-2026.web.app/files/intl-rulebook-en.pdf), preserved as `reference/challenge_2026_international.pdf`. Its pages 7–10 and 14–16 provide additional physical dimensions. The supplied national rules take priority where the two documents differ.

## Main finding

This report supersedes the earlier axis-orientation conclusion in `reference/dimension_audit.md`. It records the final dimension-refinement decision for the full document.

The full national document repeats literal outer labels of 1143 mm horizontal and 1181 mm vertical on a fence-free map. Its construction JPEGs are unchanged. The official international supplement adds explicit H dimensions, cylinder offsets, containment-beam dimensions, 60 mm laboratory slots and wall sections. Use the national outer and zone dimensions, then fill national omissions from those supplemental dimensions. The international rulebook page 8 explicitly says its illustrations are schematic and the stated technical dimensions take precedence.

This supersedes the previous audit's claim that the local dimensions require a transposed outer rectangle. That conclusion gave too much weight to an unlabeled mirrored 300 mm compartment. All printed numbers can instead be preserved with an asymmetric lower PCC compartment.

## Source image identity

All twelve dimensional and asset JPEGs are byte-for-byte identical between the full PDF and the earlier four-page extraction. SHA-256 equality confirms that the earlier Quartz export did not change their JPEG contents or resolution. Alpha/mask representation differs, but it adds no dimensional evidence.

| Full extraction image | Earlier image | Native pixels | Result |
| --- | --- | --- | --- |
| 008 | 000 | 640 × 315 | Identical JPEG bytes |
| 009 | 001 | 412 × 508 | Identical JPEG bytes |
| 011 | 003 | 522 × 261 | Identical JPEG bytes |
| 013 | 005 | 643 × 233 | Identical JPEG bytes |
| 015 | 007 | 529 × 271 | Identical JPEG bytes |
| 017 | 009 | 551 × 272 | Identical JPEG bytes |
| 019 | 011 | 209 × 145 | Identical JPEG bytes |
| 021 | 013 | 206 × 151 | Identical JPEG bytes |
| 023 | 015 | 207 × 158 | Identical JPEG bytes |
| 025 | 017 | 356 × 217 | Identical JPEG bytes |
| 027 | 019 | 283 × 204 | Identical JPEG bytes |
| 029 | 021 | 351 × 211 | Identical JPEG bytes |

The four new images are the cover logo, a finals-mission illustration on printed page 5, an ambulance illustration on printed page 8 and a preliminary map on printed page 10. Only the two map illustrations affect this audit.

## New map and setup evidence

Printed page 10 shows a fence-free map without the two elongated black rectangles in the starting zone. It repeats 1143 horizontally and 1181 vertically. Its underlying floor image is roughly 301 × 290 native pixels, so the raster proportions still conflict with its labels; its appearance must not override explicit numeric dimensions.

That image visibly contains six medical crosses in two columns of three. The immediately adjacent inventory table explicitly supplies two kits for juniors and four for seniors. Printed page 11 independently confirms two and four through the scored kit counts. The six illustrated kits and ten in the original attached photograph therefore cannot determine a competition inventory. A senior competition configuration requires four kits, twelve cylinders and three samples with one laboratory plate. The main and framed deliverables retain ten kits because the user explicitly requested the attached placement. A separate senior preliminary asset uses the national four-kit inventory without beams or a frame. The two-kit junior count is documented as setup guidance.

The text on national printed page 5 describes finals isolation construction using barrier beams. The international supplement pages 9–10 now supplies their identity, dimensions and count: two wooden containment beams, one 250 × 60 × 20 mm and one 280 × 60 × 20 mm. Their appearance matches the two formerly unclassified black rectangles. They should become real movable bodies in the photo-layout assets, with the shorter beam on the right. The separate national senior preliminary asset omits them because they are absent from its inventory and map illustration.

Printed page 13 explicitly restates the starting area as 480 × 280 mm. It also permits medical kits to be loaded onto the robot before the run, provided the robot and load remain entirely within the starting area. Samples and cylinders must begin at their designated positions. The exact choice of four unladen kit slots is not dimensioned, and the six-versus-four discrepancy prevents treating every illustrated kit center as mandatory.

## Constraint solution with literal outer dimensions

Coordinate convention: Blender X points right, Y points up, with origin at the lower-left playing-surface corner. All values below are millimetres.

| Feature | Required construction | Evidence |
| --- | --- | --- |
| Playing surface | X 1143 × Y 1181 | Explicit outer labels on printed pages 5, 10 and 17 |
| Healthcare clear depth | X 0..180 | Explicit 180 in detail C |
| Main healthcare tape | X 180..200 | Explicit 20 tape width |
| Upper PCC clear length | Y 881..1181, length 300 | The right-hand compartment in schematic C becomes the upper compartment after rotation; its 300 label is explicit |
| Upper transverse tape | Y 861..881 | Explicit 20 tape width |
| Hospital clear length | Y 358..861, length 503 | Explicit inner-edge-to-inner-edge 503 |
| Lower transverse tape | Y 338..358 | Explicit 20 tape width |
| Lower PCC clear length | Y 0..338, length 338 | Remainder after preserving the explicit outer length and other labeled spans; not separately dimensioned |
| Start clear area | X 863..1143, Y 701..1181 | Explicit 280 × 480 clear dimensions |
| Start inward boundary tape | X 843..863; lower arm Y 681..701 | Explicit 20 tape width |
| Isolation clear area | X 863..1143, Y 0..280 | Explicit 280 × 280 clear dimensions |
| Isolation inward boundary tape | X 843..863; upper arm Y 280..300 | Explicit 20 tape width |
| Clear interval between right zones | Y 300..681, length 381 | Derived from outer height and the two dimensioned zones |
| Laboratory | X 993..1143, Y 318..663, Z 0..3 | Explicit 150 × 345 × 3 plate; centering inside the 381 interval is inferred and leaves 18 at each end |

The laboratory no longer needs the previous artificial 1 mm footprint overlap at each tape boundary. Its 345 mm physical length remains exact. The centered 18 mm gaps are inferred placement, not printed measurements.

The lower PCC asymmetry and laboratory gaps depart from the drawing's approximate symmetry and adjacency. This is an explicit-dimension-first reconstruction, not proof that every illustration is internally consistent. No interpretation can claim exact agreement with both the conflicting outer labels and every raster proportion.

## Supplementary dimensions and anchor transformation

The international rulebook page 16 removes the need to scale or estimate the central H and patient grid. Rotate its I-shaped schematic counterclockwise into the supplied photo layout. Preserve each printed datum relative to the edge its arrow identifies.

| Feature | Supplemental measurement | Result in the national photo-oriented coordinate system |
| --- | --- | --- |
| Near H bar | Outer edge 330 from the healthcare-side field edge; tape 20 | X 330..350, center 340 |
| Gap between H bars |400 between inner edges | Clear X 350..750; far bar X 750..770, center 760 |
| H bar length |500 | Both bars 500 long; overall H 440 × 500 |
| H connector datum |571.5 from schematic right edge to connector centerline | This becomes 571.5 down from the photo top, so Blender Y 1181−571.5=609.5 |
| H bar ends |250 on either side of connector | Y 359.5..859.5, leaving 1.5 inside each end of the national hospital interval Y 358..861 |
| Patient rows from healthcare-side edge |450, then 200 farther | Cylinder X 450 and X 650 |
| Within-group patient spacing |100 | Three positions per group,100 apart |
| Near-end patient offset |150 from the schematic left edge | Lower photo group at Blender Y 150,250,350 |
| Opposite group | Mirrored 150 edge offset and 100 spacing | Upper photo group at Blender Y 831,931,1031 |

The 150 offset is printed at the schematic left edge, while the 100 dimensions are printed on the opposite group. Applying the same 150 offset to the opposite field end uses the illustrated mirrored layout; it is not a second separately labeled 150 dimension. The schematic right-hand 571.5 H datum, however, is explicit and should not be replaced by half of the 1181 national dimension. This is why the H center is 609.5 rather than 590.5.

Keep the attached photo's color order: lower rows 150 yellow,250 green,350 red; upper rows 831 red,931 green,1031 yellow. Each row contains the two X positions 450 and 650. These replace both the earlier inferred 503 mm H length and the intermediate plan to scale its geometry to 519.72 mm.

Kit and sample centers still come from photographic registration because their coordinates are not dimensioned. Physical asset dimensions remain exact. The national laboratory remains 150 × 345 × 3 mm, centered at 1068,490.5 with Y 318..663. Its 60 mm hole diameter is now explicit in international section 3.2; its 100 mm pitch remains image-derived. The international 440 mm plate length conflicts with the supplied national 345 mm drawing and must not replace it.

## Optional frame and beam orientation

International page 7 specifies long outer walls with a 20 ±1 mm thickness and 65 ±2 mm height. Use the nominal 20 × 65 section for all four rails of the optional single-field reference frame. Adapt their lengths to enclose the national 1143 × 1181 surface. This is a single-field adaptation, not a literal reconstruction of the international two-field board.

International page 8 gives the shared central divider as 19 ±1 mm thick and 70 ±3 mm high. That divider is omitted because only one field was requested. The previous 19 × 50 mm assumed outer frame section is superseded.

For the attached photo's initial arrangement, lay each containment beam on its 60 mm face with 20 mm vertical height, so its marked face points upward. The documented upright containment configuration has 60 mm height and 20 mm thickness, and can be reached by rotating the same rigid body. The model should preserve the physical 250/280 × 60 × 20 dimensions in either pose.

## Measurements unchanged

The full PDF confirms the same 20 mm tape width, three cylinder colors with diameter 20 and height 20, kit marked face 25 × 25 and depth 20, red cross extent 20 with 5 mm strokes, sample diameter 56 and thickness 5, and laboratory dimensions 345 × 150 × 3. All blocks are wood. Preliminary maps have no fence. The user's tape-thickness instruction of 0.15 mm remains authoritative because neither PDF supplies a tape thickness.

Laboratory hole diameter 60 mm, containment-beam dimensions and nominal frame section now have explicit supplemental sources. Hole pitch, exact kit/sample centers, floor-support thickness, wood density, friction and restitution remain inferred or assumed. No source supplies those material-physics measurements.

## RL-relevant rules, without expanding the modeling task

The full document confirms a 120-second episode and evaluates object placement at the end of the run. Scored objects must lie wholly inside the destination boundary and be detached from the robot; objects overlapping the boundary do not score. Red goes to H, yellow to PCC and green to RZ. Senior kit allocation is two to H and one to each PCC, with three samples moved into laboratory circles. The field always begins with twelve cylinders even though only nine are needed for the senior cylinder scores. These rules support correct zone-bound metadata and inventory; implementing rewards is a separate task.

Printed page 10 says the map must be fixed flat on a firm floor. It specifies no raised platform or floor thickness. The existing below-Z 0 support slab remains an implementation assumption, not a newly confirmed dimension.

## Page coverage

| PDF page | Printed page | Inspection result |
| --- | --- | --- |
| 1 | Cover | Title, date and foundation logo; no geometry |
| 2 | 1 | Competition overview and eligibility; no geometry |
| 3 | 2 | Eligibility, awards and application information; no geometry |
| 4 | 3 | Preliminary/finals distinction and submission process |
| 5 | 4 | Preliminary tasks and selection process; no dimensions |
| 6 | 5 | New finals map illustration, repeated outer labels and barrier-beam task context |
| 7 | 6 | Finals procedure and timetable; no geometry |
| 8 | 7 | Submission and participation conditions; no geometry |
| 9 | 8 | Theme explanation and ambulance illustration; no geometry |
| 10 | 9 | Preliminary scope, 120-second duration and supplied wooden elements |
| 11 | 10 | New fence-free map, zone names and explicit junior/senior inventory |
| 12 | 11 | Explicit inventory through scoring counts; full-inside-boundary placement rule |
| 13 | 12 | Timing and end-state scoring; no new measurements |
| 14 | 13 | Explicit 480 × 280 starting area, permitted preloaded kits and fixed sample/cylinder starts |
| 15 | 14 | Judging, dates and contact details; no geometry |
| 16 | 15 | Repeated laboratory schematic, identical to old JPEG 000 |
| 17 | 16 | Submission checklist; no geometry |
| 18 | 17 | Outer map and detail A, identical construction images |
| 19 | 18 | Details C, B and D, identical construction images |
| 20 | 19 | All block and laboratory construction diagrams, identical images |

## Concrete changes supported

1. Set the active field to literal photo-oriented X 1143 ×Y 1181. Rebuild tape and zone bounds from the national explicit dimensions; derive the lower PCC as 338 mm and center the 345 mm laboratory within its 381 mm interval, leaving 18 mm at each end.
2. Use the international H 500 mm length,400 mm clear gap,330 mm near outer edge and 571.5 mm right-edge datum. This produces X centers 340/760, Y 359.5..859.5 and connectorY 609.5. Keep tape 20 mm wide and the user-specified 0.15 mm thick.
3. Use cylinder X 450/650 and 100 mm group spacing, with 150 mm end offsets as described above. Preserve all 20 × 20 mm cylinder sizes and the attached color order.
4. Replace the black reference rectangles with two physical 250/280 × 60 × 20 mm beams in the photo-layout assets. Keep their ten kits, producing 27 movable bodies. Add a separate senior preliminary variant with 12 cylinders,4 kits and 3 samples, totaling 19 movable bodies, without beams or frame.
5. Keep the national 345 × 150 × 3 mm laboratory, adopt explicit 60 mm slots, and retain 100 mm pitch as inferred. Replace the optional frame's assumed 19 × 50 section with nominal 20 × 65 mm, adapting its lengths to a single national field.
6. Preserve source conflicts and inferred placement details in provenance. The national diagram and international text use conflicting outer-axis conventions, and the national 345 laboratory overrides the international 440 version. Numeric source priority should be visible in the audit rather than hidden through image scaling.
