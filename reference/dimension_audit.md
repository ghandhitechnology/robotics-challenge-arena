# Archived initial dimension audit

Superseded by [the full-document audit](full_document_audit.md) and [the international audit](international_dimension_audit.md). The orientation, H, hole-diameter, beam and frame conclusions below describe the first build and are retained only as source-inspection history. Current values are in `arena_spec.json`.

# Independent dimension audit

Audit source: all four PDF pages, all twelve embedded JPEGs and their 3 × PNG enlargements, plus the attached block placement image. The original embedded JPEGs are low resolution. Enlargement helps read labels but adds no dimensional evidence. Measurements below use millimetres, consistent with the dimensioned construction drawings.

## Explicit source requirements

The Korean heading on PDF page 2 says the map is printed paper with insulating tape attached, and the preliminary round has no fence. Internal black boundaries are flat tape. The wooden perimeter visible in the reference photograph is therefore optional reference geometry, not part of the preliminary collision environment. The PDF says all blocks are wood.

| Item | Explicit dimensions | Evidence and interpretation |
| --- | --- | --- |
| Overall reference map | 1143 horizontal × 1181 vertical | PDF page 2 photograph. Arrow extension lines appear aligned with the outside of the wooden frame. Whether these dimensions include the frame is not stated in text. |
| Boundary tape | 20 wide; 0.15 thick | Width comes from PDF details A, B and C. The user explicitly supplied the 0.15 mm thickness after the source audit. Tape occupies Z 0..0.15 mm with static mesh collision. |
| Starting / recovery zone | 480 × 280 clear interior | Detail A. Both dimensions end at the interior edges of the 20 mm boundary tape. |
| Separation zone | 280 × 280 clear interior | Detail B. Both dimensions end at the interior edges of the 20 mm boundary tape. |
| Healthcare strip | 180 clear depth, 503 clear central span, 300 clear outer compartment | Detail C. 503 runs between the two inner tape edges, not between centerlines. 300 runs from the far tape edge to the board boundary. The mirrored left compartment is visually symmetric but has no separate 300 label. |
| Red, yellow and green blocks | Cylinder diameter 20, height 20 | PDF page 4 individual orthographic and isometric views. The square views show the cylinder from the side, not an additional cuboid type. |
| Medical kit | Marked face 25 × 25; depth 20 | The red cross has 20 × 20 overall extent and 5 mm strokes. The isometric picture shows a marked vertical face; the placement photograph appears to place this face upward, giving a 25 × 25 footprint and 20 height. |
| Sample | Circular disk diameter 56, thickness 5 | Readable in the enlarged original embedded image 019. Black disk with light biohazard emblem. |
| Laboratory plate | 345 × 150, thickness 3 | Three circular through holes are shown. Their diameters and center positions have no dimension labels. The map overview shows the plate lying horizontally at the board edge. |

The PDF does not give floor or frame height, tape thickness, paint thickness, RGB values, wood density, friction, restitution, object mass or bevel size. The user subsequently specified tape thickness as 0.15 mm; that value is authoritative for this model.

## Placement and orientation

The placement photograph is rotated 90 degrees counterclockwise relative to the dimensioned schematics. In photo coordinates the healthcare strip is on the left, the starting zone is at upper right, the separation zone at lower right and the laboratory lies between those right-side zones. In schematic coordinates those positions are top, bottom right, bottom left and bottom center respectively.

There are twelve colored cylinders, four per color. The photo shows two upper columns of yellow, green, red from top to bottom, and two lower columns of red, green, yellow from top to bottom. All cylinders stand vertically. The twelve black points in the schematic correspond to these cylinder positions.

There are exactly ten medical kits in two columns of five in the top-right photo zone, beside two elongated black rectangles. The close-up shows five red crosses in each column. The two black rectangles have light directional marks, different lengths, aligned upper ends and slightly rounded ends. They are absent from the dimensioned block list and the schematic diagrams; their dimensions, height and identity cannot be established from the document. They should be retained as photo-derived visual details with separate provenance, not described as dimensioned blocks.

Three sample disks lie flat in a triangular arrangement in the lower-right photo zone. Their light biohazard marks face upward. The laboratory has three equally spaced holes in a vertical row in photo coordinates and a horizontal row in schematic coordinates.

## Every embedded image checked

| Image | Native pixels | Content and findings |
| --- | --- | --- |
| 000 | 640 × 315 | PDF page 1, schematic and detail D. Three genuine laboratory openings. Central I shape in schematic orientation. Twelve cylinder markers. No additional dimensions. Schematic crop is roughly 259 × 249 pixels and is stretched differently from later overviews. |
| 001 | 412 × 508 | PDF page 2, paired arena perspective and dimensioned single arena photograph. The paired perspective is two repeated fields; the requested attached image is one field. Overall 1143 × 1181 is readable. Ten kits, twelve cylinders and three disks per field. Perimeter timber and two black starting-zone rectangles visible. |
| 003 | 522 × 261 | PDF page 2, detail A. Clear 480 × 280 zone; both tape widths 20. The enlarged horizontal 480 line begins at the inside edge of the vertical tape. The 280 starts at the inside edge of the horizontal tape. |
| 005 | 643 × 233 | PDF page 3, detail C. Clear 503 between inner edges, outer clear compartment 300, clear depth 180 and tape width 20. Vertical dashed line indicates symmetry of central compartment. The I crossbar visually aligns with the inner edges of this 503 gap. |
| 007 | 529 × 271 | PDF page 3, detail B. Clear 280 × 280 with two separate 20 tape-width labels. Laboratory appears adjacent to the far side of the vertical tape and flush with the outside board edge. |
| 009 | 551 × 272 | PDF page 3, repeated detail D. Same three-hole laboratory layout as 000. No hole diameter or center dimensions appear even in the enlargement. |
| 011 | 209 × 145 | PDF page 4 red cylinder. Height 20 and diameter 20. Three views describe one cylindrical shape. |
| 013 | 206 × 151 | PDF page 4 yellow cylinder. Height 20 and diameter 20. |
| 015 | 207 × 158 | PDF page 4 green cylinder. Height 20 and diameter 20. |
| 017 | 356 × 217 | PDF page 4 kit. Face 25 × 25, depth 20, red cross overall 20 × 20 with 5 strokes. Cross is centered; its implied face margin is 2.5. |
| 019 | 283 × 204 | PDF page 4 sample. Diameter label reads 56, thickness 5. Biohazard emblem is visible; exact emblem dimensions are not given. |
| 021 | 351 × 211 | PDF page 4 lab. Width 345, depth 150, thickness 3; three actual circular cutouts shown in front and isometric views. No hidden hole dimensions in the side view. |

## Scale-chain reconciliation

The healthcare chain is 300 + 20 + 503 + 20 + 300 = 1143. This runs horizontally in the schematic, which becomes vertical after rotating to match the placement photograph. The photograph instead labels this direction 1181. Three possible interpretations were checked against the dimensioned healthcare depth 180 and bottom-zone depth 280.

| Interpretation | Active floor in photo orientation | Schematic healthcare depth implied by pixels | Schematic bottom depth implied by pixels | Assessment |
| --- | --- | --- | --- | --- |
| Infer 19 mm rim from overall photograph | 1105 × 1143 | About 168–172 | About 253–262 | Fits the long dimension algebraically but fails independent local depth checks. Do not prefer this solely because 1181−1143=38. |
| Use photograph labels literally as active floor | 1143 × 1181 | About 174–178 | About 262–271 | Closer depth match but the healthcare horizontal chain becomes 1181 rather than its labeled 1143. |
| Use schematic dimensions and rotate to photograph | 1181 × 1143 | About 180–184 | About 271–280 | Best simultaneous fit to dimensioned schematic zones. Requires treating photograph width/height labels as transposed relative to its rotated contents. |

These estimates use the full schematic boundary as reference and measure clear depths to the relevant tape edges. Raster blur and line width produce several millimetres of variation. Source 005 is particularly informative: map top~16.7, bottom~211.3 native pixels; healthcare clear depth~29.6 pixels gives 179.6 mm for 1181 height; bottom clear~46 pixels gives 279.2 mm. With 1105 height these instead become 168.1 and 261.2 mm. This independently favors schematic height 1181.

The opposite horizontal schematic chain gives 480 + 20 + 345 + 20 + 280 = 1145, exceeding the healthcare chain by 2. Preserving the dimensioned 480 and 280 clear zones leaves a 343 gap between tape edges. A 345 plate centered in that gap has 1 mm of footprint overlap with each tape edge. The model preserves that overlap; it does not claim a particular physical join or that the plate lies beneath the tape. Neither overlap nor a 343 opening is labeled. Do not shrink the dimensioned 345 plate to silently force the chain.

The overview schematic images have different raster aspect ratios. For example the map in 000 is approximately 259 × 249 pixels, while the map in 003 is approximately 148 × 153. Register each schematic independently in x and y. Detail drawings A, B and C should control the labeled clear zones and tape widths. The 19 mm frame hypothesis remains possible only if one assumes the repeated local-depth drawings are also materially inaccurate; it is not the best source-supported reconstruction.

## Useful inferred geometry

The I crossbar endpoints in detail C align closely with the two inner tape edges bounding the labeled 503 span. A 503 mm crossbar is therefore a strong image-derived choice, though the H/I itself has no length label. After registering the overview independently to a schematic floor of 1143 × 1181, its two crossbar centerlines are approximately 420 mm apart. With 20 mm tape this gives a 400 mm clear gap and 440 mm outside span. Native source 000 centerlines near 90 and 179 over 249 pixels imply 422 mm; source 005 near 72.7 and 142 over 194.6 pixels imply 420.5 mm. This inference is consistent across repeated diagrams once their different raster distortions are removed.

The laboratory front view supports approximately 100 mm hole pitch and center locations 72.5, 172.5 and 272.5 along the 345 mm length, centered across the 150 mm depth. Hole diameter is visually around 60 mm. The sample diameter of 56 makes 60 a plausible clearance hole. These are inferred values; the low-resolution outlines do not establish a dimensionally exact 60 mm hole.

Use the placement image to determine color order, kit and sample arrangement and photo-only markings. Use dimensioned details to set sizes. Derive undimensioned centers by registering the relevant image to the selected floor corners and retain both source pixel coordinates and inferred millimetre coordinates in a manifest. The source image has perspective and slight skew, so a four-corner projective transform is preferable to an unqualified uniform scale.

## Verification boundaries

An exact match can be certified for labeled dimensions, object counts, color order, zone topology and explicit heights. Unlabeled coordinates, plate hole geometry, rim dimensions, black rectangle geometry and material physics remain reconstruction choices even after exhaustive source inspection. Export verification should confirm those choices are preserved, while reporting them separately from dimensions printed in the PDF.

## Selected reconstruction convention

Use the dimensioned schematic as a 1143 wide × 1181 deep field, then rotate it into the attached layout. The resulting Blender/photo convention is X 1181 ×Y 1143. This preserves the explicit dimension pair, the 503 healthcare chain and the 180/280 depths, while acknowledging that the photograph's dimension labels have the opposite orientation. The schematics and photograph cannot all be treated as dimensionally consistent images.

For an image-coordinate convention with X increasing right and Y increasing downward, the central H has inferred vertical tape centerlines X 340 andX 760, each spanning Y 320..823. Its horizontal connector is centered at Y 571.5. With 20 mm tape, the vertical bars have 400 clear separation and 440 outside span. These positions come from raster registration and symmetry, not additional printed dimensions.

The colored-cylinder centers are strongly supported by the following snapped coordinates:

| Photo X | Photo Y | Color |
| --- | --- | --- |
| 450 and 650 | 150 | Yellow |
| 450 and 650 | 250 | Green |
| 450 and 650 | 350 | Red |
| 450 and 650 | 793 | Red |
| 450 and 650 | 893 | Green |
| 450 and 650 | 993 | Yellow |

Native 005 source boundary is approximately x 44.3..232.3 and y 16.7..211.3. Its marker rows near y 91 and 124 map to photo X 451 and 651. Its columns map to photo Y 150,253,353,793,894,995. Native 000 independently supports the same pattern within its blurred pixel edges. The selected values preserve the 100 mm within-group spacing and symmetry about Y 571.5. The table above uses downward source Y. The delivered Blender model uses upward Y, with Y_blender = 1143 − Y_source. For example, the top yellow source row at Y 150 becomes Blender Y 993, and the upper red source row at Y 350 becomes Blender Y 793. The symmetric color arrangement retains the same row list after reversing its order.

An optional wooden reference frame may be modeled outside the active floor with inferred 19 thickness and 50 height, but neither dimension is printed in this PDF. The preliminary arena should keep that frame disabled. The selected 345 lab plate overlapping each adjacent tape edge by 1 mm remains an inference used to resolve the 2 mm source-chain discrepancy.

## Comparison artifact

`scripts/compare_reference.py` produces `output/reference_comparison.png` and its JSON measurement record. The comparison registers the original 408 × 406 attachment using approximate inner-floor corners TL27,37; TR370,38; BR371,369; BL26,369. It normalizes both comparison panels to the selected 1181 × 1143 aspect ratio without changing source images or model geometry. Twelve color-weighted source centroids differ from model centers by mean 7.40 mm, RMS 7.85 mm and maximum 11.55 mm under this calibration. Original pixels represent about 3.45 mm, and blur and corner selection limit those measurements. They describe raster agreement rather than surveyed physical error.
