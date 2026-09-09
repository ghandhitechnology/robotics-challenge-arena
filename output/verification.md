# Arena verification

The revised Blender meshes and USD assets passed 1,055 assertions. The field uses the full national guide's photo-oriented width 1143 × length 1181 mm. The linked official international rulebook supplies missing dimensions. Both sources and their precedence are documented in [the full-document audit](../reference/full_document_audit.md).

## Geometry measured

| Element | Required dimensions in mm | Result |
| --- | --- | --- |
| Playing surface | X 1143 × Y 1181, top at Z 0 | Passed |
| All four tape meshes | Width 20, thickness 0.15, bottom Z 0 | Passed using vertex bounds and edge rays |
| Healthcare clear compartments | Depth 180; bottom 338, middle 503, top 300 | Passed; 338 is derived |
| Start/recovery and isolation | 280×480 and 280×280 clear | Passed |
| Central H | 500-long bars, 400 clear gap, 20 tape; center Y 609.5 | Passed against international edge datums |
| Cylinders | Diameter 20, height 20; X 450/650 and 100 row pitch | Passed; opposite end offset is mirrored |
| Kits and crosses | 25×25×20; cross 20×20 with 5 strokes | Passed |
| Samples | Diameter 56, thickness 5 | Passed |
| Laboratory | 150×345×3; three 60-diameter holes | Passed; 100 pitch remains inferred |
| Plate placement | Bounds X 993..1143, Y 318..663 | Passed, 18 gap to each neighboring tape |
| Containment beams | 60×280×20 and 60×250×20 | Passed, including collision and parented emblems |
| Optional frame | Thickness 20, height 65 | Passed in Blender and USD; rail lengths adapted to one field |

Every physical mesh is watertight with positive volume. Movable bodies begin on the floor inside their assigned clear zones, without overlap. Exact triangle-mesh collision preserves the thin tape and laboratory openings. Painted crosses and biohazard graphics follow their bodies without separate collision. Frames and omitted objects are removed completely from the fence-free and senior Blender files.

The largest Blender-to-USD mesh-dimension difference is 0.000100583 mm, below the 0.0005 mm verification tolerance. [Per-object measurements](dimension_measurements.csv) and [all assertions](dimension_validation.json) retain the measured values.

## Inventories and USD validation

| Setup | Dynamic bodies | Static colliders | Total colliders |
| --- | --- | --- | --- |
| Photo arrangement | 27 | 6 | 33 |
| Framed photo arrangement | 27 | 10 | 37 |
| National senior preliminary | 19 | 6 | 25 |

The senior setup has twelve cylinders, four kits and three samples. The photo setups have ten kits and two containment beams. All six reusable assets and standalone wrappers pass the 26 OpenUSD validators available in Blender 5.2.1 with zero diagnostics. [USD results](usd_validation.json) retain the counts and validator details. Units are meters and kilograms, Z is up, and `/Arena` is the reusable default prim. Standalone wrappers use relative references. No external textures are required.

## Visual inspection and source limits

Top, overview, detail, framed and senior renders were inspected, followed by an independent visual review. Color order, object counts, beam heights, laboratory openings, frame height and zone arrangement agree with the selected dimensions.

The [photo comparison](reference_comparison.png) registers the source image independently to the corrected physical rectangle. Its twelve measured cylinder centroids differ from the dimensioned model by RMS 21.60 mm and maximum 24.22 mm. Most of that shift is horizontal: the international 450/650 mm placement datums differ from the distorted photo. The source rulebook explicitly says schematic illustrations may not be to scale; printed numbers control this model. This raster comparison does not measure export error or surveyed field accuracy.

Derived details include the 338 mm lower PCC, 18 mm plate gaps, mirrored cylinder end offset, lab-hole pitch, kit/sample/beam positions and emblem dimensions. The national 345 mm lab plate takes precedence over the international 440 mm value. Masses, friction and the 10 mm support beneath Z 0 remain adjustable assumptions.

Isaac Sim runtime has not been executed on this Mac. The [provided runtime test](../scripts/isaac_smoke_test.py) checks PhysX body creation and settling on an NVIDIA GPU machine. Load the senior scene with the actual robot to check contact with the 0.15 mm tape and sample placement in the lab holes.
