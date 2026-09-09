"""Measure saved Blender and USD geometry against the source dimensions.

Run with Blender --background --factory-startup --python this_file.
Checks real vertex bounds and topology, rather than trusting object metadata.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from pxr import Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
sys.path.insert(0, str(ROOT / "scripts"))
from usd_physics import verify_usd, _resolve_objects, _own_geometry

TOL = 0.0000005  # 0.0005 mm; below the source drawing precision.
CHECKS = []

# Independent source expectations. Do not import arena_spec.json here: the
# national drawing labels and clear-zone arithmetic must catch spec mistakes.
FLOOR_X_M, FLOOR_Y_M = 1.143, 1.181
ZONE_BOUNDS = {
    "Zone_PCC_Lower": [0.0, 0.0, 0.180, 0.338],
    "Zone_H": [0.0, 0.358, 0.180, 0.861],
    "Zone_PCC_Upper": [0.0, 0.881, 0.180, 1.181],
    "Zone_Start_Recovery": [0.863, 0.701, 1.143, 1.181],
    "Zone_Isolation": [0.863, 0.0, 1.143, 0.280],
    "Zone_Laboratory": [0.993, 0.318, 1.143, 0.663],
}


def center_xy(obj):
    low, high = bounds(obj)
    return tuple(round((a + b) / 2, 5) for a, b in zip(low[:2], high[:2]))


def check(name, ok, measured=None, expected=None):
    CHECKS.append({"name": name, "passed": bool(ok), "measured": measured, "expected": expected})
    if not ok:
        raise AssertionError(f"{name}: measured {measured!r}, expected {expected!r}")


def close(name, measured, expected, tol=TOL):
    if isinstance(measured, (float, int)):
        check(name, abs(measured - expected) <= tol, measured, expected)
    else:
        m, e = list(measured), list(expected)
        check(name, len(m) == len(e) and all(abs(a - b) <= tol for a, b in zip(m, e)), m, e)


def bounds(obj):
    points = [obj.matrix_world @ v.co for v in obj.data.vertices]
    return ([min(p[i] for p in points) for i in range(3)],
            [max(p[i] for p in points) for i in range(3)])


def dimensions(obj):
    low, high = bounds(obj)
    return [b - a for a, b in zip(low, high)]


def topology(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    manifold = all(e.is_manifold for e in bm.edges)
    volume = bm.calc_volume(signed=True)
    bm.free()
    return manifold, volume


def tree(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    result = BVHTree.FromBMesh(bm)
    bm.free()
    return result


def top_hit(bvh, x, y):
    return bvh.ray_cast(Vector((x, y, 0.1)), Vector((0, 0, -1)), 0.2)[0]


def usd_bounds(stage, prims):
    cache = UsdGeom.XformCache()
    points = []
    for prim in prims:
        if prim.IsA(UsdGeom.Mesh):
            matrix = cache.GetLocalToWorldTransform(prim)
            points.extend(matrix.Transform(p) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get())
    return ([min(p[i] for p in points) for i in range(3)],
            [max(p[i] for p in points) for i in range(3)])


def main():
    bpy.ops.wm.open_mainfile(filepath=str(OUT / "challenge_arena.blend"))
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    manifest = json.loads((OUT / "arena_manifest.json").read_text())
    records = manifest["objects"]
    check("Manifest names are unique", len({r["name"] for r in records}) == len(records))
    check("Default Blender scene has no frame objects", not any(o.name.startswith("Frame_") for o in bpy.data.objects))
    close("Scene uses meter coordinates", bpy.context.scene.unit_settings.scale_length, 1.0)
    body_counts = Counter(r["body_type"] for r in records)
    check("27 photo-layout movable bodies", body_counts["dynamic"] == 27, body_counts["dynamic"], 27)
    check("6 static bodies", body_counts["static"] == 6, body_counts["static"], 6)
    check("Six named zone bounds", len(manifest["zones"]) == 6)
    close("Floor national drawing X1143 Y1181", dimensions(bpy.data.objects["Playing_Surface"])[:2], [FLOOR_X_M, FLOOR_Y_M])
    close("Floor lower-left origin", bounds(bpy.data.objects["Playing_Surface"])[0][:2], [0, 0])
    close("Floor top is Z0", bounds(bpy.data.objects["Playing_Surface"])[1][2], 0)

    cylinder_positions = {(x, y) for x in (0.450, 0.650)
                          for y in (0.150, 0.250, 0.350, 0.831, 0.931, 1.031)}
    kit_positions = {(x, y) for x in (0.907, 0.947)
                     for y in (0.918, 0.963, 1.008, 1.053, 1.098)}
    sample_positions = {(0.972, 0.145), (1.072, 0.195), (1.072, 0.095)}
    found_cylinders = set()
    found_kits, found_samples = set(), set()
    beam_dimensions = []
    color_counts = Counter()
    for rec in records:
        obj = bpy.data.objects[rec["name"]]
        if obj.type != "MESH":
            continue
        dims = dimensions(obj)
        if rec["body_type"] in {"static", "dynamic"}:
            manifold, volume = topology(obj)
            check(obj.name + " watertight", manifold)
            check(obj.name + " positive volume", volume > 0, volume)
            check(obj.name + " Blender physics exists", obj.rigid_body is not None)
        if rec["body_type"] == "dynamic":
            low, high = bounds(obj)
            close(obj.name + " rests on floor", low[2], 0)
            check(obj.name + " stays inside playing surface", low[0] >= -TOL and low[1] >= -TOL
                  and high[0] <= FLOOR_X_M + TOL and high[1] <= FLOOR_Y_M + TOL)
        if rec["category"] in {"patient", "observation", "low_risk"}:
            close(obj.name + " PDF diameter/height", dims, [0.020, 0.020, 0.020])
            for vertex in obj.data.vertices:
                if abs(math.hypot(vertex.co.x, vertex.co.y) - 0.010) > TOL:
                    raise AssertionError(f"Noncylindrical vertex in {obj.name}")
            found_cylinders.add(center_xy(obj))
            color_counts[rec["category"]] += 1
        elif rec["category"] == "medical_kit":
            close(obj.name + " PDF face and placed height", dims, [0.025, 0.025, 0.020])
            found_kits.add(center_xy(obj))
        elif rec["category"] == "sample":
            close(obj.name + " PDF diameter/thickness", dims, [0.056, 0.056, 0.005])
            found_samples.add(center_xy(obj))
        elif rec["category"] == "containment_beam":
            close(obj.name + " official beam width and height", [dims[0], dims[2]], [0.060, 0.020])
            beam_dimensions.append(tuple(round(v, 6) for v in dims))
            check(obj.name + " movable containment beam", rec["body_type"] == "dynamic"
                  and obj.rigid_body.type == "ACTIVE")
        elif rec["role"] == "cross":
            close(obj.name + " PDF cross span", dims[:2], [0.020, 0.020])
            coords = {round(abs(float(v.co.x)), 7) for v in obj.data.vertices}
            check(obj.name + " PDF 5 mm stroke", coords == {0.0025, 0.0100}, sorted(coords))
            check(obj.name + " follows kit body", obj.parent is not None and obj.parent.rigid_body is not None)
        elif rec["role"] == "tape":
            close(obj.name + " user-specified 0.15 mm thickness", dims[2], 0.00015)
            close(obj.name + " tape sits on floor", bounds(obj)[0][2], 0)
            check(obj.name + " passive exact mesh collider", obj.rigid_body.type == "PASSIVE" and obj.rigid_body.collision_shape == "MESH")
    check("Cylinder placement pattern", found_cylinders == cylinder_positions, sorted(found_cylinders), sorted(cylinder_positions))
    check("Kit centers retain starting-zone offsets", found_kits == kit_positions, sorted(found_kits), sorted(kit_positions))
    check("Sample centers retain isolation-zone offsets", found_samples == sample_positions, sorted(found_samples), sorted(sample_positions))
    check("Two official containment beam sizes", sorted(beam_dimensions) == [(0.060, 0.250, 0.020), (0.060, 0.280, 0.020)],
          sorted(beam_dimensions), [(0.060, 0.250, 0.020), (0.060, 0.280, 0.020)])
    actual_dynamic = {o.name for o in bpy.data.objects if o.rigid_body and o.rigid_body.type == "ACTIVE"}
    check("Blender active bodies match photo manifest", actual_dynamic == {r["name"] for r in records if r["body_type"] == "dynamic"})
    check("Four cylinders per category", dict(color_counts) == {"patient": 4, "observation": 4, "low_risk": 4}, dict(color_counts))
    expected_rows = {0.150: "observation", 0.250: "low_risk", 0.350: "patient",
                     0.831: "patient", 0.931: "low_risk", 1.031: "observation"}
    for r in records:
        if r["category"] in color_counts:
            actual_y = center_xy(bpy.data.objects[r["name"]])[1]
            check(r["name"] + " reference color order", r["category"] == expected_rows.get(actual_y))

    # Shoot real mesh rays on both sides of each specified 20 mm strip.
    probes = {
        "Tape_Healthcare_PCC_H_PCC": [(0.190, 0.5, "x"), (0.090, 0.348, "y"), (0.09, 0.871, "y")],
        "Tape_Start_Recovery": [(0.853, 0.95, "x"), (1.04, 0.691, "y")],
        "Tape_Isolation": [(0.853, 0.12, "x"), (1.04, 0.290, "y")],
        "Tape_Center_H": [(0.340, 0.450, "x"), (0.760, 0.450, "x"), (0.550, 0.6095, "y")],
    }
    for name, sites in probes.items():
        bvh = tree(bpy.data.objects[name])
        for idx, (cx, cy, axis) in enumerate(sites):
            hit = top_hit(bvh, cx, cy)
            check(f"{name} strip {idx} exists", hit is not None)
            close(f"{name} strip {idx} physical height", hit.z, 0.00015)
            for sign in (-1, 1):
                inside = (cx + sign * 0.0099, cy) if axis == "x" else (cx, cy + sign * 0.0099)
                outside = (cx + sign * 0.0101, cy) if axis == "x" else (cx, cy + sign * 0.0101)
                check(f"{name} strip {idx} 20 mm inside {sign}", top_hit(bvh, *inside) is not None)
                check(f"{name} strip {idx} 20 mm outside {sign}", top_hit(bvh, *outside) is None)
    check("Named zones match drawing", {z["name"] for z in manifest["zones"]} == set(ZONE_BOUNDS))
    for zone in manifest["zones"]:
        x0, y0, x1, y1 = bpy.data.objects[zone["name"]]["bounds_xy_m"]
        expected = ZONE_BOUNDS[zone["name"]]
        close(zone["name"] + " drawing clear bounds", [x0, y0, x1, y1], expected)
        close(zone["name"] + " clear dimensions", [x1 - x0, y1 - y0],
              [expected[2] - expected[0], expected[3] - expected[1]])
    close("Healthcare national clear-chain closure", 0.338 + 0.020 + 0.503 + 0.020 + 0.300, FLOOR_Y_M)
    tape_bounds = {
        "Tape_Healthcare_PCC_H_PCC": ([0, 0, 0], [0.200, 1.181, 0.00015]),
        "Tape_Start_Recovery": ([0.843, 0.681, 0], [1.143, 1.181, 0.00015]),
        "Tape_Isolation": ([0.843, 0, 0], [1.143, 0.300, 0.00015]),
    }
    for name, (expected_low, expected_high) in tape_bounds.items():
        low, high = bounds(bpy.data.objects[name])
        close(name + " source minimum", low, expected_low)
        close(name + " source maximum", high, expected_high)
    h_low, h_high = bounds(bpy.data.objects["Tape_Center_H"])
    close("Center H official outer minimum", h_low[:2], [0.330, 0.3595])
    close("Center H official outer maximum", h_high[:2], [0.770, 0.8595])
    close("Center H official 500 length and 400 clear gap", [h_high[0] - h_low[0] - 0.040, h_high[1] - h_low[1]], [0.400, 0.500])

    lab = bpy.data.objects["Laboratory_Plate"]
    close("Lab PDF 150 x 345 x 3", dimensions(lab), [0.15, 0.345, 0.003])
    lab_low, lab_high = bounds(lab)
    close("Lab minimum independent gap placement", lab_low, [0.993, 0.318, 0])
    close("Lab maximum independent gap placement", lab_high, [1.143, 0.663, 0.003])
    close("Lab center", center_xy(lab), [1.068, 0.4905])
    isolation_high = bounds(bpy.data.objects["Tape_Isolation"])[1][1]
    start_low = bounds(bpy.data.objects["Tape_Start_Recovery"])[0][1]
    close("Lab gap above isolation tape", lab_low[1] - isolation_high, 0.018)
    close("Lab gap below start tape", start_low - lab_high[1], 0.018)
    lab_tree = tree(lab)
    for idx, cy in enumerate((0.3905, 0.4905, 0.5905), 1):
        for dx in (0, -0.0299, 0.0299):
            check(f"Lab hole {idx} is open at X offset {dx}", top_hit(lab_tree, 1.068 + dx, cy) is None)
        for dx in (-0.0301, 0.0301):
            hit = top_hit(lab_tree, 1.068 + dx, cy)
            check(f"Lab hole {idx} has boundary at X offset {dx}", hit is not None)
            close(f"Lab hole {idx} surrounding top", hit.z, 0.003)
    close("Sample-to-hole diametral clearance", lab["hole_diameter_m"] - 0.056, 0.004)
    for rec in records:
        if rec["category"] not in {"medical_kit", "sample", "containment_beam"}:
            continue
        low, high = bounds(bpy.data.objects[rec["name"]])
        zone_name = "Zone_Isolation" if rec["category"] == "sample" else "Zone_Start_Recovery"
        x0, y0, x1, y1 = ZONE_BOUNDS[zone_name]
        check(rec["name"] + " stays inside clear zone", low[0] >= x0 - TOL and low[1] >= y0 - TOL
              and high[0] <= x1 + TOL and high[1] <= y1 + TOL)

    # Pairwise horizontal overlap would expose an erroneous initial placement.
    movable = [bpy.data.objects[r["name"]] for r in records if r["body_type"] == "dynamic"]
    for i, obj in enumerate(movable):
        a0, a1 = bounds(obj)
        for other in movable[i + 1:]:
            b0, b1 = bounds(other)
            overlap = all(min(a1[k], b1[k]) - max(a0[k], b0[k]) > TOL for k in (0, 1))
            check(f"Initial separation {obj.name} / {other.name}", not overlap)

    report = verify_usd(OUT / "challenge_arena.usdc", records)
    stage = Usd.Stage.Open(str(OUT / "challenge_arena.usdc"))
    prims = _resolve_objects(stage, records)
    paths = {p.GetPath() for p in prims.values()}
    measurements = []
    main_geometry = {}
    for rec in records:
        obj = bpy.data.objects[rec["name"]]
        if obj.type != "MESH":
            continue
        b0, b1 = bounds(obj)
        main_geometry[obj.name] = (b0, b1)
        u0, u1 = usd_bounds(stage, _own_geometry(prims[obj.name], paths))
        close(obj.name + " USD minimum matches Blender vertices", u0, b0)
        close(obj.name + " USD maximum matches Blender vertices", u1, b1)
        dims = [(b - a) * 1000 for a, b in zip(b0, b1)]
        usd_dims = [(b - a) * 1000 for a, b in zip(u0, u1)]
        measurements.append({"name": obj.name, "role": rec["role"], "source": rec["source"],
                             "blender_dimensions_mm": dims, "usd_dimensions_mm": usd_dims,
                             "export_error_mm": max(abs(a - b) for a, b in zip(dims, usd_dims))})
    check("Photo USD 27 rigid bodies", report["rigid_bodies"] == 27, report["rigid_bodies"], 27)
    check("Photo USD 33 colliders", report["colliders"] == 33, report["colliders"], 33)
    verify_usd(OUT / "challenge_arena_scene.usda", standalone=True)
    framed = verify_usd(OUT / "challenge_arena_framed.usdc", records + manifest["optional_frame_objects"])
    check("Framed USD preserves 27 dynamic bodies", framed["rigid_bodies"] == 27)
    check("Framed USD adds exactly 4 static colliders", framed["colliders"] == 37, framed["colliders"], 37)
    verify_usd(OUT / "challenge_arena_framed_scene.usda", standalone=True)
    framed_records = records + manifest["optional_frame_objects"]
    framed_stage = Usd.Stage.Open(str(OUT / "challenge_arena_framed.usdc"))
    framed_prims = _resolve_objects(framed_stage, framed_records)
    framed_paths = {prim.GetPath() for prim in framed_prims.values()}
    bpy.ops.wm.open_mainfile(filepath=str(OUT / "challenge_arena_framed.blend"))
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    for rec in manifest["optional_frame_objects"]:
        obj = bpy.data.objects[rec["name"]]
        expected = [0.020, 1.221, 0.065] if obj.name in {"Frame_Left", "Frame_Right"} else [1.143, 0.020, 0.065]
        close(obj.name + " international wall section and adapted length", dimensions(obj), expected)
        low, high = bounds(obj)
        close(obj.name + " bottom at floor datum", low[2], 0)
        check(obj.name + " static body", obj.rigid_body.type == "PASSIVE")
        usd_low, usd_high = usd_bounds(framed_stage, _own_geometry(framed_prims[obj.name], framed_paths))
        close(obj.name + " USD minimum matches Blender", usd_low, low)
        close(obj.name + " USD maximum matches Blender", usd_high, high)

    # The senior preliminary file follows the national inventory, separately
    # from the requested photograph's ten-kit/two-beam arrangement.
    senior_records = manifest["senior_objects"]
    senior_categories = Counter(r["category"] for r in senior_records if r["body_type"] == "dynamic")
    check("Senior preliminary body inventory", dict(senior_categories) == {
        "patient": 4, "observation": 4, "low_risk": 4, "medical_kit": 4, "sample": 3}, dict(senior_categories))
    check("Senior excludes beam and fence records", all(r["category"] != "containment_beam"
          and r["role"] != "fence" for r in senior_records))
    senior = verify_usd(OUT / "challenge_arena_senior.usdc", senior_records)
    check("Senior USD 19 rigid bodies", senior["rigid_bodies"] == 19, senior["rigid_bodies"], 19)
    check("Senior USD 25 colliders", senior["colliders"] == 25, senior["colliders"], 25)
    verify_usd(OUT / "challenge_arena_senior_scene.usda", standalone=True)
    senior_stage = Usd.Stage.Open(str(OUT / "challenge_arena_senior.usdc"))
    senior_prims = _resolve_objects(senior_stage, senior_records)
    senior_paths = {p.GetPath() for p in senior_prims.values()}
    bpy.ops.wm.open_mainfile(filepath=str(OUT / "challenge_arena_senior.blend"))
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    senior_dynamic = {o.name for o in bpy.data.objects if o.rigid_body and o.rigid_body.type == "ACTIVE"}
    check("Senior Blender has exactly its 19 active bodies", senior_dynamic == {
        r["name"] for r in senior_records if r["body_type"] == "dynamic"})
    removed_names = {r["name"] for r in records} - {r["name"] for r in senior_records}
    check("Senior Blender removes omitted photo objects", not removed_names.intersection(bpy.data.objects.keys()))
    check("Senior Blender excludes outer frame", not any(o.name.startswith("Frame_") for o in bpy.data.objects))
    for rec in senior_records:
        obj = bpy.data.objects[rec["name"]]
        if obj.type != "MESH":
            continue
        low, high = bounds(obj)
        expected_low, expected_high = main_geometry[obj.name]
        close(obj.name + " senior geometry minimum unchanged", low, expected_low)
        close(obj.name + " senior geometry maximum unchanged", high, expected_high)
        usd_low, usd_high = usd_bounds(senior_stage, _own_geometry(senior_prims[obj.name], senior_paths))
        close(obj.name + " senior USD minimum", usd_low, low)
        close(obj.name + " senior USD maximum", usd_high, high)

    result = {"status": "passed", "checks_passed": len(CHECKS), "tolerance_m": TOL,
              "source_exactness": "National map dimensions and clear zones use independent literal checks. Official international dimensions fill missing H, cylinder, hole, beam and frame measurements. Derived geometry and image-based placements are documented in arena_spec.json and reference/full_document_audit.md.",
              "isaac_runtime_validation": "not_run_on_mac", "checks": CHECKS,
              "geometry_measurements": measurements, "usd": report, "senior_usd": senior}
    (OUT / "dimension_validation.json").write_text(json.dumps(result, indent=2))
    with (OUT / "dimension_measurements.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["object", "role", "evidence", "blend_x_mm", "blend_y_mm", "blend_z_mm",
                         "usd_x_mm", "usd_y_mm", "usd_z_mm", "max_export_error_mm"])
        for row in measurements:
            writer.writerow([row["name"], row["role"], row["source"], *row["blender_dimensions_mm"],
                             *row["usd_dimensions_mm"], row["export_error_mm"]])
    print(json.dumps({"status": "passed", "checks": len(CHECKS), "meshes_measured": len(measurements),
                      "maximum_export_error_mm": max(m["export_error_mm"] for m in measurements)}, indent=2))


if __name__ == "__main__":
    main()
