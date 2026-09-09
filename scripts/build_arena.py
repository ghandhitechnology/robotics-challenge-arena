"""Build the dimensioned challenge arena in Blender and export physics USD assets.

Run: blender --background --factory-startup --python scripts/build_arena.py
All authored geometry is in meters. Source dimensions and placement evidence
remain in arena_spec.json in millimeters.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from arena_assets import make_cylinder, make_kit, make_sample, make_lab_plate, make_containment_beam
from usd_physics import configure_usd

SPEC = json.loads((ROOT / "arena_spec.json").read_text())
D = SPEC["dimensions"]
P = SPEC["placements"]
MM = 0.001
OUT = ROOT / "output"
RECORDS = []


def srgb(c):
    return c / 12.92 if c < 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def material(name, hex_color, roughness=0.65):
    rgb = tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*rgb, 1)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*[srgb(v) for v in rgb], 1)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Specular IOR Level"].default_value = 0.15
    return mat


def collection(name):
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def record(obj, role, body_type="none", category=None, source="image_measured", mass=None):
    if role == "tape":
        # The user specifies 0.15 mm physical tape, seated on Z=0.
        obj.location.z = D["tape_thickness"] * MM / 2
        bpy.context.view_layer.objects.active = obj
        solid = obj.modifiers.new("Exact_015mm_Tape", "SOLIDIFY")
        solid.thickness = D["tape_thickness"] * MM
        solid.offset = 0
        bpy.ops.object.modifier_apply(modifier=solid.name)
        body_type = "static"
    obj["arena_source_name"] = obj.name
    obj["role"] = role
    obj["category"] = category or role
    obj["dimension_evidence"] = source
    bpy.context.view_layer.update()
    entry = {
        "name": obj.name, "role": role, "category": category or role,
        "body_type": body_type, "dimensions_m": list(obj.dimensions),
        "position_m": list(obj.matrix_world.translation), "source": source,
    }
    if obj.parent:
        entry["parent"] = obj.parent.name
    if mass is not None:
        entry.update(mass_kg=mass, mass_source="assumed_600_kg_m3_wood")
    if body_type == "dynamic":
        entry["collision"] = "convexHull"
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.ops.rigidbody.object_add()
        obj.rigid_body.mass = mass
        obj.rigid_body.collision_shape = "CONVEX_HULL"
        obj.rigid_body.use_margin = True
        obj.rigid_body.collision_margin = 0.0001
        obj.rigid_body.friction = SPEC["physics"]["dynamic_friction"]
        obj.rigid_body.restitution = 0.0
        obj.select_set(False)
    elif body_type == "static":
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.ops.rigidbody.object_add()
        obj.rigid_body.type = "PASSIVE"
        obj.rigid_body.collision_shape = "MESH" if role in {"lab_plate", "tape"} else "BOX"
        obj.rigid_body.use_margin = True
        obj.rigid_body.collision_margin = 0
        obj.rigid_body.friction = SPEC["physics"]["dynamic_friction"]
        obj.select_set(False)
    RECORDS.append(entry)
    return obj


def box(name, center_mm, size_mm, mat, col):
    bpy.ops.mesh.primitive_cube_add(size=1, location=[v * MM for v in center_mm])
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = [v * MM for v in size_mm]
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    for current in list(obj.users_collection):
        current.objects.unlink(obj)
    col.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def polygon(name, coordinates_mm, mat, col, z_mm=None):
    if z_mm is None:
        z_mm = D["tape_thickness"] / 2
    xs, ys = zip(*coordinates_mm)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    verts = [((x - cx) * MM, (y - cy) * MM, 0) for x, y in coordinates_mm]
    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata(verts, [], [list(range(len(verts)))])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    col.objects.link(obj)
    obj.location = (cx * MM, cy * MM, z_mm * MM)
    obj.data.materials.append(mat)
    return obj


def rectangle(name, x0, y0, x1, y1, mat, col, z_mm=None):
    return polygon(name, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], mat, col, z_mm)


def add_zone(name, category, bounds, col):
    obj = bpy.data.objects.new(name, None)
    col.objects.link(obj)
    x0, y0, x1, y1 = bounds
    obj.location = ((x0 + x1) * MM / 2, (y0 + y1) * MM / 2, 0)
    obj.empty_display_type = "PLAIN_AXES"
    obj.empty_display_size = 0.015
    obj.hide_render = True
    obj["bounds_xy_m"] = [v * MM for v in bounds]
    obj["zone_category"] = category
    obj["arena_source_name"] = name
    record(obj, "zone_region", category=category, source="dimensioned_zone_bounds")
    return {"name": name, "category": category, "bounds_xy_m": [v * MM for v in bounds]}


def camera(name, location, target, col, ortho=None):
    data = bpy.data.cameras.new(name)
    obj = bpy.data.objects.new(name, data)
    col.objects.link(obj)
    obj.location = location
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()
    data.clip_start = 0.001
    data.clip_end = 100
    if ortho:
        data.type = "ORTHO"
        data.ortho_scale = ortho
    else:
        data.lens = 55
    return obj


def light(name, location, power, size, target, col):
    data = bpy.data.lights.new(name, "AREA")
    data.energy, data.shape, data.size = power, "DISK", size
    obj = bpy.data.objects.new(name, data)
    col.objects.link(obj)
    obj.location = location
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def export_asset(name, records, wrapper=True):
    bpy.ops.object.select_all(action="DESELECT")
    for rec in records:
        obj = bpy.data.objects[rec["name"]]
        obj.hide_set(False)
        obj.hide_render = False
        obj.select_set(True)
    path = OUT / f"{name}.usdc"
    bpy.ops.wm.usd_export(
        filepath=str(path), selected_objects_only=True,
        root_prim_path="/Arena", export_materials=True, export_textures_mode="NEW",
        export_custom_properties=True, custom_properties_namespace="userProperties",
        export_animation=False, export_lights=False, export_cameras=False,
        export_armatures=False, export_hair=False, export_curves=False,
        convert_world_material=False, generate_preview_surface=True,
        generate_materialx_network=False, convert_scene_units="METERS", meters_per_unit=1.0,
        relative_paths=True, triangulate_meshes=True, export_subdivision="BEST_MATCH",
    )
    report = configure_usd(path, records, OUT / f"{name}_scene.usda" if wrapper else None)
    return report


def build():
    OUT.mkdir(exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for col in list(bpy.data.collections):
        if col.name not in {"RigidBodyWorld", "RigidBodyConstraints"} and col.users == 0:
            bpy.data.collections.remove(col)
    scene = bpy.context.scene
    scene.name = "Photo_Reference_Arena"
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.unit_settings.length_unit = "MILLIMETERS"
    scene["source_document"] = "reference/challenge_2026_full.pdf; supplemental reference/challenge_2026_international.pdf"
    scene["coordinate_system"] = SPEC["coordinate_system"]
    scene["dimension_interpretation"] = SPEC["evidence"]["source_conflict"]
    cols = {key: collection(name) for key, name in {
        "floor": "00_Playing_Surface", "tape": "10_Zone_Tape", "zones": "11_Zone_Bounds",
        "marks": "12_Placement_Marks", "cylinders": "20_Colored_Blocks",
        "kits": "30_Medical_Kits", "samples": "40_Samples",
        "lab": "50_Laboratory", "start": "60_Containment_Beams",
        "frame": "90_Optional_Reference_Frame", "studio": "99_Studio"}.items()}
    mats = {
        "floor": material("Paper_Grey", "d6d5d1", 0.9),
        "tape": material("Electrical_Tape_Black", "101110", 0.58),
        "red": material("Paint_Patient_Red", "b52c26", 0.59),
        "green": material("Paint_Low_Risk_Green", "438c3b", 0.62),
        "yellow": material("Paint_Observation_Yellow", "e5df37", 0.62),
        "ivory": material("Paint_Medical_Kit_Offwhite", "e6e2d4", 0.7),
        "white": material("White_Emblem", "eeeada", 0.65),
        "lab": material("Laboratory_Grey_Plate", "aaa89e", 0.78),
        "wood": material("Reference_Frame_Pine", "cda669", 0.66),
    }
    w, h, t = D["surface_x"], D["surface_y"], D["tape_width"]
    record(box("Playing_Surface", [w / 2, h / 2, -D["support_thickness"] / 2],
               [w, h, D["support_thickness"]], mats["floor"], cols["floor"]),
           "floor", "static", "playing_surface", "explicit_national_width1143_length1181")

    # Every black zone shape is a single flat polygon. Adjacent arms share edges.
    dep = D["healthcare_clear_depth"]
    a = h - D["healthcare_upper_clear_length"] - 2 * t - D["healthcare_middle_clear_length"]
    b = a + t
    c = b + D["healthcare_middle_clear_length"]
    d = c + t
    points = [(dep, 0), (dep + t, 0), (dep + t, h), (dep, h),
              (dep, d), (0, d), (0, c), (dep, c),
              (dep, b), (0, b), (0, a), (dep, a)]
    record(polygon("Tape_Healthcare_PCC_H_PCC", points, mats["tape"], cols["tape"]),
           "tape", category="healthcare", source="explicit_detail_C")
    x = w - D["start_clear_depth"]
    y = h - D["start_clear_length"]
    record(polygon("Tape_Start_Recovery", [(x - t, y - t), (w, y - t), (w, y),
                  (x, y), (x, h), (x - t, h)], mats["tape"], cols["tape"]),
           "tape", category="starting_recovery_zone", source="explicit_detail_A")
    z = D["isolation_clear_side"]
    record(polygon("Tape_Isolation", [(x - t, 0), (x, 0), (x, z), (w, z),
                  (w, z + t), (x - t, z + t)], mats["tape"], cols["tape"]),
           "tape", category="isolation", source="explicit_detail_B")
    left, right = D["center_bar_x"]
    lo, hi = D["center_bar_y_limits"]
    mid = D["center_crossbar_y"]
    q = t / 2
    points = [(left - q, lo), (left + q, lo), (left + q, mid - q),
              (right - q, mid - q), (right - q, lo), (right + q, lo),
              (right + q, hi), (right - q, hi), (right - q, mid + q),
              (left + q, mid + q), (left + q, hi), (left - q, hi)]
    record(polygon("Tape_Center_H", points, mats["tape"], cols["tape"]),
           "tape", category="center_divider", source="international_p16_dimensions_national_edge_datums")
    zones = [add_zone(name, kind, bounds, cols["zones"]) for name, kind, bounds in [
        ("Zone_PCC_Lower", "PCC", [0, 0, dep, a]),
        ("Zone_H", "H", [0, b, dep, c]),
        ("Zone_PCC_Upper", "PCC", [0, d, dep, h]),
        ("Zone_Start_Recovery", "starting_recovery", [x, y, w, h]),
        ("Zone_Isolation", "isolation", [x, 0, w, z]),
        ("Zone_Laboratory", "laboratory", [P["lab_center"][0] - D["lab_width"] / 2,
          P["lab_center"][1] - D["lab_length"] / 2,
          P["lab_center"][0] + D["lab_width"] / 2,
          P["lab_center"][1] + D["lab_length"] / 2]),
    ]]

    density = SPEC["physics"]["wood_density_kg_m3"]
    cyl_mass = density * math.pi * (D["cylinder_diameter"] * MM / 2) ** 2 * D["cylinder_height"] * MM
    for row_id, row in enumerate(P["cylinder_rows"], 1):
        for col_id, cx in enumerate(P["cylinder_x"], 1):
            name = f"Block_{row['color'].title()}_R{row_id}_C{col_id}"
            marker = make_cylinder("Mark_" + name, 0.020, 0.00002,
                                   (cx * MM, row["y"] * MM, 0.00001), mats["tape"], cols["marks"])
            record(marker, "marking", category="printed_block_position")
            obj = make_cylinder(name, D["cylinder_diameter"] * MM, D["cylinder_height"] * MM,
                                (cx * MM, row["y"] * MM, D["cylinder_height"] * MM / 2),
                                mats[row["color"]], cols["cylinders"])
            record(obj, "movable", "dynamic", row["category"], "explicit_size_image_placement", cyl_mass)
    kit_mass = density * (D["kit_marked_face"] * MM) ** 2 * D["kit_height_as_placed"] * MM
    for row_id, cy in enumerate(P["kit_y"], 1):
        for col_id, cx in enumerate(P["kit_x"], 1):
            obj, decorations = make_kit(f"Medical_Kit_R{row_id}_C{col_id}",
                                        (cx * MM, cy * MM, 0.010), mats["ivory"], mats["red"], cols["kits"])
            record(obj, "movable", "dynamic", "medical_kit", "explicit_size_image_placement", kit_mass)
            for child in decorations:
                record(child, "cross", category="medical_kit_cross", source="explicit_20mm_cross_5mm_stroke")
    sample_mass = density * math.pi * (D["sample_diameter"] * MM / 2) ** 2 * D["sample_thickness"] * MM
    for idx, (cx, cy) in enumerate(P["samples"], 1):
        obj, decorations = make_sample(f"Sample_{idx:02d}", (cx * MM, cy * MM, 0.0025),
                                       mats["tape"], mats["white"], cols["samples"])
        record(obj, "movable", "dynamic", "sample", "explicit_size_image_placement", sample_mass)
        for child in decorations:
            record(child, "biohazard", category="sample_biohazard")
    lab = make_lab_plate("Laboratory_Plate", (*[v * MM for v in P["lab_center"]], 0.0015),
                         mats["lab"], cols["lab"], D["lab_hole_diameter"] * MM, D["lab_hole_pitch"] * MM)
    record(lab, "lab_plate", "static", "laboratory", "national_outer_dimensions_international_60mm_holes_inferred_pitch")

    for item in P["containment_beams"]:
        cx, cy = item["center"]
        obj, decorations = make_containment_beam(item["name"],
            (cx * MM, cy * MM, D["beam_height"] * MM / 2), item["length"] * MM,
            mats["tape"], mats["white"], cols["start"])
        mass = density * D["beam_width"] * item["length"] * D["beam_height"] * MM ** 3
        record(obj, "movable", "dynamic", "containment_beam", "explicit_international_beam_size_photo_position", mass)
        for child in decorations:
            record(child, "biohazard", category="beam_biohazard")

    frame_records_start = len(RECORDS)
    ft, fh = D["frame_thickness"], D["frame_height"]
    for name, pos, size in [
        ("Frame_Left", [-ft / 2, h / 2, fh / 2], [ft, h + 2 * ft, fh]),
        ("Frame_Right", [w + ft / 2, h / 2, fh / 2], [ft, h + 2 * ft, fh]),
        ("Frame_Bottom", [w / 2, -ft / 2, fh / 2], [w, ft, fh]),
        ("Frame_Top", [w / 2, h + ft / 2, fh / 2], [w, ft, fh]),
    ]:
        record(box(name, pos, size, mats["wood"], cols["frame"]), "fence", "static",
               "optional_reference_frame", "international_20x65_wall_section_adapted_single_field_lengths")
    frame_records = RECORDS[frame_records_start:]
    main_records = RECORDS[:frame_records_start]
    senior_kit_rows = SPEC["configurations"]["senior_preliminary"]["kit_rows"]
    senior_kits = {f"Medical_Kit_R{row}_C{col}" for row in senior_kit_rows for col in (1, 2)}
    omitted = {rec["name"] for rec in main_records
               if rec["category"] == "containment_beam"
               or (rec["category"] == "medical_kit" and rec["name"] not in senior_kits)}
    senior_records = [rec for rec in main_records
                      if rec["name"] not in omitted and rec.get("parent") not in omitted]

    scene.rigidbody_world.substeps_per_frame = 10
    scene.rigidbody_world.solver_iterations = 30
    scene.rigidbody_world.point_cache.frame_start = 1
    scene.rigidbody_world.point_cache.frame_end = 250
    scene.frame_set(1)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 48
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1500
    scene.render.resolution_y = 1500
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = -1.5
    scene.world.use_nodes = True
    scene.world.node_tree.nodes.get("Background").inputs["Color"].default_value = (0.6, 0.6, 0.6, 1)
    scene.world.node_tree.nodes.get("Background").inputs["Strength"].default_value = 0.45
    center = (w * MM / 2, h * MM / 2, 0)
    top = camera("Camera_Top_Orthographic", (*center[:2], 2.2), center, cols["studio"], 1.34)
    hero = camera("Camera_Overview", (1.85, -1.75, 2.15), center, cols["studio"], 1.68)
    detail = camera("Camera_Assets", (1.65, 0.2, 0.75), (1.03, 0.7, 0), cols["studio"], 0.92)
    light("Key_Softbox", (-0.4, -0.3, 2.1), 110, 2.3, center, cols["studio"])
    light("Fill_Softbox", (1.6, 1.8, 1.4), 65, 1.6, center, cols["studio"])
    scene.camera = hero
    for area in bpy.context.screen.areas if bpy.context.screen else []:
        if area.type == "VIEW_3D":
            area.spaces.active.region_3d.view_distance = 1.85
            area.spaces.active.region_3d.view_location = center
            area.spaces.active.region_3d.view_rotation = hero.rotation_euler.to_quaternion()
            area.spaces.active.shading.type = "MATERIAL"
    manifest = {
        "schema_version": 2, "units": "meters", "up_axis": "Z",
        "coordinate_system": SPEC["coordinate_system"],
        "surface_dimensions_m": [w * MM, h * MM],
        "objects": main_records, "optional_frame_objects": frame_records, "zones": zones,
        "senior_objects": senior_records, "configurations": SPEC["configurations"], "sources": SPEC["sources"],
        "evidence": SPEC["evidence"], "physics_assumptions": SPEC["physics"],
    }
    (OUT / "arena_manifest.json").write_text(json.dumps(manifest, indent=2))
    return scene, cols, main_records, frame_records, senior_records, top, hero, detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-usd", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    bpy.context.preferences.filepaths.save_version = 0
    scene, cols, main_records, frame_records, senior_records, top, hero, detail = build()
    if not args.skip_usd:
        reports = {
            "photo_reference": export_asset("challenge_arena", main_records),
            "framed_reference": export_asset("challenge_arena_framed", main_records + frame_records),
            "senior_preliminary": export_asset("challenge_arena_senior", senior_records),
        }
        (OUT / "usd_validation.json").write_text(json.dumps(reports, indent=2))
    # Keep any future texture assets inside each editable Blender file.
    bpy.ops.file.pack_all()
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = bpy.data.objects["Playing_Surface"]
    scene.camera = hero
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "challenge_arena_framed.blend"))
    if not args.skip_render:
        scene.render.filepath = str(OUT / "arena_framed.png")
        bpy.ops.render.render(write_still=True)
    # Remove the reference frame completely: hidden passive bodies still collide.
    for rec in frame_records:
        bpy.data.objects.remove(bpy.data.objects[rec["name"]], do_unlink=True)
    bpy.data.collections.remove(cols["frame"])
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "challenge_arena.blend"))
    if not args.skip_render:
        for cam, filename in [(top, "arena_top.png"), (hero, "arena_overview.png"), (detail, "arena_assets.png")]:
            scene.camera = cam
            scene.render.filepath = str(OUT / filename)
            bpy.ops.render.render(write_still=True)
    senior_names = {rec["name"] for rec in senior_records}
    for rec in main_records:
        if rec["name"] not in senior_names:
            bpy.data.objects.remove(bpy.data.objects[rec["name"]], do_unlink=True)
    scene.name = "Senior_Preliminary_Arena"
    scene.camera = hero
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "challenge_arena_senior.blend"))
    if not args.skip_render:
        scene.render.filepath = str(OUT / "arena_senior.png")
        bpy.ops.render.render(write_still=True)
    print("ARENA_BUILD_COMPLETE", OUT)


if __name__ == "__main__":
    main()
