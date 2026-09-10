"""Blender: export exact native module envelopes as Blender/STL references.

Run after export_swarm_robot.py using Blender's --background --python flags.
These are assembly and part envelopes; electrical/mechanical production CAD
must add cavities, bearings, shafts, fits, and fasteners.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from zipfile import ZIP_DEFLATED, ZipFile
import zlib

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", "--output", dest="out", type=Path)
    parser.add_argument("--magnetic", action="store_true",
                        help="render the body module from output/swarm/body_robot")
    blender_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return parser.parse_args(blender_args)


ARGS = arguments()
OUT = ARGS.out or ROOT / "output/swarm" / ("body_robot" if ARGS.magnetic else "robot")
PARTS = OUT / "parts"
PARTS.mkdir(parents=True, exist_ok=True)
payload = json.loads((OUT / "geometry.json").read_text())
MAGNETIC = payload.get("variant") == "magnetic_body"
if ARGS.magnetic and not MAGNETIC:
    raise ValueError(f"{OUT}/geometry.json is not a magnetic-body export")
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
bpy.context.preferences.filepaths.save_version = 0
scene.unit_settings.system = "METRIC"
scene.unit_settings.length_unit = "MILLIMETERS"
scene.render.engine = "CYCLES"
scene.cycles.samples = 48
scene.cycles.use_denoising = True
scene.render.threads_mode = "FIXED"
scene.render.threads = 4
scene.render.resolution_x = 1600
scene.render.resolution_y = 1200
scene.render.resolution_percentage = 100
scene.world.color = (.30, .30, .30)
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Medium High Contrast"
scene.view_settings.exposure = .5
reference = bpy.data.collections.new("Native module envelopes")
scene.collection.children.link(reference)
materials, module = {}, []
manifest = {"units": "millimeters", "variant": payload.get("variant", "legacy"),
            "source": (["arena_mujoco/swarm_robot.py", "arena_mujoco/swarm_magnets.py"]
                       if MAGNETIC else "arena_mujoco/swarm_robot.py"),
            "scope": "Solid native collision envelopes; assembly reference for detailed mechanical CAD",
            "part_coordinates": "Assembly coordinates, +Y forward, +Z up, base at Z=7 mm",
            "requires_engineering": ["Shell cavities", "Gear and axle bearings", "Motor mounts",
                                     "Transmission gears", "Fasteners and tolerances", "PCB and battery installation"] +
                                    (["Electropermanent magnet selection and internal packaging",
                                      "Pulse-driver circuit, power budget, and thermal testing",
                                      "Fabricated pull-in, holding, and release measurements"] if MAGNETIC else []),
            "parts": [], "sites": payload.get("sites", [])}
if MAGNETIC:
    manifest["magnetic_docking"] = payload["robot"]["magnetic_docking"]


def material(name, color, roughness=.55):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = tuple(color)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = tuple(color)
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def select_only(objects):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]


for record in payload["geoms"]:
    size, kind = record["size_m"], record["type"]
    if kind == 7:
        mesh = bpy.data.meshes.new(record["name"])
        mesh.from_pydata(record["vertices_world_m"], [], record["triangles"])
        mesh.update()
        obj = bpy.data.objects.new(record["name"], mesh)
        reference.objects.link(obj)
    else:
        if kind == 6:
            bpy.ops.mesh.primitive_cube_add(size=2)
            obj = bpy.context.object
            obj.scale = size
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        elif kind == 5:
            bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=size[0], depth=2*size[1])
            obj = bpy.context.object
        elif kind == 2:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=size[0])
            obj = bpy.context.object
        else:
            raise ValueError(f"Unsupported native geom type: {kind}")
        obj.name = record["name"]
        obj.location = record["position_m"]
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = Matrix(record["rotation_matrix"]).to_quaternion()
        for collection in list(obj.users_collection):
            collection.objects.unlink(obj)
        reference.objects.link(obj)
        if kind in (2, 5):
            for polygon in obj.data.polygons:
                polygon.use_smooth = True
    key = tuple(record["rgba"])
    if key not in materials:
        materials[key] = material(f"Native material {len(materials)}", key)
    obj.data.materials.append(materials[key])
    obj["source_geometry"] = record["name"]
    obj["purpose"] = ("Magnetic dock housing envelope" if "_magnet_" in obj.name
                      else "Native collision envelope for mechanical layout reference")
    module.append(obj)
    select_only([obj])
    part_path = PARTS / f'{record["name"]}_mm.stl'
    bpy.ops.wm.stl_export(filepath=str(part_path), export_selected_objects=True, global_scale=1000)
    bpy.context.view_layer.update()
    world_corners = [obj.matrix_world @ Vector(p) for p in obj.bound_box]
    lower = [min(p[a] for p in world_corners)*1000 for a in range(3)]
    upper = [max(p[a] for p in world_corners)*1000 for a in range(3)]
    manifest["parts"].append({"name": obj.name, "stl": str(part_path.relative_to(OUT)),
                              "bounds_mm": [lower, upper],
                              "sha256": hashlib.sha256(part_path.read_bytes()).hexdigest()})
select_only(module)
assembly_path = OUT/"assembly_reference_mm.stl"
bpy.ops.wm.stl_export(filepath=str(assembly_path), export_selected_objects=True, global_scale=1000)
manifest["assembly_bounds_mm"] = [
    [min(p["bounds_mm"][0][a] for p in manifest["parts"]) for a in range(3)],
    [max(p["bounds_mm"][1][a] for p in manifest["parts"]) for a in range(3)],
]
manifest["assembly_stl"] = {"path": assembly_path.relative_to(OUT).as_posix(),
                            "sha256": hashlib.sha256(assembly_path.read_bytes()).hexdigest()}
(OUT / "cad_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")

# Magnetic site markers carry MuJoCo transforms into Blender. They are display
# geometry with negligible thickness, so only the four housings become parts.
site_markers = bpy.data.collections.new("Magnetic dock sites")
scene.collection.children.link(site_markers)
for record in payload.get("sites", []):
    bpy.ops.mesh.primitive_cube_add(size=2)
    marker = bpy.context.object
    marker.name = record["name"]
    marker.scale = record["size_m"]
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    marker.location = record["position_m"]
    marker.rotation_mode = "QUATERNION"
    marker.rotation_quaternion = Matrix(record["rotation_matrix"]).to_quaternion()
    for collection in list(marker.users_collection):
        collection.objects.unlink(marker)
    site_markers.objects.link(marker)
    marker.data.materials.append(material(record["name"] + " face", record["rgba"], .35))
    marker["simulation_site"] = record["name"]
    marker["exported_as_stl"] = False
    marker["peak_force_assumption_n"] = payload["robot"]["magnetic_docking"]["peak_force_n"]

# Display geometry and cameras are excluded from the STL exports.
bpy.ops.object.select_all(action="DESELECT")
bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, -.0001))
ground = bpy.context.object
ground.name = "Display ground"
ground.data.materials.append(material("Warm gray backdrop", (.72, .73, .70, 1), .8))
for name, position, power, scale in (("Key", (.06, .04, .13), .025, .09),
                                     ("Fill", (-.08, -.04, .07), .012, .08),
                                     ("Rim", (.02, -.08, .12), .020, .06)):
    light_data = bpy.data.lights.new(name, "AREA")
    light_data.energy, light_data.shape, light_data.size = power, "DISK", scale
    light = bpy.data.objects.new(name, light_data)
    scene.collection.objects.link(light)
    light.location = position
    light.rotation_euler = (Vector((0, 0, .008))-light.location).to_track_quat("-Z", "Y").to_euler()


def camera(name, position, target, scale):
    data = bpy.data.cameras.new(name)
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.location = position
    obj.rotation_euler = (Vector(target)-obj.location).to_track_quat("-Z", "Y").to_euler()
    data.type, data.ortho_scale, data.lens = "ORTHO", scale, 60
    data.clip_start, data.clip_end = .0001, 10.
    return obj


hero = camera("Module overview", (.080, -.105, .065), (0, 0, .008), .079)
if MAGNETIC:
    front = camera("Magnetic dock detail", (.052, -.040, .028), (.011, -.014, .011), .026)
else:
    front = camera("Contact bands", (.045, .10, .026), (0, .022, .009), .033)
plan = camera("Dimension review top", (0, 0, .15), (0, 0, 0), .090)
for cam, filename in ((hero, "module_overview.png"), (front, "contact_detail.png"), (plan, "module_top.png")):
    scene.camera = cam
    scene.render.filepath = str(OUT/filename)
    bpy.ops.render.render(write_still=True)
scene.camera = hero
select_only(module)
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/("swarm_body_robot.blend" if MAGNETIC else "swarm_module.blend")))


def inspect_stl(path):
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"Truncated STL: {path}")
    triangle_count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + 50*triangle_count:
        raise ValueError(f"Invalid binary STL length: {path}")
    edges, finite = Counter(), True
    for triangle in struct.iter_unpack("<12fH", raw[84:]):
        vertices = [tuple(round(value, 6) for value in triangle[start:start+3])
                    for start in (3, 6, 9)]
        finite &= all(math.isfinite(value) for vertex in vertices for value in vertex)
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edges[tuple(sorted((vertices[a], vertices[b])))] += 1
    return {"part": path.name, "triangles": triangle_count, "finite": finite,
            "closed_edges": bool(edges) and all(count == 2 for count in edges.values())}


validation = [inspect_stl(assembly_path)]
validation.extend(inspect_stl(OUT/item["stl"]) for item in manifest["parts"])
(OUT/"cad_validation.json").write_text(json.dumps(validation, indent=2)+"\n")
if not all(item["finite"] and item["closed_edges"] for item in validation):
    raise ValueError("An exported STL is non-finite or open")

if MAGNETIC:
    readme = """# Magnetic swarm body module assets

This 24 x 55 mm, 40 g design adds four releasable shoulder docking ports to the legacy three-motor module. Each port has a solid housing STL and an oriented MuJoCo site recorded in the Blender scene.

- `swarm_body_robot.blend`: 13 native solid envelopes, four visible dock-site markers, materials, lights, and review cameras.
- `assembly_reference_mm.stl`: complete 13-part assembly in millimeters.
- `parts/*.stl`: nine legacy solids and four magnetic housing solids.
- `cad_manifest.json`: part bounds, SHA-256 hashes, dock transforms, and force assumptions.
- `cad_validation.json`: finite-triangle and closed-edge checks for the assembly and every part.
- `robot.xml`, `geometry.json`, `swarm_body_robot_design.json`: native magnetic simulation export.
- `contact_validation.json`: legacy carry coupons repeated with the housings installed.
- `magnetic_validation.json`: native join and coordinated release coupons.
- `module_overview.png`, `contact_detail.png`, `module_top.png`, `module.png`: Blender and MuJoCo review images.

The EPMs, internal cavities, switching circuit, PCB, battery, wiring, fasteners, and thermal design remain packaging work. The 0.15 N per-port force and 3 mm reach are simulation assumptions.

Regenerate from the repository root:

```sh
.venv/bin/python scripts/export_swarm_robot.py --magnetic --validate --render
/Applications/Blender.app/Contents/MacOS/Blender --background --python scripts/render_swarm_robot.py -- --magnetic
```

See `docs/swarm_robot_design.md` for the hardware assumptions and source precedent.
"""
    (OUT/"README.md").write_text(readme)
    support = [ROOT/path for path in ("arena_mujoco/swarm_robot.py",
                                      "arena_mujoco/swarm_magnets.py",
                                      "scripts/export_swarm_robot.py",
                                      "scripts/render_swarm_robot.py",
                                      "docs/swarm_robot_design.md",
                                      "requirements-mujoco.txt")]
    artifacts = sorted(path for path in OUT.rglob("*")
                       if path.is_file() and path.name != "bundle_manifest.json")
    bundled = artifacts + support
    entries = []
    for path in bundled:
        content = path.read_bytes()
        entries.append({"path": path.relative_to(ROOT).as_posix(), "bytes": len(content),
                        "crc32": f"{zlib.crc32(content) & 0xffffffff:08x}",
                        "sha256": hashlib.sha256(content).hexdigest()})
    bundle_manifest = {"schema_version": 1, "bundle": "swarm_body_robot_assets",
                       "variant": "magnetic_body", "file_count_excluding_manifest": len(entries),
                       "files": entries}
    manifest_path = OUT/"bundle_manifest.json"
    manifest_path.write_text(json.dumps(bundle_manifest, indent=2)+"\n")
    archive = OUT.parent/"swarm_body_robot_assets.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as package:
        for path in bundled + [manifest_path]:
            package.write(path, path.relative_to(ROOT).as_posix())
    with ZipFile(archive) as package:
        corrupt = package.testzip()
        if corrupt is not None:
            raise ValueError(f"Archive CRC check failed: {corrupt}")
    print("Saved", len(manifest["parts"]), "closed CAD parts and CRC-checked bundle to", archive)
else:
    print("Saved module CAD references and renders to", OUT)
