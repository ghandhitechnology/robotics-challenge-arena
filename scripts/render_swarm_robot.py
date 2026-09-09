"""Blender: export exact native module envelopes as Blender/STL references.

Run after export_swarm_robot.py using Blender's --background --python flags.
These are assembly and part envelopes; electrical/mechanical production CAD
must add cavities, bearings, shafts, fits, and fasteners.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/swarm/robot"
PARTS = OUT / "parts"
PARTS.mkdir(parents=True, exist_ok=True)
payload = json.loads((OUT / "geometry.json").read_text())
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
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
manifest = {"units": "millimeters", "source": "arena_mujoco/swarm_robot.py",
            "scope": "Solid native collision envelopes; assembly reference for detailed mechanical CAD",
            "part_coordinates": "Assembly coordinates, +Y forward, +Z up, base at Z=7 mm",
            "requires_engineering": ["Shell cavities", "Gear and axle bearings", "Motor mounts",
                                     "Transmission gears", "Fasteners and tolerances", "PCB and battery installation"],
            "parts": []}


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
    obj["purpose"] = "Native collision envelope for mechanical layout reference"
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
bpy.ops.wm.stl_export(filepath=str(OUT/"assembly_reference_mm.stl"), export_selected_objects=True, global_scale=1000)
manifest["assembly_bounds_mm"] = [
    [min(p["bounds_mm"][0][a] for p in manifest["parts"]) for a in range(3)],
    [max(p["bounds_mm"][1][a] for p in manifest["parts"]) for a in range(3)],
]
(OUT / "cad_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")

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
front = camera("Contact bands", (.045, .10, .026), (0, .022, .009), .033)
plan = camera("Dimension review top", (0, 0, .15), (0, 0, 0), .090)
for cam, filename in ((hero, "module_overview.png"), (front, "contact_detail.png"), (plan, "module_top.png")):
    scene.camera = cam
    scene.render.filepath = str(OUT/filename)
    bpy.ops.render.render(write_still=True)
scene.camera = hero
select_only(module)
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/"swarm_module.blend"))
print("Saved module CAD references and renders to", OUT)
