"""Render source-comparable views from the saved Blender deliverables."""

from pathlib import Path
import bpy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def render(camera, filename, size=1500):
    scene = bpy.context.scene
    scene.camera = bpy.data.objects[camera]
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(OUT / filename)
    bpy.ops.render.render(write_still=True)


bpy.ops.wm.open_mainfile(filepath=str(OUT / "challenge_arena.blend"))
render("Camera_Top_Orthographic", "arena_top.png")
render("Camera_Overview", "arena_overview.png")
render("Camera_Assets", "arena_assets.png")
bpy.ops.wm.open_mainfile(filepath=str(OUT / "challenge_arena_framed.blend"))
render("Camera_Overview", "arena_framed.png")
bpy.ops.wm.open_mainfile(filepath=str(OUT / "challenge_arena_senior.blend"))
render("Camera_Overview", "arena_senior.png")
print("PREVIEWS_COMPLETE")
