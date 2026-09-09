"""Export the existing Blender biohazard emblem for MuJoCo visual geometry."""
import json
from pathlib import Path
import bpy

root = Path(__file__).resolve().parents[1]
bpy.ops.wm.open_mainfile(filepath=str(root / "output/challenge_arena.blend"))
symbol = next(obj for obj in bpy.data.objects if obj.type == "MESH" and obj.name.endswith("_biohazard"))
mesh = symbol.data
mesh.calc_loop_triangles()
result = {"source": "output/challenge_arena.blend", "object": symbol.name,
          "vertices": [list(vertex.co) for vertex in mesh.vertices],
          "triangles": [list(triangle.vertices) for triangle in mesh.loop_triangles]}
path = root / "arena_mujoco/assets/biohazard.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(result, separators=(",", ":")) + "\n")
print(path)
