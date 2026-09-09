# Swarm module assets

The module is 24 mm wide and 55 mm long. It retains the two tapered lobes and dark ball housing from the reference photo. Two independent wheels provide locomotion; two rubber contact bands share one vertical lift carriage.

- `swarm_module.blend`: assembly reference with native geometry, materials, lights, and three review cameras. Scene coordinates are meters; displayed units are millimeters.
- `assembly_reference_mm.stl`: full assembly in millimeters.
- `parts/*.stl`: nine separate native collision solids, in millimeters and assembly coordinates.
- `cad_manifest.json`: part bounds and SHA-256 hashes.
- `cad_validation.json`: finite-vertex and closed-edge checks for every STL part.
- `robot.xml`, `geometry.json`: source native simulation assembly and geometry export.
- `contact_validation.json`: six native two-module carry and release tests.
- `module_overview.png`, `contact_detail.png`, `module_top.png`: Blender review renders.
- `module.png`: native MuJoCo preview.

These assets describe solid geometry envelopes for mechanical layout. Engineering the shell cavities, bearings, gears, mounts, fasteners, PCB, and manufacturing fits remains necessary before fabrication.

Regenerate from the repository root:

```sh
.venv/bin/python scripts/export_swarm_robot.py --validate --render
/Applications/Blender.app/Contents/MacOS/Blender --background --python scripts/render_swarm_robot.py
```

See `docs/swarm_robot_design.md` for hardware sources, control conventions, packaging assumptions, and validation limits.
