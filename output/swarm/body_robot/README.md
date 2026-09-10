# Magnetic swarm body module assets

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
