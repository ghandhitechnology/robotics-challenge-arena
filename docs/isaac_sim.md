# Isaac Sim import and physics checks

For the national senior preliminary task, use `output/challenge_arena_senior.usdc` or open `output/challenge_arena_senior_scene.usda`. That setup has 19 rigid bodies and 25 colliders: twelve cylinders, four kits, three samples, the floor, four tape meshes and the laboratory. The photo arrangement in `challenge_arena.usdc` has ten kits and two containment beams, giving 27 rigid bodies and 33 colliders. `challenge_arena_framed.usdc` adds four static outer rails, giving 37 colliders. Each reusable asset uses meters, kilograms, Z up and `/Arena` as its default prim. Matching standalone wrappers add a physics scene and preview lights. No external textures are required.

The floor, tape, and lab plate use their original mesh for static collisions, preserving the plate holes. Black boundary tape is a solid layer 0.15 mm thick, with contact geometry at that thickness. Movable objects use convex hull or box approximation. Painted zones, crosses, and other surface graphics have no collider. Graphics parented to a movable object follow its rigid body. These assignments follow [OpenUSD's rigid-body and collision schema](https://openusd.org/release/api/usd_physics_page_front.html).

The build derives masses from each object's volume using an assumed wood density of 600 kg/m³. Each cylinder is about 3.770 g, each kit is 7.500 g, and each sample is about 7.389 g. All movable bodies have explicit masses marked as assumptions. The 280 mm beam is 201.6 g and the 250 mm beam is 180 g. Static friction is 0.6, dynamic friction is 0.5, and restitution is 0. Contact offset is 0.5 mm and rest offset is zero. These values need hardware measurements before calibrated experiments. The standalone scene uses gravity of 9.81 m/s², 240 Hz physics, TGS, and CCD. See [NVIDIA's collision offset guidance](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html).

## Static USD verification

Run with a Python environment that provides `pxr`, including Blender's bundled Python:

```sh
/Applications/Blender.app/Contents/Resources/5.2/python/bin/python3.13 scripts/usd_physics.py output/challenge_arena.usdc --manifest output/arena_manifest.json
/Applications/Blender.app/Contents/Resources/5.2/python/bin/python3.13 scripts/usd_physics.py output/challenge_arena_scene.usda --standalone
```

This checks stage composition, SI units, default prim, body assignments, positive masses, physics material bindings, small-part collision offsets, decorative collision exclusion, and exact floor, tape, and lab meshes. When `UsdValidation` is available, it also runs the installed OpenUSD validators for physics, materials, references, geometry, and schema types. All six exports passed the 26 validators bundled with Blender 5.2.1. This check does not run a physics engine.

## Isaac Sim runtime check

On a supported NVIDIA GPU machine with Isaac Sim 5.1, run from this repository:

```sh
/path/to/isaac-sim/python.sh scripts/isaac_smoke_test.py output/challenge_arena_senior_scene.usda --report output/isaac_runtime_report.json
```

Add `--gui` for a visible run. The test loads the standalone stage, initializes PhysX, verifies that every authored body has a physics handle, simulates five seconds, checks each body's position every step, and requires peak linear speed below 0.03 m/s during the final half-second. It reads body state from the [PhysX tensor API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html), which avoids relying on USD pose writeback. A failed check returns a nonzero exit status.

The runtime script targets the Isaac Sim 5.1 standalone API. [NVIDIA's standalone Python guide](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/python_scripting/manual_standalone_python.html) describes its launcher. Isaac Sim has not been run on the Mac export host. Static verification and rendered previews do not establish simulation stability.

For RL use, reference the reusable asset once per environment, provide the training world's physics scene, and bind the actual robot separately. Randomize object poses and physics parameters through the training task. Keep the lab plate as a static mesh when checking insertion into its holes.
