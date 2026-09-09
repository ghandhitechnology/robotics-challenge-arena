"""Run with Isaac Sim 5.1's python.sh on a supported NVIDIA GPU machine."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd", type=Path, help="Standalone scene USD")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.seconds < 2:
        parser.error("--seconds must be at least 2 to check settling")

    # Kit must start before importing its extension modules.
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": not args.gui})
    try:
        import numpy as np
        import omni.physics.tensors as tensors
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import open_stage, is_stage_loading
        from pxr import Usd, UsdGeom, UsdPhysics
        from usd_physics import verify_usd

        static_report = verify_usd(args.usd, standalone=True)
        if not open_stage(str(args.usd.resolve())):
            raise RuntimeError(f"Could not open stage: {args.usd}")
        while is_stage_loading():
            app.update()
        stage = omni.usd.get_context().get_stage()
        bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        floors = [p for p in stage.Traverse() if p.GetAttribute("arena:role").Get() == "floor"]
        if not bodies or not floors:
            raise RuntimeError("Expected dynamic bodies and a tagged floor")
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        floor_z = max(bounds.ComputeWorldBound(p).ComputeAlignedRange().GetMax()[2] for p in floors)
        scene = next(p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene))
        world = World(stage_units_in_meters=1.0, physics_dt=1 / 240,
                      rendering_dt=1 / 60, physics_prim_path=str(scene.GetPath()))
        world.reset()
        sim_view = tensors.create_simulation_view("numpy")
        # Exact paths avoid wildcard matching differences across Kit versions.
        views = [(str(p.GetPath()), sim_view.create_rigid_body_view(str(p.GetPath()))) for p in bodies]
        for path, view in views:
            if view.count != 1:
                raise RuntimeError(f"PhysX did not create exactly one body for {path}")
        max_final_speed = 0.0
        for step in range(round(args.seconds * 240)):
            world.step(render=args.gui and step % 4 == 0)
            for path, view in views:
                pose = view.get_transforms()[0]
                if not np.all(np.isfinite(pose)) or pose[2] < floor_z - 0.003:
                    raise AssertionError(f"Body fell through the floor or has invalid pose: {path}: {pose}")
                if step >= round((args.seconds - 0.5) * 240):
                    velocity = view.get_velocities()[0]
                    if not np.all(np.isfinite(velocity)):
                        raise AssertionError(f"Invalid body velocity: {path}")
                    max_final_speed = max(max_final_speed, float(np.linalg.norm(velocity[:3])))
        if max_final_speed > 0.03:
            raise AssertionError(f"Bodies have not settled; final 0.5 s peak speed {max_final_speed:.5f} m/s")
        result = {"runtime_validation": "passed", "simulated_seconds": args.seconds,
                  "dynamic_bodies": len(views), "floor_top_z_m": float(floor_z),
                  "final_peak_speed_m_s": max_final_speed, "static_validation": static_report}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        app.close()


if __name__ == "__main__":
    main()
