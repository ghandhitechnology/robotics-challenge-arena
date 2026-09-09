"""Build, inspect, simulate and view the native arena."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["build","run","view","render"])
    parser.add_argument("--config",default="senior_preliminary",
                        choices=["senior_preliminary","photo_reference","framed_reference"])
    parser.add_argument("--tape",default="flex",choices=["flex","rigid","none"])
    parser.add_argument("--profile",type=Path)
    parser.add_argument("--no-robot",action="store_true")
    parser.add_argument("--seconds",type=float,default=.1)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--randomize",action="store_true")
    parser.add_argument("--drive",type=float,nargs=2,default=[0,0],metavar=("LEFT","RIGHT"))
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.seconds<0:
        parser.error("--seconds cannot be negative")
    if args.command=="build":
        from .builder import export_arena
        path=args.output or Path("output/mujoco")/f"{args.config}.xml"
        from .materials import load_profile
        profile=load_profile(args.profile,seed=args.seed,randomize=args.randomize)
        metadata=export_arena(path,config=args.config,tape_mode=args.tape,robot=not args.no_robot,profile=profile)
        print(json.dumps({"xml":str(path),"metadata":str(path.with_suffix('.json')),
                          "bonds":len(metadata['tape_nodes']),"objects":len(metadata['objects'])},indent=2))
        return
    from .runtime import ArenaSimulation
    sim=ArenaSimulation(config=args.config,tape_mode=args.tape,robot=not args.no_robot,
                        profile=args.profile,seed=args.seed,randomize=args.randomize)
    action=None if args.no_robot else np.asarray(args.drive)
    start=time.monotonic()
    if args.command=="view":
        import mujoco.viewer
        with mujoco.viewer.launch_passive(sim.model,sim.data) as viewer:
            viewer.cam.lookat[:]=[.5715,.5905,0]
            viewer.cam.distance=1.65
            viewer.cam.azimuth=90
            viewer.cam.elevation=-65
            while viewer.is_running():
                sim.step(action,nstep=10)
                viewer.sync()
        return
    if args.seconds:
        sim.step(action,nstep=max(1,round(args.seconds/sim.model.opt.timestep)))
    report={**sim.diagnostics(),"wall_seconds":time.monotonic()-start,"seed":args.seed,
            "mujoco_version":__import__('mujoco').__version__,"backend":"native CPU"}
    if args.command=="render":
        import mujoco
        from PIL import Image
        path=args.output or Path("output/mujoco/arena.png")
        path.parent.mkdir(parents=True,exist_ok=True)
        with mujoco.Renderer(sim.model,height=960,width=1280) as renderer:
            renderer.update_scene(sim.data,camera="overview")
            Image.fromarray(renderer.render()).save(path)
        report["image"]=str(path)
    elif args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
