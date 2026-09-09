"""Check compiled arena dimensions and optionally drive across flex tape."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time
import mujoco
import numpy as np

from arena_mujoco.runtime import ArenaSimulation


def validate(config="senior_preliminary", seconds=.05, drive=False):
    start=time.monotonic()
    sim=ArenaSimulation(config=config)
    m,d=sim.model,sim.data
    checks=[]
    def check(condition,label):
        if not condition:
            raise AssertionError(label)
        checks.append(label)
    floor=m.geom("floor").id
    check(np.allclose(m.geom_size[floor],[.5715,.5905,.005],atol=1e-12),"1143 x 1181 mm playing surface")
    check(np.allclose(m.flex_radius,.000075),"0.15 mm tape collision thickness")
    check(m.nflex==10,"Ten deformable tape strips")
    expected=19 if config=="senior_preliminary" else 27
    check(len(sim.metadata["objects"])==expected,"Configuration object count")
    for item in sim.metadata["objects"]:
        actual=m.body_mass[m.body(item["name"]).id]
        check(np.isclose(actual,item["mass_kg"],rtol=1e-8),f"Volume-derived mass: {item['name']}")
    check(np.isclose(sum(x["mass_kg"] for x in sim.metadata["tape_strips"]),.08602*.00015*1350),"Tape mass equals area x thickness x density")
    check(np.all(m.geom_adhesion==0),"Visible body surfaces have no adhesion")
    dryids=[m.pair(x["name"]).id for x in sim.metadata["contact_pairs"]]
    check(np.all(m.pair_adhesion[dryids]==0),"Dry contact pairs have no adhesion")
    group=np.array([1,0,0,0,0,0],dtype=np.uint8)
    for y in [.3905,.4905,.5905]:
        hit=np.array([-1],dtype=np.int32)
        distance=mujoco.mj_ray(m,d,np.array([1.068,y,.1]),np.array([0.,0.,-1.]),group,1,-1,hit)
        check(hit[0]==floor and abs(distance-.1)<1e-9,"Laboratory hole passes through to floor")
    hit=np.array([-1],dtype=np.int32)
    distance=mujoco.mj_ray(m,d,np.array([1.01,.4905,.1]),np.array([0.,0.,-1.]),group,1,-1,hit)
    check(abs(distance-.097)<1e-8,"Laboratory plate is 3 mm tall")
    if drive:
        q=int(m.joint("robot_free").qposadr[0])
        d.qpos[q:q+2]=[.9,.78]
        d.qpos[q+3:q+7]=[np.cos(np.pi/4),0,0,np.sin(np.pi/4)]
        mujoco.mj_forward(m,d)
    sim.step([.3,.3] if drive else [0,0],nstep=max(1,round(seconds/m.opt.timestep)))
    check(np.isfinite(d.qpos).all() and not any(w.number for w in d.warning),"Finite simulation without solver warnings")
    if not drive:
        check(sim.tape.metrics()["tape_damage_fraction"]==0,"Intact laid tape stays bonded at rest")
    report={"configuration":config,"mujoco_version":mujoco.__version__,"backend":"native CPU",
            "checks":checks,"check_count":len(checks),"wall_seconds":time.monotonic()-start,
            "drive_across_tape":drive,**sim.diagnostics()}
    if drive:
        report["robot_final_position_m"]=d.qpos[q:q+3].tolist()
        check(d.qpos[q]<.84,"Robot wheels crossed start-zone tape")
        report["check_count"]=len(checks)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="senior_preliminary")
    parser.add_argument("--seconds",type=float,default=.05)
    parser.add_argument("--drive",action="store_true")
    parser.add_argument("--output",type=Path,default=Path("output/mujoco/validation.json"))
    args=parser.parse_args()
    report=validate(args.config,args.seconds,args.drive)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
