#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arena_mujoco.competition_mission import Mission
p=argparse.ArgumentParser()
p.add_argument('--policy',choices=['neural','expert','zero'],default='neural')
p.add_argument('--output',default='output/competition/proof')
p.add_argument('--max-tasks',type=int)
p.add_argument('--seed',type=int,default=0)
p.add_argument('--randomize',action='store_true')
p.add_argument('--start-jitter-mm',type=float,default=0.)
a=p.parse_args()
r=Mission(a.policy,seed=a.seed,randomize=a.randomize,start_jitter_m=a.start_jitter_mm*.001).run(output=a.output,max_tasks=a.max_tasks)
print(json.dumps(r,indent=2))
sys.exit(1 if r['failure'] or (a.max_tasks is None and not r['success']) else 0)
