#!/usr/bin/env python3
"""Train and export the compact goal controller on CUDA.

Teacher distillation avoids sparse-reward exploration. All demonstrations live
on the GPU. A bias-free odd network guarantees zero command at zero goal error.
Validation includes small errors, unseen seeds and independently varied axes.
No simulator, rendering, host minibatch transfer, torch.compile startup, or
policy-gradient rollout is required for this feedback-learning stage.
"""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', default='output/competition/policy')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260910)
    parser.add_argument('--allow-other-gpu', action='store_true')
    parser.add_argument('--print-artifact', action='store_true')
    args=parser.parse_args()
    start=time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the requested training run')
    device=torch.device('cuda')
    gpu=torch.cuda.get_device_name(0)
    if not args.allow_other_gpu and 'A100' not in gpu:
        raise RuntimeError(f'A100 required, allocated {gpu}')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    # High-accuracy FP32 avoids quantization at millimetre-scale goals; TF32
    # handles the GEMMs. This model is too small to amortize compiler startup.
    layers=[nn.Linear(4,48,bias=False),nn.Tanh(),nn.Linear(48,48,bias=False),nn.Tanh(),nn.Linear(48,4,bias=False),nn.Tanh()]
    net=nn.Sequential(*layers).to(device)
    # A stopped axis must remain stopped while another axis moves. This known
    # mechanical independence constrains learning and makes final alignment safe.
    masks=[]
    for layer_index,layer in enumerate((net[0],net[2],net[4])):
        mask=torch.zeros_like(layer.weight)
        for channel in range(4):
            if layer_index==0:
                mask[channel*12:(channel+1)*12,channel]=1
            elif layer_index==1:
                mask[channel*12:(channel+1)*12,channel*12:(channel+1)*12]=1
            else:
                mask[channel,channel*12:(channel+1)*12]=1
        layer.weight.data.mul_(mask)
        layer.weight.register_hook(lambda grad, mask=mask: grad*mask)
        masks.append(mask)
    optimizer=torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=0, fused=True)
    n=131072
    def dataset(count, generator=None):
        x=torch.rand((count,4),device=device,generator=generator)*16-8
        # Repeated small-error examples devote capacity to final alignment.
        x[:count//2]*=.125
        x[:count//8]*=.1
        # Axis-only examples prevent coupling between otherwise stopped axes.
        idx=torch.arange(count//4, device=device)
        x[-count//4:]*=torch.nn.functional.one_hot(idx%4,4)
        return x,torch.tanh(x)
    train_x,train_y=dataset(n)
    permutation=torch.randperm(n,device=device)
    train_x,train_y=train_x[permutation],train_y[permutation]
    generator=torch.Generator(device=device).manual_seed(args.seed+1)
    valid_x,valid_y=dataset(16384,generator)
    torch.cuda.synchronize()
    preparation=time.perf_counter()-start
    train_start=time.perf_counter()
    history=[]
    batch=8192
    best=float('inf')
    best_state=None
    for step in range(args.steps):
        # Contiguous rotated batches cover the frozen on-device dataset.
        offset=(step*batch)%n
        x,y=train_x[offset:offset+batch],train_y[offset:offset+batch]
        output=net(x)
        # More weight near zero, where motion must settle rather than drift.
        weight=1+4*(x.abs()<.2)
        loss=((output-y).square()*weight).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step%100==0 or step==args.steps-1:
            with torch.no_grad():
                err=(net(valid_x)-valid_y).abs()
                mae=err.mean().item()
                p99=torch.quantile(err.flatten(),.99).item()
                near=(err[valid_x.abs()<.1]).mean().item()
            history.append({'step':step+1,'validation_mae':mae,'validation_p99':p99,'near_zero_mae':near})
            if mae<best:
                best=mae
                best_state={k:v.detach().clone() for k,v in net.state_dict().items()}
            if step%500==0:
                print(json.dumps(history[-1]),flush=True)
            if p99<.003 and near<.0005 and step>=500:
                break
            if step in (4000,7500):
                for group in optimizer.param_groups:
                    group['lr']*=.3
    net.load_state_dict(best_state)
    torch.cuda.synchronize()
    train_seconds=time.perf_counter()-train_start
    arrays={f'w{i}':layer.weight.detach().cpu().numpy() for i,layer in enumerate((net[0],net[2],net[4]))}
    with torch.no_grad():
        pred=net(valid_x)
        e=(pred-valid_y).abs()
        zero=float(net(torch.zeros(4,device=device)).abs().max())
        final={'mae':e.mean().item(),'p99':torch.quantile(e.flatten(),.99).item(),'max':e.max().item(),'zero_action_max':zero}
    report={
        'method':'supervised teacher distillation of goal feedback; geometric task planner outside policy',
        'gpu':gpu,'torch':torch.__version__,'cuda':torch.version.cuda,'seed':args.seed,
        'parameters':sum(p.numel() for p in net.parameters()),'architecture':[4,48,48,4],
        'bias':False,'activation':'tanh at every layer','channel_independence':'block diagonal masks','active_parameters':sum(int(m.sum()) for m in masks),'training_examples':n,'validation_examples':len(valid_x),
        'batch_size':batch,'optimizer':'fused AdamW','precision':'FP32 with TF32 matmul allowed',
        'steps':step+1,'preparation_seconds':preparation,'training_seconds':train_seconds,
        'validation':final,'history':history,
        'efficiency':['on-device frozen demonstrations with one seeded shuffle','near-goal oversampling','axis-only curriculum','independent physical channels','odd symmetry and exact zero equilibrium','validation early stopping','small NumPy export'],
        'physics_validation':'Performed separately with exported weights in native MuJoCo. These validation errors alone do not establish task completion.'}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);blob=buffer.getvalue()
    digest=hashlib.sha256(blob).hexdigest()
    report['weights_sha256']=digest
    (out/'weights.npz').write_bytes(blob)
    (out/'training.json').write_text(json.dumps(report,indent=2)+'\n')
    print('TRAINING_REPORT '+json.dumps(report),flush=True)
    if args.print_artifact:
        import zipfile
        package=io.BytesIO()
        with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('weights.npz',blob)
            z.writestr('training.json',json.dumps(report,indent=2)+'\n')
        payload=package.getvalue()
        print(f'ARTIFACT_BYTES {len(payload)} SHA256 {hashlib.sha256(payload).hexdigest()}',flush=True)
        print('ARTIFACT_BASE64 '+base64.b64encode(payload).decode(),flush=True)

if __name__=='__main__':
    main()
