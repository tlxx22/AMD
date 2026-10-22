#!/usr/bin/env python3
"""Owned C.1 baseline CLI. Metadata commands do not import torch."""
import argparse
import json
from moderntcn_c1 import launch
from moderntcn_c1.contract import *

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['preflight','arm','chain','worker','status','stop','check','plan']);p.add_argument('--task-id');p.add_argument('--runtime');p.add_argument('--runtime-sha');args=p.parse_args()
    if args.command=='preflight':result=launch.preflight(CONFIG,read(launch.AUTH_PATH))
    elif args.command=='plan':
        c=validate(read(CONFIG));result=dict(tasks=len(c['tasks']),waves=c['waves'],budget=[c['max_run_epochs'],c['max_optimizer_steps']],upstream=c['upstream'])
    elif args.command=='arm':result=launch.arm()
    elif args.command=='chain':launch.chain();result=launch.status()
    elif args.command=='worker':
        from moderntcn_c1.worker import run
        run(args.task_id,args.runtime,args.runtime_sha);result=dict(worker_completed=args.task_id)
    elif args.command=='stop':result=launch.stop()
    elif args.command=='check':
        folder=RESULT_ROOT/ATTEMPT;result=launch.status()
        if not (folder/'complete.json').exists():result.update(technical_complete=False,reason='not completed')
        else:
            done=read(folder/'complete.json');c=read(CONFIG);a=read(launch.AUTH_PATH)
            from moderntcn_c1.artifacts import audit
            from moderntcn_c1.upstream import live
            if any(live(ch)for ch in result.get('children',[])) or (launch.CONTROL/'failure-or-stop.json').exists():raise ValueError('completion has active child or failure/STOP')
            if done['task_ids']!=[t['id']for t in c['tasks']]or done['waves']!=14 or list((folder/'staging').iterdir())or result.get('owner_alive'):raise ValueError('completion/exit/staging mismatch')
            for t in c['tasks']:audit(folder/'runs'/t['id'],c,t,a['commit'])
            result.update(technical_complete=True,summary=bound(done['summary_ref']),result_review='Pending')
    else:result=launch.status()
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
if __name__=='__main__':main()
