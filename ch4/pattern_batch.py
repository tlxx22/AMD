"""Finite CPU-only PDFormer cache worklist; no model training or auto-retry."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from .runtime_io import read,write_new,proc_start,publish_complete
from .imports import sha
from .pattern_identity import validate_pattern_identity


def start(plan_path,folder):
    plan=read(plan_path);folder=Path(folder)
    if plan['kind']!='m7-pdformer-train-only-cpu-cache' or plan['threads']!=2 or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise PermissionError('bound CPU-only preparation required')
    if sys.executable!=plan['python']:raise PermissionError('M7 isolated interpreter required')
    from .queue import verify_environment
    verify_environment(plan['environment'])
    for path,expected in plan['bindings'].items():
        if sha(path)!=expected:raise PermissionError('preparation binding changed: '+path)
    if any(Path(t['output']).exists() for t in plan['tasks']):raise FileExistsError('prior attempt exists; no automatic retry/overwrite')
    folder.mkdir(parents=True,exist_ok=False);(folder/'logs').mkdir()
    write_new(folder/'owner.json',dict(pid=os.getpid(),start_ticks=proc_start(os.getpid()),hostname=socket.gethostname(),
             plan_path=str(Path(plan_path).resolve()),plan_sha256=sha(plan_path),tmux=os.environ.get('TMUX')))
    outcomes=[];began=time.monotonic()
    for task in plan['tasks']:
        if (folder/'STOP').exists():break
        command=[sys.executable,'-B','-m','ch4.prepare_native_patterns',task['domain'],'--output',task['output'],
                 '--fixed-identity',task['fixed_graph_identity']['path'],
                 '--candidate-identity',task['candidate_identity']['path'],'--threads','2']
        if task.get('reuse_dtw'):command+=['--reuse-dtw',task['reuse_dtw']['path']]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMBA_NUM_THREADS='2')
        with (folder/'logs'/(task['domain']+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=plan['worktree'],env=env,stdout=log,stderr=subprocess.STDOUT)
            child_start=proc_start(child.pid);signalled=False
            write_new(folder/(task['domain']+'-process.json'),dict(pid=child.pid,start_ticks=child_start,command=command))
            while child.poll() is None:
                if (folder/'STOP').exists() and not signalled:
                    if proc_start(child.pid)!=child_start:raise RuntimeError('owned CPU child identity changed')
                    child.send_signal(signal.SIGINT);signalled=True
                time.sleep(.5)
        identity=Path(task['output'])/'pattern-identity.json'
        if child.returncode==0:
            reference=dict(path=str(identity),sha256=sha(identity))
            validate_pattern_identity(reference,task['domain'],task['metadata_path'],task['fixed_graph_identity'])
            result=dict(domain=task['domain'],status='completed',pattern_identity=reference)
        else:
            failure=Path(task['output'])/'pattern-failure.json'
            result=dict(domain=task['domain'],status='interrupted' if signalled else 'failed',returncode=child.returncode,
                        failure_identity=dict(path=str(failure),sha256=sha(failure)) if failure.exists() else None,
                        log=str(folder/'logs'/(task['domain']+'.log')))
        outcomes.append(result);write_new(folder/(task['domain']+'-receipt.json'),result)
        if result['status']!='completed':break  # retain first failure; no implicit retry or tuning
    attempted={r['domain'] for r in outcomes}
    outcomes += [dict(domain=t['domain'],status='not_executed') for t in plan['tasks'] if t['domain'] not in attempted]
    write_new(folder/'results.json',dict(outcomes=outcomes,wall_seconds=time.monotonic()-began,formal_training=False,gpu=False))
    publish_complete(folder/'preparation-finished.json',dict(all_attempted=len(attempted)==len(plan['tasks']),
                     all_caches_valid=all(r['status']=='completed' for r in outcomes),results_sha256=sha(folder/'results.json'),
                     scientific_or_engineering_review_passed=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['start','status','stop']);p.add_argument('--plan');p.add_argument('--folder',required=True)
    args=p.parse_args();folder=Path(args.folder)
    if args.command=='start':start(args.plan,folder)
    elif args.command=='stop':
        owner=read(folder/'owner.json')
        if owner['hostname']!=socket.gethostname() or proc_start(owner['pid'])!=owner['start_ticks']:raise PermissionError('CPU owner identity mismatch')
        if not (folder/'STOP').exists():write_new(folder/'STOP',dict(reason='user stop',time=time.time()))
    else:print(json.dumps(dict(receipts=[p.name for p in folder.glob('*-receipt.json')],finished=(folder/'preparation-finished.json').exists(),
                              stopped=(folder/'STOP').exists(),results=read(folder/'results.json') if (folder/'results.json').exists() else None),ensure_ascii=False))


if __name__=='__main__':main()
