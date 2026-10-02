"""Exact recovery wrapper and tracked synchronous children; no model imports."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from utils import ch3_native_recovery as recovery
from utils import ch3_native_recovery_records as records


def reference(path,sha=None):
    return dict(path=str(Path(path)),sha256=sha) if sha else records.ref(path)


def cli():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('dry-run','preflight','recovery-preflight','prepare-launch','start',
        'group-child','probe-child','status','logs','complete','safe-stop','drain-stop'))
    p.add_argument('--approval',default=str(recovery.PACKAGE/'recovery-review.json'))
    p.add_argument('--approval-sha');p.add_argument('--wrapper-pid',type=int)
    p.add_argument('--stage',choices=('tmark','m'));p.add_argument('--model');p.add_argument('--runtime');p.add_argument('--runtime-sha')
    a=p.parse_args()
    if a.action in ('dry-run','preflight','recovery-preflight'):
        approval=records.bound(reference(a.approval,a.approval_sha)) if Path(a.approval).exists() else None
        reasons=recovery.readiness(approval)
        budget=records.budget_projection(recovery.original_configs()['tmark'],recovery.prepared('inventory'))
        print(json.dumps(dict(scope=recovery.ID,states=recovery.STATES,blocked=reasons,budget=budget,
            classifications=recovery.prepared('inventory')['counts'],manual_review=False,
            permit_generation=records.MODE,prepared_only=True,tmark_probe_repeated=False,
            original_science_execution_commit=recovery.BASE,controller_execution_commit=None),ensure_ascii=False,indent=2))
        return 2 if a.action!='dry-run' and reasons else 0
    if a.action=='prepare-launch':
        print(recovery.prepare_launch(reference(a.approval,a.approval_sha),a.wrapper_pid));return 0
    if a.action=='start':
        recovery.start(reference(a.approval,a.approval_sha),os.environ.get(recovery.TOKEN_ENV));return 0
    if a.action in ('group-child','probe-child'):
        if not a.stage or not a.approval_sha:raise PermissionError('exact owned child scope/permit SHA')
        permit_ref=reference(a.approval,a.approval_sha)
        # Parent publishes the owned record immediately after Popen. The child
        # may start before that atomic JSON write; bounded registration only.
        for _ in range(50):
            try:recovery.validate_child_parent(permit_ref);break
            except (FileNotFoundError,KeyError,PermissionError):time.sleep(.02)
        else:raise PermissionError('actual tracked parent registration absent')
        sys.path.insert(0,str(recovery.ROOT/'tools/restricted_regression'))
        with recovery.activate(a.stage) as c:
            permit=records.bound(permit_ref);probe=a.action=='probe-child'
            recovery.validate_permit_light(c,permit,probe)
            try:
                if probe:
                    if a.stage!='m':raise PermissionError('no tmark probe in recovery')
                    from ch3_runner import GPULock
                    from utils import ch3_native_execution as execution
                    root=execution.roots(c,True);root.mkdir(parents=True,exist_ok=False)
                    records.exclusive(root/'controller.json',dict(**recovery.identity(os.getpid()),approval=permit_ref))
                    records.exclusive(root/'approval.json',permit)
                    with GPULock(c):execution.run_probe(c,permit)
                else:
                    if not a.model or not a.runtime or not a.runtime_sha:raise PermissionError('owned formal model/runtime required')
                    from utils.ch3_native_recovery_execution import run_group
                    run_group(c,permit,permit_ref,reference(a.runtime,a.runtime_sha),a.model)
            except recovery.DrainStop:return 3
        return 0
    if a.action in ('safe-stop','drain-stop'):
        print(json.dumps(recovery.safe_stop(a.action=='drain-stop')));return 0
    status=recovery.status()
    if a.action=='logs':status['logs']=[str(recovery.LOG)]+[str(p) for p in recovery.CONTROL.glob('*.log')]
    if a.action=='complete':
        if not status.get('complete') or status['running'] or status['STOP'] or status['DRAIN_STOP'] or status['failure']:
            print(json.dumps(status));return 2
        v=json.loads((recovery.CONTROL/'complete.json').read_text())
        if v.get('scope')!=recovery.ID or v.get('technical_complete')is not True or v.get('result_review')!='pending':raise ValueError('exact recovery technical completion')
        recovery.validate_boundary_light(v['receipts']['SEAL_87_REPLACEMENT_BOUNDARY'])
        status.update(technical_complete=True,result_review='pending')
    print(json.dumps(status,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':sys.exit(cli())
