"""Guarded CLI for the successor tmux supervisor and synchronous owned stages."""
import argparse
import json
import os
import sys
from pathlib import Path
from utils import ch3_native_chain as chain
from utils import ch3_native_tasks as scope
from utils import ch3_native_execution as execution
from utils import ch3_m_tasks as source
from m6_remaining_entry import same


def cli():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('dry-run','preflight','prepare-launch','start','logs','status','complete','safe-stop','stage-start','group'))
    p.add_argument('--wrapper-pid',type=int);p.add_argument('--stage',choices=('tmark','m'));p.add_argument('--probe',action='store_true');p.add_argument('--approval');p.add_argument('--model')
    a=p.parse_args()
    if a.action in ('dry-run','preflight'):
        cs=chain.configs();reasons=chain.preflight()
        print(json.dumps(dict(scope=scope.ID,states=chain.STATES,plans={k:scope.plan(c) for k,c in cs.items()},blocked=reasons,manual_review=False,permit_generation=chain.MODE,prepared_only=True),ensure_ascii=False,indent=2))
        return 2 if a.action=='preflight' and reasons else 0
    if a.action=='prepare-launch':print(chain.prepare_launch('handoff',a.wrapper_pid));return 0
    if a.action=='start':chain.start(os.environ.get(chain.TOKEN_ENV));return 0
    if a.action=='safe-stop':print(json.dumps(chain.safe_stop()));return 0
    if a.action in ('stage-start','group'):
        if not a.stage or not a.approval:raise PermissionError('exact owned successor stage/permit required')
        c=chain.configs()[a.stage];permit=source.bound(source.ref(a.approval))
        if a.action=='stage-start':chain.stage_start(c,permit,a.approval,a.probe,os.environ.get(chain.TOKEN_ENV))
        else:
            controller=json.loads((scope.context(c)['control']/'controller.json').read_text())
            if not same(controller) or controller['pid']!=os.getppid() or controller['approval']!=source.ref(a.approval):raise PermissionError('group requires actual owned synchronous stage parent')
            chain.stop_check();sys.path.insert(0,str(scope.ROOT/'tools/restricted_regression'));execution.run_group(c,permit,a.model)
        return 0
    value=chain.status()
    if a.action=='complete':
        if not value['complete']:print(json.dumps(value));return 2
        report=json.loads((chain.ROOT_CONTROL/'complete.json').read_text());cs=chain.configs()
        if value['running'] or value['STOP'] or 'failure'in value or report['scope']!=scope.ID or report['result_review']!='pending' or report['technical_complete']is not True:raise ValueError('successor total completion identity')
        for stage,c in cs.items():
            for probe in (True,False):chain.verify_stage(c,source.bound(report['receipts'][stage+('_probe_permit' if probe else '_formal_permit')]),probe)
        chain.replacement_boundary(cs['m'],report['replacement_boundary'])
        print(json.dumps(dict(technical_complete=True,result_review='pending',manual_review=False)));return 0
    if a.action=='logs':value['logs']=[str(chain.launch_paths(k)['log']) for k in ('handoff','tmark-probe','tmark-formal','m-probe','m-formal')]
    print(json.dumps(value,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':sys.exit(cli())
