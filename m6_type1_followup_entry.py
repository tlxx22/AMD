"""No-load preparation and one explicitly authorized unified tmux chain."""
import argparse,json,os,sys,time
from pathlib import Path
from utils import ch3_type1_chain as q
from utils import ch3_type1_tasks as s
from utils.ch3_native_recovery_records import bound,ref,exclusive


def cli():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('dry-run','preflight','prepare-launch','start','probe-child','group-child','status','logs','complete','safe-stop','summary'))
    parser.add_argument('--approval',default=str(s.PACKAGE/'start-review.json'));parser.add_argument('--approval-sha')
    parser.add_argument('--wrapper-pid',type=int);parser.add_argument('--stage',choices=('URBAN_SUBSET','EPF_ALL'));parser.add_argument('--model',choices=s.MODELS)
    parser.add_argument('--runtime');parser.add_argument('--runtime-sha');a=parser.parse_args()
    value=dict(path=a.approval,sha256=a.approval_sha) if a.approval_sha else ref(a.approval) if Path(a.approval).exists() else None
    if a.action in ('dry-run','preflight'):
        readiness=q.readiness_report(bound(value) if value else None);reasons=readiness['blocked']
        plans={k:s.plan(c) for k,c in q.configs().items()}
        print(json.dumps(dict(scope=s.ID,scientific_protocol=s.PROTOCOL,states=q.STATES,manual_review=False,
            permit_generation='preauthorized_machine_gate',prepared_only=True,**readiness,plans=plans),ensure_ascii=False,indent=2))
        return 2 if reasons and a.action=='preflight' else 0
    if a.action=='prepare-launch':
        if not value:raise PermissionError('actual reviewed approval required')
        print(q.prepare_launch(value,a.wrapper_pid));return 0
    if a.action=='start':
        if not value:raise PermissionError('actual reviewed approval required')
        q.start(value,os.environ.get(q.TOKEN));return 0
    if a.action in ('probe-child','group-child'):
        if not a.stage or not a.approval_sha:raise PermissionError('exact child stage/permit SHA')
        for _ in range(50):
            try:q.validate_child(value);break
            except (FileNotFoundError,KeyError,PermissionError):time.sleep(.02)
        else:raise PermissionError('owned child registration absent')
        sys.path.insert(0,str(q.ROOT/'tools/restricted_regression'))
        c=q.configs()[a.stage];permit=bound(value);probe=a.action=='probe-child';q.validate_permit(c,permit,probe)
        if probe:
            from ch3_runner import GPULock
            from utils.ch3_native_execution import run_probe
            root=s.context(c)['probe_root'];root.mkdir(parents=True,exist_ok=False)
            exclusive(root/'approval.json',permit)
            exclusive(root/'controller.json',dict(owner=q.owner(),scope=s.context(c)['probe_scope']))
            with GPULock(c):run_probe(c,permit)
        else:
            if not a.model or not a.runtime or not a.runtime_sha:raise PermissionError('exact formal group/runtime')
            from utils.ch3_type1_execution import run_group
            run_group(c,permit,a.model,dict(path=a.runtime,sha256=a.runtime_sha))
        return 0
    if a.action=='safe-stop':print(json.dumps(q.safe_stop()));return 0
    if a.action=='summary':
        from utils.ch3_type1_summary import result_index
        print(json.dumps(result_index(q.configs()),ensure_ascii=False,indent=2));return 0
    status=q.status()
    if a.action=='logs':status['logs']=[str(q.LOG)]+[str(p) for p in q.CONTROL.glob('*.log')]
    if a.action=='complete':
        if status['running'] or status['STOP'] or status['failure'] or not status['complete']:print(json.dumps(status));return 2
        complete=json.loads((q.CONTROL/'complete.json').read_text())
        if complete.get('scope')!=s.ID or complete.get('technical_complete')is not True or complete.get('result_review')!='pending' or complete.get('total_runs')!=63:raise ValueError('exact technical complete required')
        q.validate_boundary_light(complete['URBAN_boundary'])
        q.validate_boundary_light(complete['EPF_boundary'],'EPF_ALL')
    print(json.dumps(status,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':sys.exit(cli())
