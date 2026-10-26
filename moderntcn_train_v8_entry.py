"""Independent candidate CLI. Real execution needs a reviewed clean launch binding."""
import argparse
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('resolve','worker','supervise','status','stop','summary','preflight','waves'))
    for name in ('config','task-id','root','attempt','runtime','runtime-sha','receipt','authorization','task-list','output','synthetic-fixture-path','synthetic-fault'):parser.add_argument('--'+name)
    parser.add_argument('--empty-synthetic',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--gate-policy',help='optional exact versioned policy JSON; cannot override the approved contract')
    a=parser.parse_args()
    if a.action=='resolve':
        from moderntcn_dev_v8.training_config import derive
        p=Path(a.output)
        if p.exists():raise FileExistsError('resolved source snapshot cannot be overwritten')
        p.write_text(json.dumps(derive(),ensure_ascii=False,indent=2)+'\n');return
    if a.action=='worker':
        from moderntcn_dev_v8.worker import run_worker,request_stop
        import signal
        signal.signal(signal.SIGTERM,request_stop)
        result=run_worker(json.loads(Path(a.config).read_text()),a.task_id,a.root,a.attempt,device='cpu' if a.synthetic_fixture_path else 'cuda:0',resume=a.resume,path_override=a.synthetic_fixture_path,fault=a.synthetic_fault,launch_context=dict(path=a.runtime,sha256=a.runtime_sha))
        if a.receipt:
            from moderntcn_dev_v8.training_state import atomic_json
            atomic_json(a.receipt,result)
    elif a.action=='supervise':
        from moderntcn_dev_v8.launcher import supervise
        ids=[] if a.empty_synthetic else json.loads(Path(a.task_list).read_text())
        supervise(a.config,ids,a.root,a.attempt,a.authorization,empty_synthetic=a.empty_synthetic,synthetic_fixture_path=a.synthetic_fixture_path,resume=a.resume);return
    elif a.action in ('preflight','waves'):
        from moderntcn_dev_v8.launcher import batch,validate_authorization
        from moderntcn_dev_v8.training_config import validate
        config=validate(json.loads(Path(a.config).read_text()));ids=json.loads(Path(a.task_list).read_text())
        if a.action=='preflight':
            from moderntcn_dev_v8.worker import RUN_NAMESPACE
            import re
            if not a.root or not Path(a.root).resolve().is_relative_to(RUN_NAMESPACE) or not a.attempt or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,90}',a.attempt):raise ValueError('candidate root/attempt required')
            validate_authorization(config,a.config,ids,json.loads(Path(a.authorization).read_text()))
        result=dict(waves=batch(config,ids),training_started=False,GPU_query=False)
    elif a.action in ('status','stop'):
        from moderntcn_dev_v8.launcher import status,stop
        result=stop(a.root,a.attempt) if a.action=='stop' else status(a.root,a.attempt)
    else:
        from moderntcn_dev_v8.summary import summarize
        policy=json.loads(Path(a.gate_policy).read_text()) if a.gate_policy else None
        result=summarize(json.loads(Path(a.config).read_text()),a.root,a.attempt,policy)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
