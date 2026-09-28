"""Narrow M6 supplement and appended-model orchestrator; default is read-only."""
import argparse,json,os,signal,sys
from pathlib import Path
from utils.ch3_contract import read_profiles,validate_manifest,digest,task_by_id,summarize
from utils.ch3_extension import (queue_ids,fixed_waves,result_path,joined_index,summarized_index,
 extension_reasons,reject_existing,execute_waves,safe_stop,controller_live)

def cli(argv=None):
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dry-run','preflight','start','status','logs','complete','safe-stop','index'],nargs='?',default='dry-run')
    p.add_argument('--model');p.add_argument('--approval');p.add_argument('--probe-report');p.add_argument('--receipts');p.add_argument('--output')
    a=p.parse_args(argv);c=validate_manifest(read_profiles());ids=queue_ids(c,a.model)
    control=Path(c['extension']['result_root'])/'controllers'/('model-'+a.model if a.model else 'catchup-41')
    if a.output:raise ValueError('extension result root is fixed; arbitrary --output forbidden')
    if a.action=='safe-stop':return safe_stop(control)
    if a.action in ('status','logs','complete','index'):
        if a.action=='logs':print(json.dumps([str(result_path(c,task_by_id(c,r))/'worker.log')for r in ids],ensure_ascii=False));return
        if a.action=='index':
            from utils.ch3_result_index import load_receipts,DEFAULT_RECEIPTS
            records=load_receipts(c,a.receipts or DEFAULT_RECEIPTS)
            rows=joined_index(c,records);print(json.dumps(dict(rows=rows,panels=summarized_index(rows)),ensure_ascii=False));return
        if a.action=='complete':
            if not (control/'complete.json').exists():print(json.dumps(dict(complete=False,reason='no validated completion record')));return 2
            result=json.loads((control/'complete.json').read_text())
            if result.get('protocol_sha')!=digest(c) or result.get('task_ids')!=ids:raise ValueError('foreign/incomplete extension report')
            from utils.ch3_m6 import verify_result
            for run in ids:verify_result(c,run,json.loads((result_path(c,task_by_id(c,run))/'result.json').read_text()))
            print(json.dumps(dict(complete=True,tasks=len(ids),result_review='Pending')));return
        print(json.dumps(dict(control=str(control),running=controller_live(control/'controller.json'),tasks=len(ids),progress=str(control/'progress.json'),complete_file=(control/'complete.json').exists()),ensure_ascii=False));return
    approval=json.loads(Path(a.approval).read_text()) if a.approval else None
    reasons=extension_reasons(c,approval,a.model)
    # No absent/old/template approval can reach a GPU or guarded worker.
    if approval and not reasons:
        from ch3_runner import preflight,validate_probe_report,code_binding,environment_binding,hardware_binding
        for model in sorted({task_by_id(c,r)['model']for r in ids}):reasons += preflight(c,model,approval)
    report=None
    if a.probe_report:
        import hashlib
        from ch3_runner import validate_probe_report,code_binding,environment_binding,hardware_binding
        report=json.loads(Path(a.probe_report).read_text())
        validate_probe_report(c,report)
        if report.get('code')!=code_binding() or report.get('environment')!=environment_binding() or report.get('hardware')!=hardware_binding():reasons.append('resource source/environment/hardware mismatch')
        if not approval or hashlib.sha256(Path(a.probe_report).read_bytes()).hexdigest()!=approval.get('probe_report_sha'):reasons.append('resource report not bound by approval')
    else:reasons.append('reviewed extension/carry-forward resource report missing')
    if a.action in ('dry-run','preflight'):
        print(json.dumps(dict(batch=c['extension']['id'],tasks=len(ids),task_ids=ids,total_unique_runs=552,total_run_epochs=6240,technical_budget=c['extension']['technical_budget'],control=str(control),blocked=sorted(set(reasons))),ensure_ascii=False,indent=2))
        return 2 if a.action=='preflight' and reasons else 0
    if reasons:raise PermissionError('; '.join(sorted(set(reasons))))
    reject_existing(c,ids,control);waves=fixed_waves(c,ids,report['decisions'])
    from ch3_runner import GPULock,dump
    def stopped(sig,frame):raise InterruptedError('owned extension controller safe stop')
    old=signal.signal(signal.SIGTERM,stopped)
    try:
        with GPULock(c):
            reject_existing(c,ids,control);control.mkdir(parents=True,exist_ok=False)
            dump(control/'controller.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],batch=c['extension']['id'],model=a.model,protocol_sha=digest(c)))
            execute_waves(c,waves,control,approval)
            dump(control/'complete.json',dict(protocol_sha=digest(c),task_ids=ids,batch=c['extension']['id'],result_review='Pending'))
    finally:signal.signal(signal.SIGTERM,old)

if __name__=='__main__':sys.exit(cli())
