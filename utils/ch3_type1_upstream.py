"""Read-only anchored upstream handoff. Never signals or launches the old scope."""
import json,subprocess,time
from pathlib import Path
from utils.ch3_contract import ROOT,digest
from utils.ch3_native_recovery_records import bound,ref,sha
from m6_remaining_entry import same
BASE='df6a16403e10d51097db8c88829909c533d15652'
OLD_SCOPE='m6-baseline-unified96-oc01-v3'
OLD_PROTOCOL='baseline-unified96-onecycle001-v3'
OLD_WORK=ROOT.parent/'AMD-m-baselines-v1'
OLD_RESULT=ROOT.parent/'amd-execution-evidence/m6/m6-formal-launch-dhozikhu'/OLD_PROTOCOL
OLD_PACKAGE=ROOT.parent/'amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1'/OLD_PROTOCOL
START_REF=dict(path=str(OLD_PACKAGE/'start-review.json'),sha256='4c39ec5e025ca8ea492129d459ba395c70e68d1d2a6a698b2b142518bf8e632d')
CONTROLLER_REF=dict(path=str(OLD_RESULT/'queue/controller/controller.json'),sha256='1b8e6098450326577a3f42091ac09237a2c84caac12d2aff7a9b6403645dda79')
OWNER=dict(pid=36799,start_ticks='171530843')
CODE_REF=dict(path=str(OLD_PACKAGE.with_name('baseline-type1-followup-v1')/'upstream-code-binding.json'),sha256='4ec3c6872e2b48ba6263b05c3c35af0db33c7bb10ed402ed4c299d51f650a153')
ENV_REF=dict(path=str(OLD_PACKAGE/'environment-hardware.json'),sha256='e2fa17dc103fcc70acb9d7c49db75171126afdedf84f0cb5db0bb9fb944576c2')
def anchors():
    from utils import ch3_ms_seal_recovery as recovery
    return recovery.anchors()
def historical_anchors():
    return dict(scope=OLD_SCOPE,scientific_protocol=OLD_PROTOCOL,commit=BASE,worktree=str(OLD_WORK),result_root=str(OLD_RESULT),start_ref=START_REF,controller_ref=CONTROLLER_REF,owner=OWNER,code_ref=CODE_REF,environment_ref=ENV_REF,expected_runs=dict(MS=203,M=84,total=287))
def records():
    start=bound(START_REF);control=bound(CONTROLLER_REF)
    if start.get('scientific_protocol')!=OLD_PROTOCOL or start.get('scope')!=OLD_SCOPE or start.get('closure_commit')!=BASE or start.get('execution_permitted')is not True or control.get('scope')!=OLD_SCOPE or control.get('owner')!=OWNER or control.get('authorization')!=START_REF:raise ValueError('fixed upstream authorization/controller identity')
    return start,control
def health():
    records()
    if any((OLD_RESULT/p).exists()for p in ('queue/controller/STOP','queue/controller/failure.json','probe/MS/STOP','probe/MS/failure.json','probe/M/STOP','probe/M/failure.json')):raise RuntimeError('upstream STOP/failure; successor cannot run')
    for args in (('rev-parse','HEAD'),('rev-parse','@{u}')):
        if subprocess.check_output(['git',*args],cwd=OLD_WORK,text=True).strip()!=BASE:raise ValueError('upstream reviewed commit changed')
    if subprocess.check_output(['git','status','--porcelain'],cwd=OLD_WORK,text=True).strip():raise ValueError('upstream unreviewed source change')
    return True
def owned_refs():
    refs={ (OWNER['pid'],OWNER['start_ticks']):OWNER }
    def register(value):
        pid=int(value['pid'])
        if 'start_ticks' not in value:
            if value.get('state')!='metadata_unavailable' or (Path('/proc')/str(pid)).exists():raise ValueError('unverifiable retained sampled process is still present')
            instance=dict(pid=pid,start_ticks=None,exit_proof='sample metadata_unavailable; PID absent; ticks unknown')
        else:instance=dict(pid=pid,start_ticks=str(value['start_ticks']))
        refs[(instance['pid'],instance['start_ticks'])]=instance
    current=OLD_RESULT/'queue/controller/current.json'
    if current.exists():
        value=json.loads(current.read_text())
        if value.get('owner')!=OWNER:raise ValueError('old current owner mismatch')
        child=value.get('child')
        if child:refs[(child['pid'],child['start_ticks'])]=child
    # Registered formal/probe workers, including sampled monitor descendants, are read only.
    for stage in ('MS','M'):
        for memory in (OLD_RESULT/'queue'/stage).glob('group-*/wave-*/memory.jsonl'):
            with memory.open()as handle:
                for line in handle:
                    for value in json.loads(line).get('owned_pid_metadata',{}).values():
                        register(value)
        complete=OLD_RESULT/'probe'/stage/'complete.json'
        if complete.exists():
            report=json.loads(complete.read_text())
            for value in report.get('evidence',{}).values():
                process=bound(value['process']);memory=Path(value['process']['path']).with_name('memory.jsonl')
                expected=report.get('artifacts',{}).get(str(memory))
                if expected is None or sha(memory)!=expected['sha256']:raise ValueError('upstream bound probe ownership evidence')
                with memory.open()as handle:
                    for line in handle:
                        for v in json.loads(line).get('owned_pid_metadata',{}).values():
                            register(v)
    return list(refs.values())
def expected_ref(value,path):
    if value.get('path')!=str(path):raise ValueError('upstream expected reference path')
    return bound(value)
def validate_completion(top):
    start,_=records()
    if set(top)!={'scope','technical_complete','result_review','MS_boundary','M_boundary','total_runs'} or top['scope']!=OLD_SCOPE or top['technical_complete']is not True or top['result_review']!='pending' or top['total_runs']!=287:raise ValueError('full old 203 MS + 84 M technical completion required')
    receipts={}
    from utils import ch3_baseline_unified_tasks as prior
    for stage,count in (('MS',203),('M',84)):
        c=bound(start['config_refs'][stage]);ids=[t['id']for t in c['tasks']]
        if len(ids)!=count or len(set(ids))!=count:raise ValueError('upstream exact unique task count')
        b=expected_ref(top[stage+'_boundary'],OLD_RESULT/'queue'/stage/'technical-boundary.json')
        if b.get('purpose')!='baseline_unified_'+stage+'_boundary_v1' or b.get('scope')!=OLD_SCOPE or b.get('technical_complete')is not True or b.get('result_review')!='pending' or b.get('task_ids')!=ids or b.get('protocol_sha')!=digest(c) or b.get('commit')!=BASE:raise ValueError('upstream full bound stage')
        permit_path=OLD_RESULT/'queue'/stage/'formal-permit.json';a=json.loads(permit_path.read_text())
        if a.get('start_authorization_ref')!=START_REF or a.get('unified_scope')!=OLD_SCOPE or a.get('commit')!=BASE or a.get('protocol_sha')!=digest(c) or a.get('authorized_task_ids')!=ids or a.get('data_binding_ref')!=c['baseline_unified']['data_ref']:raise ValueError('upstream formal permit fixed ancestry')
        expected_code=bound(CODE_REF)['code'];env=bound(ENV_REF)
        if b.get('code')!=expected_code or a.get('code')!=expected_code or b.get('environment')!=env['environment'] or b.get('hardware')!=env['hardware']:raise ValueError('upstream exact code/environment/hardware coverage')
        for filename,expected in b['code'].items():
            raw=subprocess.check_output(['git','show',BASE+':'+filename],cwd=ROOT)
            import hashlib
            if hashlib.sha256(raw).hexdigest()!=expected or a['code'].get(filename)!=expected:raise ValueError('upstream science source exact commit')
        summary=bound(a['summary_ref'])
        if summary.get('technical_admission')is not True or summary.get('owner')!=OWNER or summary.get('protocol_sha')!=digest(c):raise ValueError('upstream admission lineage')
        report=bound(summary['complete_ref']);probe=bound(report['approval'])
        if probe.get('start_authorization_ref')!=START_REF or report.get('execution_complete')is not True or report.get('scope')!=OLD_SCOPE+'-'+stage+'-probe' or report.get('commit')!=BASE:raise ValueError('upstream probe receipt lineage; no numeric replay')
        if set(b['receipts'])!=set(prior.MODELS):raise ValueError('all seven old model groups')
        data=bound(a['data_binding_ref']);a=dict(a,data_bindings=data['data_bindings'])
        for model in prior.MODELS:
            saved=expected_ref(b['receipts'][model],OLD_RESULT/'queue'/stage/('group-'+model)/'complete.json')
            expected_ids=[t['id']for t in c['tasks']if t['model']==model]
            if saved.get('technical_complete')is not True or saved.get('result_review')!='pending' or saved.get('model')!=model or saved.get('task_ids')!=expected_ids or set(saved.get('artifacts',{}))!=set(expected_ids):raise ValueError('upstream exact sealed result index')
            for task in (t for t in c['tasks']if t['model']==model):
                refs=saved['artifacts'][task['id']];root=OLD_RESULT/stage/('formal-'+model)/task['id']
                if set(refs)!={'manifest.json','result.json','history.jsonl','budget.json','runtime.json','best.pt','last.pt'}:raise ValueError('upstream complete standard artifact inventory')
                for name,value in refs.items():
                    if value['path']!=str(root/name) or Path(value['path']).is_symlink() or not Path(value['path']).is_file():raise ValueError('old artifact exact source path')
                manifest=bound(refs['manifest.json']);result=bound(refs['result.json']);runtime=bound(refs['runtime.json']);budget=bound(refs['budget.json'])
                if sha(refs['history.jsonl']['path'])!=refs['history.jsonl']['sha256']:raise ValueError('upstream history SHA')
                p=c['resolved_profiles'][task['id']]
                if manifest['task']!=task or manifest['profile']!=p or manifest['identity'].get('commit')!=BASE or result.get('id')!=task['id'] or result.get('commit')!=BASE or result.get('scientific_protocol')!=OLD_PROTOCOL or result.get('protocol_sha')!=digest(c) or result.get('profile_sha')!=digest(p) or result.get('data_sha')!=data['data_bindings'][task['dataset']][task['id']]:raise ValueError('old formal science identity')
                final=result.get('final_test',{})
                if final.get('calls')!=1 or final.get('selected')!='best.pt' or final.get('sha256')!=refs['best.pt']['sha256'] or final.get('epoch')!=result['best_epoch'] or runtime.get('error')is not None or runtime.get('task')!=task['id']:raise ValueError('old sealed final-test and runtime completeness')
                if budget.get('counts',{}).get('adam')!=result.get('scheduler_updates'):raise ValueError('old result/accounting identity')

        receipts[stage]=top[stage+'_boundary']
    return receipts
def status(full=False):
    from utils.ch3_ms_seal_recovery import status as recovery_status
    return recovery_status(full)
def historical_status(full=False):
    health();top_path=OLD_RESULT/'queue/controller/complete.json';live=same(OWNER)
    if not top_path.exists():
        if not live:raise RuntimeError('upstream owner exited without legal complete')
        return dict(state='WAIT_V3_COMPLETE_AND_RELEASED',READY_FOR_GPU_EXECUTION=False,owner=OWNER,owner_live=True)
    if live:return dict(state='WAIT_OLD_OWNED_RELEASE',READY_FOR_GPU_EXECUTION=False,owner=OWNER,owner_live=True)
    if not full:return dict(state='OLD_COMPLETE_CANDIDATE_NEEDS_VERIFICATION',READY_FOR_GPU_EXECUTION=False,owner_live=False)
    top=json.loads(top_path.read_text());receipts=validate_completion(top);instances=owned_refs();live=[v for v in instances if same(v)]
    if live:return dict(state='WAIT_OLD_OWNED_RELEASE',READY_FOR_GPU_EXECUTION=False,owned_live=live)
    # SHA is recorded only after anchored provenance, expected subordinate refs and complete 287-task proof passed.
    return dict(state='OLD_V3_TECHNICAL_COMPLETE_AND_RELEASED',READY_FOR_GPU_EXECUTION=True,top_complete_ref=ref(top_path),boundaries=receipts,owned_exited=instances,anchors=anchors(),result_review='pending',numeric_replays=0,test_calls=0)
