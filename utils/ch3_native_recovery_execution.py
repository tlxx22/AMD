"""Restricted read-old/write-new workers and fixed-wave recovery execution."""
import contextlib
import json
import math
import os
import shutil
from pathlib import Path
from utils.ch3_contract import ROOT, digest, profile, task_by_id, step_arithmetic
from utils import ch3_native_recovery as recovery
from utils import ch3_native_recovery_records as records
from utils import ch3_native_tasks as scope
from utils import ch3_native_execution as execution
from m6_remaining_entry import identity, same


def read_config(s):
    if s.get('recovery_scope')!=recovery.ID:raise PermissionError('foreign recovery worker')
    stage='tmark' if s.get('protocol_file')==str(scope.PROFILE_FILE) else 'm'
    c=recovery.original_configs()[stage]
    allowed=recovery.ID+'-'+stage+('-probe' if s.get('purpose')=='ch3_probe' else '-formal')
    expected=str(scope.PROFILE_FILE if stage=='tmark' else ROOT/'configs/ch3_formal_profiles.json')
    if s.get('protocol_file')!=expected or s.get('protocol_sha')!=digest(c) or s.get('successor_scope')!=allowed:
        raise PermissionError('exact recovery task protocol/scope')
    if stage=='tmark' and s.get('purpose')=='ch3_probe':raise PermissionError('never repeat v4 probe')
    return c


def permit_reference(s):
    return s.get('formal_permit_ref') or s.get('probe_permit_ref')


def resume_row(c,a,run):
    inv=records.validate_inventory(c,records.bound(a['inventory_ref']))
    row=next(r for r in inv['rows'] if r['task_id']==run)
    if row['classification']!='B':return None
    for name in ('manifest.json','last.pt','best.pt','history.jsonl'):
        r=row['files'][name];p=Path(r['path'])
        if p.parent!=Path(row['original_root']) or p.is_symlink() or records.sha(p)!=r['sha256']:
            raise ValueError('immutable resume prefix changed')
    manifest=json.loads(Path(row['files']['manifest.json']['path']).read_text())
    t=task_by_id(c,run)
    if manifest['task']!=t or manifest['profile']!=profile(c,t) or manifest['identity']['commit']!=records.BASE or manifest['identity']['data_sha']!=row['data_sha']:
        raise ValueError('resume task/profile/data/science identity')
    return row


def prefix_files(c,t):
    d=c['datasets'][t['dataset']]
    return {str(Path(d['path'])/f):c['urban_folds'][t['fold']-1][2] for f in ('volume.csv','e_price.csv','s_price.csv','weather_central.csv')} if t['dataset']=='UrbanEV' else {d['path']:d['endpoints'][2]}


def make_formal_config(c,run,a,permit_ref,runtime_ref):
    from m5_formal_entry import repository_files
    from resource_budget import initialize
    recovery.validate_permit_light(c,a);recovery.validate_runtime_light(c,runtime_ref,permit_ref)
    t=task_by_id(c,run);ctx=scope.context(c);out=scope.result_path(c,t)
    row=resume_row(c,a,run) if ctx['stage']=='tmark' else None
    if ctx['stage']=='tmark':
        inv=records.bound(a['inventory_ref']);r=next(r for r in inv['rows'] if r['task_id']==run)
        if r['classification']=='A':raise PermissionError('complete run is inherited; never train/test')
        counts=records.future_counts(c,t,r)
    else:counts=records.future_counts(c,t,dict(classification='D'))
    out.mkdir(parents=True,exist_ok=False)
    s=dict(version='restricted-regression-minimal-v3',repo=str(ROOT),tool_root=str(ROOT/'tools/restricted_regression'),
           purpose='ch3_formal',task=run,case=None,ids=[run],protocol_file=str(ctx['protocol_file']),protocol_sha=digest(c),
           recovery_scope=recovery.ID,successor_scope=ctx['formal_scope'],successor_phase=None,
           session_root=str(ctx['result_root']),fixture_root=str(ctx['fixture']),audit_log=str(out/'audit.jsonl'),
           budget_file=str(out/'budget.json'),output=str(out),artifact_root=str(out),
           limits=dict(**counts,seconds=None),bound_files=repository_files(),author_roots=[v['repository'] for v in c['sources'].values()],
           author_files=c['sources'].get(t['model'],{}).get('files',{}),prefix_files=prefix_files(c,t),
           formal_permit_ref=permit_ref,runtime_admission_ref=runtime_ref,approval=None,
           metadata_files={},forbidden_roots=[],device='cuda:0',resume=bool(row),kernel_probe=False)
    s['metadata_files']=dict(recovery.metadata_files(c,a,runtime_ref,row),**{permit_ref['path']:permit_ref['sha256']})
    if row:
        s['resume_source_ref']=records.exclusive(out/'resume-source.json',dict(
            purpose='native_recovery_resume_source_v1',task_id=run,inventory_ref=a['inventory_ref'],row=row,
            science_baseline_commit=records.BASE,budget_refund=False,read_old_write_new=True))
        s['metadata_files'][s['resume_source_ref']['path']]=s['resume_source_ref']['sha256']
    for name in ('cache/torch/kernels','mpl','cuda-cache'):(out/name).mkdir(parents=True,exist_ok=True)
    records.exclusive(out/'config.json',s);initialize(s['budget_file'],'ch3_formal',s['limits'])
    return s


def validate_worker(c,s):
    from m5_formal_entry import repository_files
    probe=s.get('purpose')=='ch3_probe';stage='tmark' if 'native_replacement' in c else 'm'
    with recovery.activate(stage):
        if s.get('purpose') not in ('ch3_probe','ch3_formal'):raise PermissionError('exact recovery worker purpose')
        ctx=scope.context(c);t=task_by_id(c,s['task']);read_config(s)
        if s['bound_files']!=repository_files() or s['ids']!=[t['id']] or s['device']!='cuda:0' or s['kernel_probe']is not False:
            raise PermissionError('recovery bound restricted worker identity')
        a=records.bound(permit_reference(s));recovery.validate_permit_light(c,a,probe,worker=True)
        execution.exact_path(c,t,probe,s.get('successor_phase'),s['output'])
        if s['artifact_root']!=s['output'] or s['session_root']!=str(ctx['probe_root'] if probe else ctx['result_root']) or s['fixture_root']!=str(ctx['fixture']) or os.environ.get('TMPDIR')!=s['fixture_root'] or not ctx['fixture'].is_dir():raise ValueError('exact recovery fixture/output')
        row=None
        if not probe:
            if s.get('approval')is not None:raise ValueError('formal config must not embed permit')
            recovery.validate_runtime_light(c,s['runtime_admission_ref'],s['formal_permit_ref'])
            row=resume_row(c,a,t['id']) if stage=='tmark' else None
            inv=records.bound(a['inventory_ref']) if stage=='tmark' else None
            r=next(r for r in inv['rows'] if r['task_id']==t['id']) if inv else dict(classification='D')
            if r['classification']=='A':raise PermissionError('completed task cannot execute')
            limits=dict(**records.future_counts(c,t,r),seconds=None)
            metadata=recovery.metadata_files(c,a,s['runtime_admission_ref'],row)
            metadata[s['formal_permit_ref']['path']]=s['formal_permit_ref']['sha256']
            if bool(row)!=s['resume']:raise ValueError('resume eligibility exact inventory')
            if row:
                rs=records.bound(s['resume_source_ref'])
                if rs['row']!=row or rs['task_id']!=t['id'] or rs['inventory_ref']!=a['inventory_ref']:raise ValueError('exact resume source')
                metadata[s['resume_source_ref']['path']]=s['resume_source_ref']['sha256']
        else:
            limits=dict(**scope.worker_counts(c,t),seconds=1800);metadata=recovery.metadata_files(c,a)
            metadata[s['probe_permit_ref']['path']]=s['probe_permit_ref']['sha256']
            if s['resume']is not False:raise ValueError('M probe is fresh only')
        if s['metadata_files']!=metadata or s['limits']!=limits or s['prefix_files']!=({} if probe else prefix_files(c,t)) or s['author_files']!=c['sources'].get(t['model'],{}).get('files',{}):raise ValueError('worker exact capability/budget/data bindings')
        if any('numeric-full-step' in p or p.endswith('.bin') for p in metadata):raise PermissionError('formal/probe metadata cannot follow raw probe sidecars')


def validate_wave(c,configs,out):
    if not configs:raise ValueError('empty recovery wave')
    a=records.bound(permit_reference(configs[0]));probe=configs[0]['purpose']=='ch3_probe';ctx=scope.context(c)
    ids=[s['task'] for s in configs]
    if len(set(ids))!=len(ids) or any(permit_reference(s)!=permit_reference(configs[0]) for s in configs):raise ValueError('mixed recovery wave/permit')
    if probe:
        g=execution.group_for(c,ids[0]);phase=configs[0]['successor_phase']
        allowed=[[r] for r in g['representatives']] if phase=='serial' else execution.wave_ids(g['representatives'],int(phase[1:])) if phase in ('q4','q2') and int(phase[1:]) in scope.attempt_widths(g) else []
        root=ctx['probe_root']
    else:
        model=task_by_id(c,ids[0])['model'];waves=scope.formal_waves(c,dict(decisions=a['compact_decisions']),model)
        inventory=records.bound(a['inventory_ref']) if ctx['stage']=='tmark' else dict(rows=[])
        allowed=[w for _,w in records.pending_waves(waves,inventory)];root=ctx['control']
        for s in configs:recovery.validate_runtime_light(c,s['runtime_admission_ref'],s['formal_permit_ref'])
    if ids not in allowed or not Path(out).resolve().is_relative_to(root):raise ValueError('fixed original wave; no refill')
    for s in configs:execution.exact_path(c,task_by_id(c,s['task']),probe,s.get('successor_phase'),s['output'])
    recovery.stop_check()
    return recovery.CONTROL/'STOP'


def resume_before_training(c,t,out,a,model,opt,generator):
    """No identity relaxation: restore old state against exact old identity."""
    from ch3_runner import restore_state, dump
    row=resume_row(c,a,t['id']);old=Path(row['original_root']);manifest=json.loads((old/'manifest.json').read_text())
    best,epoch,steps=restore_state(old/'last.pt',model,opt,manifest['identity'],generator)
    if epoch!=row['checkpoint_committed_epoch'] or steps!=row['checkpoint_committed_steps']:raise ValueError('loaded checkpoint progress differs from audited inventory')
    # Byte-identical inherited best; original artifacts remain read-only.
    with (old/'best.pt').open('rb') as src,(Path(out)/'best.pt').open('xb') as dst:shutil.copyfileobj(src,dst)
    if records.sha(Path(out)/'best.pt')!=row['files']['best.pt']['sha256']:raise ValueError('inherited best copy changed')
    dump(Path(out)/'inherited-best.json',dict(source=row['files']['best.pt'],identity=manifest['identity'],epoch=best.epoch))
    return best,epoch,steps


def best_identity(c,t,out,a,current):
    inherited=Path(out)/'inherited-best.json'
    if inherited.exists():
        row=resume_row(c,a,t['id'])
        if records.sha(Path(out)/'best.pt')==row['files']['best.pt']['sha256']:
            return json.loads(Path(row['files']['manifest.json']['path']).read_text())['identity']
    return current


def effective_history(out):
    out=Path(out);prefix=[]
    if (out/'resume-source.json').exists():
        r=records.bound(records.ref(out/'resume-source.json'))['row']['files']['history.jsonl']
        if records.sha(r['path'])!=r['sha256']:raise ValueError('old history prefix changed')
        prefix=[json.loads(x) for x in Path(r['path']).read_text().splitlines()]
    suffix=[json.loads(x) for x in (out/'history.jsonl').read_text().splitlines()] if (out/'history.jsonl').exists() else []
    return prefix+suffix,len(prefix)


def technical_task(c,t,out,a,original=False):
    """No tensor load and no test re-evaluation. Saved JSON and bound best SHA."""
    out=Path(out);names=('result.json','manifest.json','history.jsonl','budget.json','runtime.json','best.pt','last.pt')
    refs={n:records.ref(out/n) for n in names}
    for n in names:
        if (out/n).is_symlink():raise ValueError('artifact symlink')
    result=json.loads((out/'result.json').read_text());m=json.loads((out/'manifest.json').read_text())
    history,prefix=effective_history(out);best=records.validate_history(c,t,history)
    if m['task']!=t or m['profile']!=profile(c,t) or m['identity']['protocol_sha']!=digest(c) or m['identity']['profile_sha']!=digest(profile(c,t)) or m['identity']['data_sha']!=a['data_bindings'][t['dataset']][t['id']]:raise ValueError('technical result manifest task/data/profile')
    commit=records.BASE if original else a['worker_execution_commit']
    if m['identity']['commit']!=commit or result.get('commit')!=commit or result['id']!=t['id'] or result['best_epoch']!=best.epoch or any(not math.isfinite(result[k]) for k in ('mse','mae','sse','sae')):raise ValueError('result actual execution/best/finite')
    if len(history)<profile(c,t)['training']['epochs'] and not best.stopped:raise ValueError('task not actually ended by epoch cap/early stop')
    if not original and (m['identity']['source']!=a['code'] or any(m['identity'].get(k)!=a[k] for k in ('science_baseline_commit','worker_execution_commit','controller_execution_commit','science_computation_fingerprint'))):raise ValueError('actual worker/controller scientific inheritance')
    test=result['final_test']
    if test!=dict(calls=1,selected='best.pt',sha256=refs['best.pt']['sha256'],epoch=best.epoch):raise ValueError('exact one best final test')
    b=json.loads((out/'budget.json').read_text());runtime=json.loads((out/'runtime.json').read_text());ar=step_arithmetic(c,t);p=profile(c,t)
    epochs=len(history)-prefix
    counts=dict(adam=epochs*ar['train_batches'],backward=epochs*ar['train_batches'],forward=epochs*(ar['train_batches']+math.ceil(ar['validation_windows']/p['training']['eval_batch']))+math.ceil(ar['test_windows_arithmetic_only']/p['training']['eval_batch']))
    if b['counts']!=counts or runtime.get('error')is not None or runtime['task']!=t['id'] or b['by_pid']!={str(runtime['pid']):counts}:raise ValueError('new/old actual accounting ownership')
    if original:
        row=next(r for r in records.bound(a['inventory_ref'])['rows'] if r['task_id']==t['id'])
        if any(refs[n]['sha256']!=row['files'][n]['sha256'] for n in names):raise ValueError('inherited complete changed')
    for line in (out/'audit.jsonl').read_text().splitlines():
        if json.loads(line).get('event')=='denied':raise ValueError('formal guard denial')
    return dict(task_id=t['id'],sources=refs,epochs=len(history),recovery_epochs=epochs,actual_counts=counts,
                scientific_steps=history[-1]['steps'],worker_execution_commit=commit,result_review='pending',runtime_pid=runtime['pid'])


def run_group(c,a,permit_ref,runtime_ref,model):
    from ch3_runner import GPULock, dump
    from m5_formal_entry import run_configs
    ctx=scope.context(c);root=ctx['control']/model
    if model not in ctx['models'] or root.exists():raise FileExistsError('retained/foreign recovery model group')
    inv=records.bound(a['inventory_ref']) if ctx['stage']=='tmark' else dict(rows=[])
    rows={r['task_id']:r for r in inv['rows']};completed=[]
    with GPULock(c):
        root.mkdir(parents=True,exist_ok=False);dump(root/'controller.json',dict(**identity(os.getpid()),model=model,formal_permit_ref=permit_ref,runtime_admission_ref=runtime_ref))
        try:
            for n,ids in records.pending_waves(scope.formal_waves(c,dict(decisions=a['compact_decisions']),model),inv):
                # Dispatch lock brackets DRAIN/STOP and all spawning, not active work.
                recovery.stop_check(drain=True);recovery.validate_permit_light(c,a)
                cfg=[make_formal_config(c,r,a,permit_ref,runtime_ref) for r in ids]
                measured=run_configs(cfg,root/('wave-'+str(n)),monitor=True)
                if not execution.wave_passed(measured):raise RuntimeError('recovery wave resource/exit failure')
                summaries={r:technical_task(c,task_by_id(c,r),scope.result_path(c,task_by_id(c,r)),a) for r in ids}
                resource_receipt(root/('wave-'+str(n)),ids,summaries)
                records.exclusive(root/('wave-'+str(n))/'technical-handoff.json',dict(task_ids=ids,runs=summaries,process_ref=records.ref(root/('wave-'+str(n))/'process.json'),technical_complete=True,result_review='pending'))
                completed+=ids;dump(root/'progress.json',dict(wave=n,completed=completed))
                recovery.stop_check(drain=True)
            recovery.stop_check(drain=True)
            group_root=ctx['result_root']/('formal-'+model)
            expected={t['id'] for t in c['tasks'] if t['model']==model and rows.get(t['id'],{}).get('classification')!='A'}
            if group_root.exists() and {p.name for p in group_root.iterdir()}!=expected:raise ValueError('foreign/missing/staging recovery model artifact')
            records.exclusive(root/'complete.json',dict(task_ids=[t['id'] for t in c['tasks'] if t['model']==model],dispatched=completed,
                formal_permit_ref=permit_ref,runtime_admission_ref=runtime_ref,protocol_sha=digest(c),
                wave_receipts={p.parent.name:records.ref(p) for p in root.glob('wave-*/technical-handoff.json')},
                technical_complete=True,result_review='pending'))
        except recovery.DrainStop:
            records.exclusive(root/'drain-boundary.json',dict(completed=completed,state='DRAINED',technical_failure=False,resume_eligible=True,unconditional_resume=False));raise
        except BaseException as exc:
            records.exclusive(root/'failure.json',dict(error=repr(exc),completed=completed,automatic_retry=False));raise


def closeout(c,a,effective_ref,attempt):
    """Retryable before seal; never reconstruct a successfully sealed boundary."""
    path=recovery.PACKAGE/'native-time-mark-replacement-boundary-recovery1.json'
    if path.exists():return records.ref(path) if recovery.validate_boundary_light(records.ref(path)) else None
    recovery.stop_check(drain=True);mapping=records.bound(effective_ref);runs={}
    if list(mapping['rows'])!=[t['id'] for t in c['tasks']]:
        # Canonical JSON serialization sorts mappings; exact set is authoritative.
        if set(mapping['rows'])!={t['id'] for t in c['tasks']}:raise ValueError('exact effective 87 map')
    for t in c['tasks']:
        r=mapping['rows'][t['id']];runs[t['id']]=technical_task(c,t,r['effective_root'],a,r['origin']=='original_v4_complete')
    wave_refs={};inventory=records.bound(a['inventory_ref'])
    for model in scope.context(c)['models']:
        waves=scope.formal_waves(c,dict(decisions=a['compact_decisions']),model)
        for n,ids in records.pending_waves(waves,inventory):
            wave_root=scope.context(c)['control']/model/('wave-'+str(n))
            resource_receipt(wave_root,ids,runs);wave_refs[model+'/'+str(n)]=records.ref(wave_root/'technical-handoff.json')
    # A run resources are still the original v4 waves, never relabeled as new.
    token=recovery._ACTIVE.set(None)
    try:old_control=scope.context(c)['control']
    finally:recovery._ACTIVE.reset(token)
    complete={r['task_id'] for r in inventory['rows'] if r['classification']=='A'}
    for model in scope.MODELS:
        for n,ids in enumerate(scope.formal_waves(c,dict(decisions=a['compact_decisions']),model)):
            if set(ids)<=complete:
                wave_root=old_control/model/('wave-'+str(n));resource_receipt(wave_root,ids,runs)
                wave_refs['original/'+model+'/'+str(n)]=records.ref(wave_root/'process.json')
    recovery.stop_check(drain=True)
    return records.exclusive(path,dict(purpose='native_recovery_replacement_boundary_v1',scope=recovery.ID,
        task_ids=[t['id'] for t in c['tasks']],protocol_sha=digest(c),science_baseline_commit=records.BASE,
        controller_execution_commit=a['controller_execution_commit'],effective_map_ref=effective_ref,
        old_retirement_boundary_ref=a['old_retirement_boundary_ref'],runs=runs,
        wave_resource_refs=wave_refs,
        successful_closeout_attempt=attempt,technical_complete=True,result_review='pending'))


def resource_receipt(root,ids,runs):
    root=Path(root);p=json.loads((root/'process.json').read_text())
    pids={str(runs[r]['runtime_pid']) for r in ids}
    if not execution.wave_passed(p) or p.get('exit_transitions_resolved')is not True or p.get('process_attribution')!='Measured' or set(p['process_peaks'])!=pids or any(p['process_peaks'][pid] is None for pid in pids):raise ValueError('technical wave resource/ownership/exit')
    with (root/'memory.jsonl').open() as handle:first=json.loads(next(handle))
    for pid in pids:
        v=dict(pid=int(pid),start_ticks=str(first['owned_pid_metadata'][pid]['start_ticks']))
        if same(v):raise ValueError('owned worker original instance remains live')
    return records.ref(root/'process.json')
