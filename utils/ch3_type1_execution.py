"""Guarded unified workers; compact formal records never replay probe numerics."""
import json,os,time
from pathlib import Path
from utils.ch3_contract import ROOT,digest,profile,task_by_id,step_arithmetic,BestState
from utils import ch3_type1_tasks as scope
from utils.ch3_native_recovery_records import bound,ref,exclusive,sha


def read_config(s):
    if s.get('probe_schema_recovery_ref'):
        from utils.ch3_probe_schema_recovery import activate_worker
        activate_worker(s['probe_schema_recovery_ref'])
    stage=s.get('unified_stage')
    if stage not in scope.STAGES or s.get('type1_scope')!=scope.ID or s.get('protocol_file')!=str(scope.file(stage)):raise PermissionError('exact unified protocol/stage')
    c=scope.validate(json.loads(scope.file(stage).read_text()))
    if s['protocol_sha']!=digest(c):raise PermissionError('unified protocol changed')
    return c


def authorization_reasons(c,a,probe=False,worker=False):
    from utils.ch3_type1_chain import validate_permit
    from utils.ch3_m_execution import gpu_environment_reasons
    reasons=gpu_environment_reasons()
    try:validate_permit(c,a,probe,worker)
    except (OSError,KeyError,ValueError,PermissionError,TypeError) as exc:reasons.append(str(exc))
    return list(dict.fromkeys(reasons))


def metadata_files(c,a,runtime_ref=None):
    from utils.ch3_type1_chain import ENVIRONMENT_REF
    values=[scope.parent_ref(c['baseline_unified']['stage']),scope.AUTHOR_RECIPE,c['baseline_unified']['data_ref'],a['start_authorization_ref']]
    values.append(ENVIRONMENT_REF)
    if a.get('probe_recovery_ref'):
        from utils.ch3_probe_schema_recovery import current_recovery
        recovery=current_recovery();values += [recovery.SOURCE_REF,recovery.REUSE_REF]
        for name in ('POLICY_REF','ALL_M_POLICY_REF','PRIOR_REUSE_REF','INCLUSION_REF'):
            if hasattr(recovery,name):values.append(getattr(recovery,name))
    values += list(c['baseline_unified'].get('extension_refs',{}).values())
    values += [ref(scope.package(stage)/(stage.lower()+'-plan.json')) for stage in scope.STAGES]
    values += list(a.get('predecessor_boundaries',{}).values())
    for key in ('summary_ref','ms_boundary_ref','upstream_boundary_ref','base287_boundary_ref','round2_boundary_ref'):
        if a.get(key):values.append(a[key])
    if runtime_ref:values.append(runtime_ref)
    # Raw probe payloads and the large probe completion report are deliberately absent.
    return {r['path']:r['sha256'] for r in values}


def worker_permit(s):
    a=bound(s['formal_permit_ref']);data=bound(a['data_binding_ref'])
    return dict(a,data_bindings=data['data_bindings'])


def make_config(c,purpose,out,*,task,approval,artifact_root=None,resume=False,runtime_ref=None):
    from m5_formal_entry import repository_files
    from utils.ch3_native_execution import exact_path
    from utils.ch3_type1_chain import validate_runtime,stop_check
    from ch3_runner import dump
    probe=purpose=='ch3_probe';ctx=scope.context(c);t=task_by_id(c,task);out=Path(out)
    if purpose not in ('ch3_probe','ch3_formal') or resume:raise PermissionError('unified fresh only; old checkpoint cannot resume')
    if not probe and runtime_ref is None:raise PermissionError('owned formal runtime required')
    stop_check()
    reasons=authorization_reasons(c,approval,probe)
    if reasons:raise PermissionError('; '.join(reasons))
    if not probe:validate_runtime(c,runtime_ref,ref(ctx['control']/'formal-permit.json'))
    phase=out.parent.name if probe else None;exact_path(c,t,probe,phase,out)
    d=c['datasets'][t['dataset']];prefix={}
    if not probe:prefix={str(Path(d['path'])/f):c['urban_folds'][t['fold']-1][2] for f in ('volume.csv','e_price.csv','s_price.csv','weather_central.csv')} if t['dataset']=='UrbanEV' else {d['path']:d['endpoints'][2]}
    counts=scope.worker_counts(c,t) if probe else {k:scope.formal_budget(c)['tasks'][t['id']][k] for k in ('adam','forward','backward')}
    limits=dict(counts,seconds=1800 if probe else None)
    s=dict(version='restricted-regression-minimal-v3',repo=str(ROOT),tool_root=str(ROOT/'tools/restricted_regression'),purpose=purpose,
        task=task,case=None,ids=[task],protocol_file=str(ctx['protocol_file']),protocol_sha=digest(c),type1_scope=scope.ID,
        unified_stage=ctx['stage'],successor_scope=ctx['probe_scope'] if probe else ctx['formal_scope'],successor_phase=phase,
        session_root=str(ctx['probe_root'] if probe else ctx['result_root']),fixture_root=str(ctx['fixture']),audit_log=str(out/'audit.jsonl'),
        budget_file=str(out/'budget.json'),output=str(out),limits=limits,bound_files=repository_files(),author_roots=[v['repository'] for v in c['sources'].values()],
        author_files=c['sources'].get(t['model'],{}).get('files',{}),prefix_files=prefix,metadata_files=metadata_files(c,approval,runtime_ref),
        forbidden_roots=[],device='cuda:0',artifact_root=str(out),resume=False,kernel_probe=False)
    if approval.get('probe_recovery_ref'):s['probe_schema_recovery_ref']=approval['probe_recovery_ref']
    if probe:s['approval']=approval
    else:
        s['approval']=None;s['formal_permit_ref']=ref(ctx['control']/'formal-permit.json');s['runtime_admission_ref']=runtime_ref
        s['metadata_files'][s['formal_permit_ref']['path']]=s['formal_permit_ref']['sha256']
    out.mkdir(parents=True,exist_ok=False)
    for name in ('cache/torch/kernels','mpl','cuda-cache'):(out/name).mkdir(parents=True,exist_ok=True)
    dump(out/'config.json',s)
    from resource_budget import initialize
    initialize(s['budget_file'],purpose,limits)
    return s


def validate_worker(c,s):
    from m5_formal_entry import repository_files
    from utils.ch3_native_execution import exact_path
    from utils.ch3_type1_chain import validate_runtime,stop_check
    probe=s.get('purpose')=='ch3_probe';ctx=scope.context(c);t=task_by_id(c,s['task']);a=s.get('approval') if probe else bound(s['formal_permit_ref'])
    if s.get('probe_schema_recovery_ref')!=a.get('probe_recovery_ref'):raise PermissionError('worker/permit recovery identity differs')
    stop_check()
    if s.get('purpose')not in ('ch3_probe','ch3_formal') or s.get('resume')is not False or s.get('kernel_probe')is not False or s.get('device')!='cuda:0':raise PermissionError('exact unified fresh GPU worker')
    if s['successor_scope']!=(ctx['probe_scope'] if probe else ctx['formal_scope']) or s['bound_files']!=repository_files() or s['ids']!=[t['id']]:raise ValueError('unified source/task/scope')
    exact_path(c,t,probe,s.get('successor_phase'),s['output'])
    if s['artifact_root']!=s['output'] or s['session_root']!=str(ctx['probe_root'] if probe else ctx['result_root']) or s['fixture_root']!=str(ctx['fixture']) or os.environ.get('TMPDIR')!=str(ctx['fixture']) or not ctx['fixture'].is_dir():raise ValueError('exact cache/output capability')
    runtime_ref=None if probe else s['runtime_admission_ref']
    if runtime_ref:validate_runtime(c,runtime_ref,s['formal_permit_ref'])
    expected=metadata_files(c,a,runtime_ref)
    if not probe:expected[s['formal_permit_ref']['path']]=s['formal_permit_ref']['sha256']
    d=c['datasets'][t['dataset']];prefix={}
    if not probe:prefix={str(Path(d['path'])/f):c['urban_folds'][t['fold']-1][2] for f in ('volume.csv','e_price.csv','s_price.csv','weather_central.csv')} if t['dataset']=='UrbanEV' else {d['path']:d['endpoints'][2]}
    limits=dict(**scope.worker_counts(c,t),seconds=1800) if probe else dict(**{k:scope.formal_budget(c)['tasks'][t['id']][k]for k in ('adam','forward','backward')},seconds=None)
    if s['prefix_files']!=prefix or s['limits']!=limits or s['author_files']!=c['sources'].get(t['model'],{}).get('files',{}) or s['metadata_files']!=expected:raise ValueError('exact unified data/limits/author/metadata')
    if any(sha(p)!=h for p,h in s['author_files'].items()):raise ValueError('own bound author files changed')
    reasons=authorization_reasons(c,a,probe,True)
    if reasons:raise PermissionError('; '.join(reasons))


def validate_wave(c,configs,out):
    from utils.ch3_native_execution import group_for,exact_path
    from utils.ch3_m_execution import wave_ids
    from utils.ch3_type1_chain import stop_check
    if not configs:raise ValueError('empty unified wave')
    stop_check();probe=configs[0]['purpose']=='ch3_probe';ctx=scope.context(c);ids=[s['task'] for s in configs]
    if len(set(ids))!=len(ids) or any(s['unified_stage']!=ctx['stage'] or s['purpose']!=configs[0]['purpose'] for s in configs):raise ValueError('mixed unified scope')
    if probe:
        g=group_for(c,ids[0]);phase=configs[0]['successor_phase']
        widths=(4,2) if g['planned_q']==4 else (2,) if g['planned_q']==2 else ()
        allowed=[[r] for r in g['representatives']] if phase=='serial' else wave_ids(g['representatives'],int(phase[1:])) if phase in ('q4','q2') and int(phase[1:])in widths else []
        if any(s['approval']!=configs[0]['approval'] for s in configs):raise ValueError('mixed probe permit')
    else:
        a=bound(configs[0]['formal_permit_ref']);report=bound(a['summary_ref'])
        allowed=scope.formal_waves(c,report,task_by_id(c,ids[0])['model'])
        if any(s['formal_permit_ref']!=configs[0]['formal_permit_ref'] or s['runtime_admission_ref']!=configs[0]['runtime_admission_ref'] for s in configs):raise ValueError('mixed formal runtime')
    if ids not in allowed or not Path(out).resolve().is_relative_to(ctx['probe_root'] if probe else ctx['control']):raise ValueError('fixed unified wave membership')
    for s in configs:exact_path(c,task_by_id(c,s['task']),probe,s.get('successor_phase'),s['output'])
    from utils.ch3_type1_chain import CONTROL
    return CONTROL/'STOP'


def technical_group(c,model,a):
    """JSON/file integrity closeout only; never deserialize checkpoint or re-test."""
    from utils.ch3_native_tasks import result_path
    expected=[t for t in c['tasks'] if t['model']==model];root=scope.context(c)['result_root']/('formal-'+model)
    if {p.name for p in root.iterdir()}!={t['id'] for t in expected}:raise ValueError('foreign/duplicate formal artifact')
    rows={}
    for t in expected:
        out=result_path(c,t);p=profile(c,t);a_steps=step_arithmetic(c,t);files=('manifest.json','result.json','history.jsonl','budget.json','runtime.json','best.pt','last.pt')
        refs={f:ref(out/f) for f in files};m=bound(refs['manifest.json']);r=bound(refs['result.json']);b=bound(refs['budget.json']);runtime=bound(refs['runtime.json'])
        history=[json.loads(line) for line in (out/'history.jsonl').read_text().splitlines()]
        best=BestState(p['training']['patience'])
        for i,row in enumerate(history,1):
            if row['epoch']!=i or row['steps']!=i*a_steps['train_batches']:raise ValueError('epoch/history/scheduler step identity')
            val=row['validation'];elements=a_steps['validation_windows']*p['pred_len']*(p['C'] if t['task']=='M' else 1)
            import math
            if best.stopped or val['elements']!=elements or any(not math.isfinite(val[k]) for k in ('mse','mae','sse','sae')) or val['mse']!=val['sse']/elements or val['mae']!=val['sae']/elements:raise ValueError('finite weighted metric/early-stopping contract')
            best.update(row['validation']['mse'],i)
            if row['best_epoch']!=best.epoch:raise ValueError('validation selected best tie policy')
        if not history or (len(history)!=p['training']['epochs'] and not best.stopped):raise ValueError('incomplete epoch contract')
        steps=history[-1]['steps'];test=r.get('final_test',{})
        if m['task']!=t or m['profile']!=p or m['identity']['commit']!=a['commit'] or m['identity']['profile_sha']!=digest(p) or m['identity']['protocol_sha']!=digest(c) or m['identity']['data_sha']!=a['data_bindings'][t['dataset']][t['id']]:raise ValueError('formal manifest profile/data/version')
        if r['id']!=t['id'] or r['commit']!=a['commit'] or r['protocol_sha']!=digest(c) or r['profile_sha']!=digest(p) or r['scientific_protocol']!=c['baseline_unified']['id'] or r['scheduler_updates']!=steps or r['scheduler_sha']!=digest(p['training']['scheduler']):raise ValueError('formal result scheduler identity')
        if t['task']=='M' and (p.get('metric_scope')!='all_channels' or p.get('supervised_channels')!=list(range(p['C'])) or p.get('output_order')!=p['features'] or r.get('metric_scope')!='all_channels' or r.get('elements')!=a_steps['test_windows_arithmetic_only']*p['pred_len']*p['C']):raise ValueError('genuine M all-channel supervision/evaluation/output order')
        import math
        if not all(math.isfinite(r[k]) for k in ('mse','mae')) or test!=dict(calls=1,selected='best.pt',sha256=refs['best.pt']['sha256'],epoch=best.epoch) or r['best_epoch']!=best.epoch:raise ValueError('finite/test-once/selected checkpoint')
        forward=len(history)*(a_steps['train_batches']+math.ceil(a_steps['validation_windows']/p['training']['eval_batch']))+math.ceil(a_steps['test_windows_arithmetic_only']/p['training']['eval_batch'])
        counts=dict(adam=steps,backward=steps,forward=forward)
        if runtime.get('error')is not None or runtime['task']!=t['id'] or b['counts']!=counts or b.get('by_pid')!={str(runtime['pid']):counts} or steps>a_steps['max_optimizer_steps']:raise ValueError('runtime/budget exact accounting')
        rows[t['id']]=refs
    return dict(technical_complete=True,result_review='pending',model=model,task_ids=[t['id'] for t in expected],artifacts=rows)


def run_group(c,a,model,runtime_ref):
    from ch3_runner import GPULock,dump
    from m5_formal_entry import run_configs
    from utils.ch3_native_tasks import result_path
    from utils.ch3_m_execution import wave_passed
    from utils.ch3_type1_chain import stop_check,validate_runtime,CONTROL
    ctx=scope.context(c);summary=bound(a['summary_ref']);waves=scope.formal_waves(c,summary,model);group=ctx['control']/('group-'+model)
    group.mkdir(parents=True,exist_ok=False);done=[]
    with GPULock(c):
        for n,ids in enumerate(waves):
            stop_check();validate_runtime(c,runtime_ref,ref(ctx['control']/'formal-permit.json'))
            configs=[make_config(c,'ch3_formal',result_path(c,task_by_id(c,r)),task=r,approval=a,runtime_ref=runtime_ref) for r in ids]
            measured=run_configs(configs,group/('wave-'+str(n)),monitor=True)
            if not wave_passed(measured):raise RuntimeError('unified formal technical/resource failure')
            stop_check();done+=ids;dump(group/'progress.json',dict(model=model,wave=n,completed=done))
    receipt=technical_group(c,model,dict(a,data_bindings=bound(a['data_binding_ref'])['data_bindings']))
    exclusive(group/'complete.json',receipt)
    return receipt
