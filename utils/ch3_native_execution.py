"""Restricted successor workers and saved-evidence technical audit, no raw worker."""
import json
import math
import os
import sys
import time
from pathlib import Path
from utils.ch3_contract import ROOT,digest,profile,task_by_id,step_arithmetic,BestState
from utils import ch3_m_tasks as source
from utils import ch3_native_tasks as scope
from utils.ch3_m_execution import wave_ids,wave_passed,resource_fallback,gpu_environment_reasons


def read_config(s):
    if s.get('type1_scope'):
        from utils.ch3_type1_execution import read_config as handler
        return handler(s)
    if s.get('unified_scope'):
        from utils.ch3_baseline_unified_execution import read_config
        return read_config(s)
    if s.get('recovery_scope'):
        from utils.ch3_native_recovery_execution import read_config
        return read_config(s)
    from utils.ch3_contract import read_profiles,validate_manifest
    allowed={str(scope.PROFILE_FILE),str(ROOT/'configs/ch3_formal_profiles.json')}
    path=s.get('protocol_file')
    if path not in allowed or s.get('successor_scope') not in (scope.ID+'-tmark-probe',scope.ID+'-tmark-formal',scope.ID+'-m-probe',scope.ID+'-m-formal'):
        raise PermissionError('exact successor protocol file/scope before fixture')
    c=validate_manifest(read_profiles(Path(path)))
    ctx=scope.context(c)
    if path!=str(ctx['protocol_file']) or s['successor_scope'] not in (ctx['probe_scope'],ctx['formal_scope']) or s.get('protocol_sha')!=digest(c):
        raise PermissionError('successor protocol/scope mismatch')
    return c


def roots(c,probe):
    ctx=scope.context(c)
    return ctx['probe_root'] if probe else ctx['control']


def confirmation_endpoint(c,s):
    """Only this exact replacement probe keeps the already approved 6/10/6 path."""
    if not (c.get('type1_followup') or 'native_replacement' in c or c.get('baseline_unified',{}).get('stage')=='MS') or s.get('purpose')!='ch3_probe' or s.get('successor_scope')!=scope.context(c)['probe_scope']:return False
    t=task_by_id(c,s['task'])
    return t['model']=='TimeMixer' and t['dataset']=='UrbanEV'


def group_for(c,run):
    for g in scope.probe_groups(c):
        if run in g['representatives']:return g
    raise ValueError('not an exact successor representative')


def validate_probe_memory(tr,producer):
    from ch3_runner import memory_growth_review
    mode=producer.get('resource_mode')
    if mode is not None:
        if mode!='exclusive_gpu_whole_card_v1' or any(tr.get(k)!=producer[k]for k in('resource_mode','resource_contract_ref')):raise ValueError('probe memory resource binding')
        if not {'allocated','reserved'}.issubset(tr) or tr['allocated']is not None or tr['reserved']is not None or tr.get('endpoint_resources',{}).get('allocated')is not None:raise ValueError('worker GPU memory is not collected')
    review=memory_growth_review(tr['memory'],mode)
    if tr.get('memory_review')!=review or review['blocked'] or review['needs_long_window']:raise ValueError('saved probe memory screen')
    return review


def wave_resource_identities(value,producer,memory,ids):
    if producer.get('resource_mode')=='exclusive_gpu_whole_card_v1':
        from utils.ch3_native_recovery_records import validate_whole_card_receipt
        return validate_whole_card_receipt(value,memory,producer,ids)
    if producer.get('resource_mode')is not None or value.get('resource_mode')is not None:raise ValueError('unbound resource mode')
    if not wave_passed(value) or value.get('exit_transitions_resolved')is not True or value.get('process_attribution')!='Measured':raise ValueError('historical resource attribution/exit gate')
    with Path(memory).open()as f:first=json.loads(next(f))
    return {pid:dict(pid=int(pid),start_ticks=str(row['start_ticks']))for pid,row in first['owned_pid_metadata'].items()}


def decision_for(c,report,t):
    if c.get('type1_followup'):
        from utils.ch3_type1_tasks import decision_for as handler
        return handler(c,report,t)
    if 'baseline_unified' in c:
        from utils.ch3_baseline_unified_tasks import decision_for as unified_decision
        return unified_decision(c,report,t)
    if 'native_replacement' not in c:return report['decisions'][t['group']]
    for g in scope.probe_groups(c):
        if any(t['id'] in ids for ids in g['coverage'].values()):return report['decisions'][g['id']]
    raise ValueError('missing native computational coverage')


def exact_path(c,t,probe,phase,out):
    ctx=scope.context(c)
    if probe:
        if phase not in ('serial','q4','q2'):raise ValueError('exact successor phase')
        expected=ctx['probe_root']/group_for(c,t['id'])['id']/phase/t['id']
    else:expected=scope.result_path(c,t)
    if Path(out)!=expected or expected.is_symlink() or expected.resolve()!=expected:raise ValueError('exact successor output root/task')
    return expected


def authorization_reasons(c,a,probe=False,worker=False):
    if c.get('type1_followup'):
        from utils.ch3_type1_execution import authorization_reasons as handler
        return handler(c,a,probe,worker)
    if 'baseline_unified' in c:
        from utils.ch3_baseline_unified_execution import authorization_reasons
        return authorization_reasons(c,a,probe,worker)
    if a and a.get('recovery_scope'):
        from utils.ch3_native_recovery import validate_permit_light
        reasons=gpu_environment_reasons()
        try:validate_permit_light(c,a,probe,worker)
        except (OSError,ValueError,KeyError,TypeError,PermissionError) as exc:reasons.append(str(exc))
        return list(dict.fromkeys(reasons))
    from utils.ch3_native_chain import validate_permit
    reasons=gpu_environment_reasons()
    try:validate_permit(c,a,probe,worker=worker)
    except (OSError,ValueError,KeyError,TypeError,PermissionError) as exc:reasons.append(str(exc))
    return list(dict.fromkeys(reasons))


def metadata_files(c,a):
    if c.get('type1_followup'):
        from utils.ch3_type1_execution import metadata_files as handler
        return handler(c,a)
    if 'baseline_unified' in c:
        from utils.ch3_baseline_unified_execution import metadata_files
        return metadata_files(c,a)
    if a.get('recovery_scope'):
        from utils.ch3_native_recovery import metadata_files
        return metadata_files(c,a)
    from utils.ch3_native_chain import PREPARATION
    result={r['path']:r['sha256'] for r in PREPARATION.values()}
    from utils.ch3_ms_retirement import RECEIPT
    retired=source.bound(RECEIPT)
    for ref in (RECEIPT,retired['run_inventory'],retired['production']['config']):
        result[ref['path']]=ref['sha256']
    for key in ('old_boundary','replacement_boundary','resource_report'):
        if a.get(key):result.update({a[key]['path']:a[key]['sha256']})
    if a.get('resource_report'):
        r=source.bound(a['resource_report'])
        for key in ('probe_complete','probe_permit'):
            result[r[key]['path']]=r[key]['sha256']
        raw=source.bound(r['probe_complete'])
        result[raw['approval']['path']]=raw['approval']['sha256']
    return result


def validate_worker(c,s):
    if c.get('type1_followup'):
        from utils.ch3_type1_execution import validate_worker as handler
        return handler(c,s)
    if s.get('unified_scope'):
        from utils.ch3_baseline_unified_execution import validate_worker
        return validate_worker(c,s)
    if s.get('recovery_scope'):
        from utils.ch3_native_recovery_execution import validate_worker
        return validate_worker(c,s)
    from m5_formal_entry import repository_files
    probe=s.get('purpose')=='ch3_probe';ctx=scope.context(c);t=task_by_id(c,s['task'])
    if s.get('purpose')not in ('ch3_probe','ch3_formal') or s.get('resume')is not False or s.get('kernel_probe')is not False or s.get('device')!='cuda:0':raise PermissionError('successor fresh guarded GPU worker only')
    if s['successor_scope']!=(ctx['probe_scope'] if probe else ctx['formal_scope']) or s['bound_files']!=repository_files() or s['ids']!=[t['id']]:raise ValueError('successor source/task scope')
    exact_path(c,t,probe,s.get('successor_phase'),s['output'])
    fixture=scope.PACKAGE/'fixtures';session=ctx['probe_root'] if probe else ctx['result_root']
    if s['artifact_root']!=s['output'] or s['session_root']!=str(session) or s['fixture_root']!=str(fixture) or os.environ.get('TMPDIR')!=str(fixture) or not fixture.is_dir():raise ValueError('successor fixture/cache/output binding')
    expected_prefix={}
    if not probe:
        d=c['datasets'][t['dataset']]
        expected_prefix={str(Path(d['path'])/f):c['urban_folds'][t['fold']-1][2] for f in ('volume.csv','e_price.csv','s_price.csv','weather_central.csv')} if t['dataset']=='UrbanEV' else {d['path']:d['endpoints'][2]}
    limits=dict(**scope.worker_counts(c,t),seconds=1800) if probe else dict(adam=None,forward=None,backward=None,seconds=None)
    if s['prefix_files']!=expected_prefix or s['limits']!=limits or s['author_files']!=c['sources'].get(t['model'],{}).get('files',{}) or s['metadata_files']!=metadata_files(c,s['approval']):raise ValueError('successor data/limits/author/metadata binding')
    reasons=authorization_reasons(c,s['approval'],probe,worker=True)
    if reasons:raise PermissionError('; '.join(reasons))


def make_config(c,purpose,out,*,task,approval,artifact_root=None,resume=False):
    if c.get('type1_followup'):
        from utils.ch3_type1_execution import make_config as handler
        return handler(c,purpose,out,task=task,approval=approval,artifact_root=artifact_root,resume=resume)
    if 'baseline_unified' in c:
        from utils.ch3_baseline_unified_execution import make_config
        return make_config(c,purpose,out,task=task,approval=approval,artifact_root=artifact_root,resume=resume)
    if purpose=='ch3_formal' and approval and approval.get('recovery_scope'):
        raise PermissionError('recovery formal config requires explicit runtime admission reference')
    from ch3_runner import dump
    from m5_formal_entry import repository_files
    probe=purpose=='ch3_probe';t=task_by_id(c,task);ctx=scope.context(c);out=Path(out);phase=out.parent.name if probe else None
    exact_path(c,t,probe,phase,out)
    reasons=authorization_reasons(c,approval,probe)
    if reasons or resume:raise PermissionError('; '.join(reasons) or 'no successor resume/fresh retry')
    prefix={}
    if not probe:
        d=c['datasets'][t['dataset']]
        prefix={str(Path(d['path'])/f):c['urban_folds'][t['fold']-1][2] for f in ('volume.csv','e_price.csv','s_price.csv','weather_central.csv')} if t['dataset']=='UrbanEV' else {d['path']:d['endpoints'][2]}
    limits=dict(**scope.worker_counts(c,t),seconds=1800) if probe else dict(adam=None,forward=None,backward=None,seconds=None)
    out.mkdir(parents=True,exist_ok=False)
    s=dict(version='restricted-regression-minimal-v3',repo=str(ROOT),tool_root=str(ROOT/'tools/restricted_regression'),purpose=purpose,task=task,case=None,ids=[task],protocol_file=str(ctx['protocol_file']),protocol_sha=digest(c),successor_scope=ctx['probe_scope'] if probe else ctx['formal_scope'],successor_phase=phase,session_root=str(ctx['probe_root'] if probe else ctx['result_root']),fixture_root=str(ctx.get('fixture',scope.PACKAGE/'fixtures')),audit_log=str(out/'audit.jsonl'),budget_file=str(out/'budget.json'),output=str(out),limits=limits,bound_files=repository_files(),author_roots=[v['repository'] for v in c['sources'].values()],author_files=c['sources'].get(t['model'],{}).get('files',{}),prefix_files=prefix,metadata_files=metadata_files(c,approval),forbidden_roots=[],device='cuda:0',approval=approval,artifact_root=str(out),resume=False,kernel_probe=False)
    if approval.get('recovery_scope'):
        s['recovery_scope']=approval['recovery_scope']
        from utils.ch3_native_recovery import PACKAGE
        s['probe_permit_ref']=source.ref(PACKAGE/'m-probe-permit.json')
        s['metadata_files'][s['probe_permit_ref']['path']]=s['probe_permit_ref']['sha256']
    for name in ('cache/torch/kernels','mpl','cuda-cache'):(out/name).mkdir(parents=True,exist_ok=True)
    dump(out/'config.json',s)
    from resource_budget import initialize
    initialize(s['budget_file'],purpose,limits)
    return s


def validate_wave(c,configs,out):
    if c.get('type1_followup'):
        from utils.ch3_type1_execution import validate_wave as handler
        return handler(c,configs,out)
    if configs and configs[0].get('unified_scope'):
        from utils.ch3_baseline_unified_execution import validate_wave
        return validate_wave(c,configs,out)
    if configs and configs[0].get('recovery_scope'):
        from utils.ch3_native_recovery_execution import validate_wave
        return validate_wave(c,configs,out)
    if not configs:raise ValueError('empty successor wave')
    probe=configs[0]['purpose']=='ch3_probe';ids=[s['task'] for s in configs];ctx=scope.context(c)
    if any(s['approval']!=configs[0]['approval'] or s['successor_scope']!=configs[0]['successor_scope'] for s in configs) or len(set(ids))!=len(ids):raise ValueError('mixed successor scopes/permits')
    if probe:
        g=group_for(c,ids[0]);phase=configs[0]['successor_phase'];reps=g['representatives']
        allowed=[[r] for r in reps] if phase=='serial' else wave_ids(reps,int(phase[1:])) if phase in ('q4','q2') and int(phase[1:]) in scope.attempt_widths(g) else []
    else:
        t=task_by_id(c,ids[0]);report=source.bound(configs[0]['approval']['resource_report']);allowed=scope.formal_waves(c,report,t['model'])
    if ids not in allowed or not Path(out).resolve().is_relative_to(roots(c,probe)):raise ValueError('successor fixed wave scope')
    for s in configs:exact_path(c,task_by_id(c,s['task']),probe,s.get('successor_phase'),s['output'])
    return roots(c,probe)/'STOP'


def compare(c,t,x,y):
    from ch3_runner import compare_probe_trajectories,_compare_full_numeric_files
    # Authorize both producers' exact payload paths before the numeric reader.
    for value in (x,y):
        for point in value.get('M_full_state_trace',[]):
            for key in ('schema_file','data_file'):
                p=Path(point[key])
                if p.is_symlink():raise ValueError('successor full state payload namespace')
                if not p.resolve().is_relative_to(scope.context(c)['probe_root']):
                    from utils.ch3_probe_schema_recovery import retained_payload
                    if not retained_payload(c,point):raise ValueError('successor full state payload namespace')
    row=compare_probe_trajectories(c,t,x,y)
    if c.get('type1_followup') and profile(c,t)['training']['scheduler']['name']!='OneCycleLR':
        p=profile(c,t)
        if p['training']['scheduler']['name']=='type1_horizon_scaled_v1':
            from utils.ch3_type1_scaled import validate_probe_trace
        else:
            from utils.ch3_type1 import validate_probe_trace
        for value in (x,y):validate_probe_trace(p['training']['scheduler'],value.get('scheduler_trace',[]),p['training']['betas'])
        if x['scheduler_trace']!=y['scheduler_trace']:raise ValueError('serial/parallel type1 scheduler exact')
    elif 'baseline_unified' in c:
        for value in (x,y):
            traces=value.get('scheduler_trace',[])
            if len(traces)!=6:raise ValueError('six scheduler/LR/beta1 identities required')
            for i,s in enumerate(traces,1):
                cfg=profile(c,t)['training']['scheduler'];state=s['scheduler']
                if s['config']!=cfg or s['config_sha']!=digest(cfg) or s['updates']!=i or state['last_epoch']!=i or state['_step_count']!=i+1 or state['total_steps']!=cfg['epochs']*cfg['steps_per_epoch']:raise ValueError('scheduler profile/step identity')
        if x['scheduler_trace']!=y['scheduler_trace']:raise ValueError('serial/parallel scheduler LR/beta identity must be exact')
    if any(len(v.get('M_full_state_trace',[]))!=6 for v in (x,y)):raise ValueError('six full state/gradient/Adam snapshots')
    from utils.ch3_contract import numeric_probe_policy
    rule=numeric_probe_policy(c,t)
    generic_full=rule is not None and rule.get('kind')=='full_float_state'
    if generic_full and any(v.get('full_numeric_trace')!=v['M_full_state_trace'] for v in (x,y)):
        raise ValueError('generic and M full-state payload identities differ')
    if rule is not None and rule.get('kind') not in ('full_float_state','named_tensor'):
        raise ValueError('unsupported numerical policy branch')
    for left,right in zip(x['M_full_state_trace'],y['M_full_state_trace']):
        if not generic_full and not _compare_full_numeric_files(rule or dict(state_atol=0),left,right)['passed']:row['passed']=False
    if (c.get('type1_followup') or 'native_replacement' in c or c.get('baseline_unified',{}).get('stage')=='MS') and t['model']=='TimeMixer' and t['dataset']=='UrbanEV':
        from utils.ch3_urban_capture import compare_traces
        from utils.ch3_urban_confirmation import compare_measured
        extra=compare_measured(c,t,x,y,compare_traces(x,y,scope.context(c)['probe_root']))
        row['endpoint_gate']=extra;row['passed']=row['passed'] and extra['passed']
    return row


def run_probe(c,a):
    from ch3_runner import dump
    from m5_formal_entry import make_config,run_configs
    ctx=scope.context(c);root=ctx['probe_root'];zero=dict(adam=0,forward=0,backward=0)
    budget=dict(caps=ctx['caps'],reserved=dict(zero),actual=dict(zero),refund=False);decisions={};evidence={};artifacts={}
    seed=None;serial_check=False
    if a.get('probe_recovery_ref'):
        from utils.ch3_probe_schema_recovery import current_recovery
        recovery=current_recovery()
        if c['baseline_unified']['stage']=='M_BASE' or c['baseline_unified']['stage']in getattr(recovery,'SEED_STAGES',()):
            seed=recovery.load_seed(c,a);budget=seed['budget'];decisions=seed['decisions'];evidence=seed['evidence'];artifacts=seed['artifacts']
            serial_check=getattr(recovery,'SERIAL_CHECK',False)
            if serial_check and(c['baseline_unified']['stage']!='PATCH_ENC1' or a.get('execution_attempt')!='PATCHTST-depth-Urban6-exit-r5-worker-guard-serial-check'):raise PermissionError('precise single serial acceptance context')
    def wave(g,phase,ids,n):
        key=g['id']+'/'+phase+'/'+str(n)
        if seed and key in seed['evidence']:
            entry=seed['evidence'][key]
            if entry['task_ids']!=ids:raise ValueError('retained wave membership differs')
            return source.bound(entry['process'])
        if c.get('type1_followup'):
            from utils.ch3_type1_chain import stop_check
            stop_check()
        elif 'baseline_unified' in c:
            from utils.ch3_baseline_unified_chain import stop_check
            stop_check()
        if a.get('recovery_scope'):
            from utils.ch3_native_recovery import stop_check
            stop_check(drain=True)
        if (root/'STOP').exists():raise InterruptedError('successor probe STOP')
        request={k:sum(scope.worker_counts(c,task_by_id(c,r))[k] for r in ids) for k in zero}
        if any(budget['reserved'][k]+request[k]>ctx['caps'][k] for k in zero):raise ValueError('probe budget before dispatch')
        for k in zero:budget['reserved'][k]+=request[k]
        dump(root/'budget.json',budget);cfg=[];location=root/g['id']/phase/('wave-'+str(n))
        try:
            for r in ids:cfg.append(make_config(c,'ch3_probe',root/g['id']/phase/r,task=r,approval=a))
            measured=run_configs(cfg,location,monitor=True)
            evidence[g['id']+'/'+phase+'/'+str(n)]=dict(process=source.ref(location/'process.json'),task_ids=ids)
            return measured
        finally:
            for s in cfg:
                b=json.loads(Path(s['budget_file']).read_text())
                for k in zero:budget['actual'][k]+=b['counts'][k]
                d=Path(s['output'])
                for p in d.iterdir():
                    if p.is_file():artifacts[str(p)]=source.ref(p)
                if (d/'trajectory.json').exists():
                    tr=json.loads((d/'trajectory.json').read_text())
                    for point in tr.get('M_full_state_trace',[])+tr.get('urban_diagnostic_trace',[]):
                        for key in ('schema_file','data_file','meta_file'):
                            if key in point:
                                p=Path(point[key])
                                if p.is_symlink() or not p.resolve().is_relative_to(d):raise ValueError('worker payload namespace')
                                artifacts[str(p)]=source.ref(p)
            if (location/'memory.jsonl').exists():artifacts[str(location/'memory.jsonl')]=source.ref(location/'memory.jsonl')
            if seed:budget['new_actual']={k:budget['actual'][k]-budget['historical_actual'][k]for k in zero}
            dump(root/'budget.json',budget)
    try:
        for g in scope.probe_groups(c):
            if seed and g['id']in decisions:continue
            serial=[];traces={};reps=g['representatives']
            for n,r in enumerate(reps):
                key=g['id']+'/serial/'+str(n);retained=bool(seed and key in seed['evidence'])
                v=wave(g,'serial',[r],n);serial.append(v)
                if not wave_passed(v):raise RuntimeError('serial technical gate failed; no fallback')
                path=root/g['id']/'serial'/r/'trajectory.json'
                if retained:path=Path(seed['evidence'][key]['process']['path']).parent.parent/r/'trajectory.json'
                tr=json.loads(path.read_text())
                self_check=None if retained else compare(c,task_by_id(c,r),tr,tr)
                if tr.get('finite')is not True or(self_check is not None and not self_check['passed']):raise ValueError('serial finite/state gate')
                traces[r]=tr
                if serial_check:
                    if g!=scope.probe_groups(c)[0] or n!=0 or r!=recovery.SERIAL_TASK or tr.get('steps')!=6 or profile(c,task_by_id(c,r))['training']['batch']!=128:raise ValueError('fixed one-worker six-step shape')
                    known=wave_resource_identities(v,a,root/g['id']/'serial/wave-0/memory.jsonl',[r]);runtime=json.loads((path.parent/'runtime.json').read_text());pid=str(runtime['pid'])
                    if pid not in known or runtime['error']is not None:raise ValueError('complete owned serial runtime')
                    validate_probe_memory(tr,a)
                    body=dict(purpose='native_single_serial_acceptance_v1',serial_check_passed=True,task_id=r,approval=source.ref(ctx['control']/'probe-permit.json'),scope=ctx['probe_scope'],budget=budget,evidence=evidence,artifacts=artifacts,decisions={},serial_self_check=self_check,worker_identity=dict(pid=int(pid),start_ticks=known[pid]['start_ticks']),probe_recovery_ref=a.get('probe_recovery_ref'),**{k:a[k]for k in ('commit','code','environment','hardware','source_states','protocol_sha')})
                    for field in('resource_mode','resource_contract_ref'):
                        if field in a:body[field]=a[field]
                    if a.get('resource_mode')is not None:body['worker_identity']=known[pid]
                    from utils.ch3_native_recovery_records import exclusive
                    exclusive(root/'serial-check.json',body);return body
            q=1;attempts=[];parallel=[];comparisons=[]
            if g['planned_q']>1:
                for q in scope.attempt_widths(g):
                    parallel=[];comparisons=[];failed=False
                    for n,ids in enumerate(wave_ids(reps,q)):
                        v=wave(g,'q'+str(q),ids,n);parallel.append(v)
                        if not wave_passed(v):
                            if not resource_fallback(v):raise RuntimeError('nonresource/unknown probe failure')
                            failed=True;break
                        for r in ids:
                            row=compare(c,task_by_id(c,r),traces[r],json.loads((root/g['id']/('q'+str(q))/r/'trajectory.json').read_text()));comparisons.append(row)
                            if not row['passed']:raise ValueError('numeric gate failure; no resource fallback')
                    attempts.append(dict(q=q,waves=parallel,resource_failed=failed))
                    if failed:
                        if q==2:q=1
                        continue
                    if sum(v['elapsed'] for v in parallel)>=sum(v['elapsed'] for v in serial):raise ValueError('short-package makespan benefit failed')
                    break
            decisions[g['id']]=dict(status='Passed',concurrency=q,serial=serial,parallel=parallel,attempts=attempts,numerical_comparisons=comparisons,coverage=g.get('coverage'),makespan_scope='captured synthetic short package; not formal training speedup')
            dump(root/'progress.json',dict(decisions=decisions,budget=budget))
        report=dict(purpose='native_successor_probe_complete_v1',execution_complete=True,reviewed=False,manual_review=False,admission_granted=False,scope=ctx['probe_scope'],plan=scope.plan(c),approval=source.ref(root/'approval.json'),decisions=decisions,budget=budget,evidence=evidence,artifacts=artifacts,**{k:a[k] for k in ('commit','protocol_sha','code','environment','hardware')})
        for field in('resource_mode','resource_contract_ref'):
            if field in a:report[field]=a[field]
        if seed:
            report['probe_recovery_ref']=a['probe_recovery_ref']
            if seed.get('single_serial_source'):report['single_serial_source']=seed['single_serial_source']
        # Recovery, unified and type1 retain immediate group gates; one final
        # saved-evidence replay belongs to their AUTO_AUDIT. Legacy behavior stays.
        if not a.get('recovery_scope') and not a.get('unified_scope') and not a.get('type1_scope'):validate_probe_completion(c,report)
        dump(root/'complete.json',report);return report
    except BaseException as exc:
        if a.get('recovery_scope'):
            from utils.ch3_native_recovery import DrainStop
            if isinstance(exc,DrainStop):
                dump(root/'drain-boundary.json',dict(decisions=decisions,budget=budget,evidence=evidence,artifacts=artifacts,
                    technical_failure=False,resume_eligible=True,unconditional_resume=False));raise
        dump(root/'failure.json',dict(error=repr(exc),decisions=decisions,budget=budget,evidence=evidence,artifacts=artifacts,automatic_retry=False));raise


def validate_probe_completion(c,r):
    ctx=scope.context(c);root=ctx['probe_root'];groups=scope.probe_groups(c)
    if r.get('purpose')!='native_successor_probe_complete_v1' or r.get('scope')!=ctx['probe_scope'] or r.get('plan')!=scope.plan(c) or r.get('execution_complete')is not True or set(r['decisions'])!={g['id'] for g in groups}:raise ValueError('actual exact probe completion')
    if (root/'STOP').exists() or (root/'failure.json').exists():raise ValueError('retained probe failure/STOP')
    if r['approval']!=source.ref(root/'approval.json'):raise ValueError('actual probe permit path/SHA')
    a=source.bound(r['approval'])
    for field in('resource_mode','resource_contract_ref'):
        if r.get(field)!=a.get(field):raise ValueError('probe resource policy/permit binding')
    from utils.ch3_probe_schema_recovery import current_recovery
    recovery=current_recovery()
    readonly=recovery.retained_refs(r)
    for k in ('commit','protocol_sha','code','environment','hardware'):
        if r[k]!=a[k]:raise ValueError('probe actual source mismatch')
    expected={};reserved=dict(adam=0,forward=0,backward=0);actual=dict(reserved)
    for g in groups:
        d=r['decisions'][g['id']];reps=g['representatives'];q=d['concurrency'];attempts=d['attempts']
        if d['status']!='Passed' or type(q)is not int or q not in (1,2,4) or d['coverage']!=g.get('coverage'):raise ValueError('probe final q/coverage')
        widths=list(scope.attempt_widths(g))
        expected_widths=widths[:widths.index(q)+1] if q in widths else widths
        if q>g['planned_q'] or [x['q'] for x in attempts]!=expected_widths:raise ValueError('resource fallback order/history')
        for n,run in enumerate(reps):expected[g['id']+'/serial/'+str(n)]=[run]
        for attempt in attempts:
            waves=attempt['waves'];width=attempt['q'];failed=not wave_passed(waves[-1]) if waves else None
            if not waves or failed!=attempt['resource_failed'] or not all(wave_passed(v) for v in waves[:-1]) or (failed and not resource_fallback(waves[-1])):raise ValueError('resource-only fallback classification')
            if not failed and len(waves)!=len(wave_ids(reps,width)):raise ValueError('incomplete successful attempt')
            if (width==4 and q!=4 and not failed) or (width==2 and (q==1)!=failed):raise ValueError('chosen q inconsistent')
            for n in range(len(waves)):expected[g['id']+'/q'+str(width)+'/'+str(n)]=wave_ids(reps,width)[n]
        if d['parallel']!=(attempts[-1]['waves'] if attempts else []):raise ValueError('final measurements mismatch')
    if set(r['evidence'])!=set(expected):raise ValueError('attempted exact evidence coverage')
    required_artifacts=set()
    for key,ids in expected.items():
        group,phase,n=key.split('/');entry=r['evidence'][key];p=recovery.location(r,key,root/group/phase/('wave-'+n))
        producer=recovery.producer(r,key,a)
        if entry!=dict(process=source.ref(p/'process.json'),task_ids=ids):raise ValueError('process/task evidence binding')
        v=source.bound(entry['process'])
        if not wave_passed(v) and not resource_fallback(v):raise ValueError('failed nonresource evidence')
        memory=p/'memory.jsonl';required_artifacts.add(str(memory))
        identities=wave_resource_identities(v,producer,memory,ids)if wave_passed(v)else None
        for run in ids:
            out=p.parent/run;t=task_by_id(c,run);counts=scope.worker_counts(c,t)
            cfg=source.bound(r['artifacts'][str(out/'config.json')]);b=source.bound(r['artifacts'][str(out/'budget.json')])
            if cfg['task']!=run or cfg['output']!=str(out) or cfg['approval']!=producer or cfg['protocol_sha']!=producer['protocol_sha'] or cfg['successor_scope']!=ctx['probe_scope'] or cfg['successor_phase']!=phase or cfg['prefix_files'] or cfg['limits']!=dict(**counts,seconds=1800):raise ValueError('actual worker scope/permit/config')
            for name in ('config.json','budget.json'):required_artifacts.add(str(out/name))
            if set(b['counts'])!=set(counts) or any(type(b['counts'][k])is not int or not 0<=b['counts'][k]<=counts[k] for k in counts):raise ValueError('actual count range')
            for k in counts:reserved[k]+=counts[k];actual[k]+=b['counts'][k]
            if wave_passed(v):
                if b['counts']!=counts:raise ValueError('exact successful worker costs')
                for name in ('trajectory.json','runtime.json','audit.jsonl'):required_artifacts.add(str(out/name))
                tr=source.bound(r['artifacts'][str(out/'trajectory.json')])
                if tr['id']!=run or tr['profile_sha']!=digest(profile(c,t)) or not tr['finite'] or tr.get('time_mark')!=profile(c,t).get('time_mark') or (not recovery.self_review(r,out/'trajectory.json') and not compare(c,t,tr,tr)['passed']):raise ValueError('serial/self identity/finite')
                runtime=source.bound(r['artifacts'][str(out/'runtime.json')]);pid=str(runtime['pid'])
                if runtime.get('task')!=run or runtime.get('error')is not None or b.get('by_pid')!={pid:counts}:raise ValueError('worker runtime/accounting ownership')
                if pid not in identities:raise ValueError('probe owned control identity/runtime')
                if producer.get('resource_mode')is not None and identities[pid]['task_id']!=run:raise ValueError('probe control PID/task pairing')
                if producer.get('resource_mode')is not None and any(cfg.get(k)!=producer[k]for k in('resource_mode','resource_contract_ref')):raise ValueError('probe worker resource binding')
                if producer.get('resource_mode')is None and(pid not in v.get('process_peaks',{})or v['process_peaks'][pid]is None or not isinstance(v.get('cpu_peaks',{}).get(pid),(int,float))):raise ValueError('historical measured peaks required')
                ticks=identities[pid]['start_ticks']
                from m6_remaining_entry import same
                if same(dict(pid=int(pid),start_ticks=str(ticks))):raise ValueError('successful probe worker original still live')
                if tr.get('threads')!=4 or tr.get('affinity')!=a['hardware']['cpu_affinity']:raise ValueError('probe thread/affinity identity')
                if len(tr.get('trajectory',[]))!=6 or any(not math.isfinite(row['loss']) for row in tr['trajectory']) or any(not math.isfinite(tr['validation'][k]) for k in ('mse','mae','sse','sae')):raise ValueError('probe saved loss/validation finite')
                validate_probe_memory(tr,producer)
                if any(json.loads(line).get('event')=='denied' for line in (out/'audit.jsonl').read_text().splitlines()):raise ValueError('guard denied access cannot be admission')
                for point in tr['M_full_state_trace']+tr.get('urban_diagnostic_trace',[]):
                    for k in ('schema_file','data_file','meta_file'):
                        if k in point:required_artifacts.add(str(Path(point[k])))
    for path in required_artifacts:
        if path not in r['artifacts'] or r['artifacts'][path]!=source.ref(path):raise ValueError('required actual payload/memory SHA missing')
    for path,value in r['artifacts'].items():
        if Path(path).is_symlink() or (not Path(path).resolve().is_relative_to(root) and readonly.get(path)!=value) or value!=source.ref(path):raise ValueError('probe artifact changed')
    for g in groups:
        d=r['decisions'][g['id']];reps=g['representatives'];serial=[source.bound(r['evidence'][g['id']+'/serial/'+str(n)]['process']) for n in range(len(reps))]
        if d['serial']!=serial or not all(wave_passed(v) for v in serial):raise ValueError('serial measured gates')
        reviewed=recovery.group_review(r,g)
        comparisons=[]
        for attempt in d['attempts']:
            comparisons=[];phase='q'+str(attempt['q'])
            actual_waves=[source.bound(r['evidence'][g['id']+'/'+phase+'/'+str(n)]['process']) for n in range(len(attempt['waves']))]
            if actual_waves!=attempt['waves']:raise ValueError('fallback history modified')
            for n,v in enumerate(actual_waves):
                if not wave_passed(v):continue
                for run in wave_ids(reps,attempt['q'])[n]:
                    if reviewed is not None:
                        comparisons.append(reviewed[len(comparisons)]);continue
                    serial_index=reps.index(run);serial_path=recovery.location(r,g['id']+'/serial/'+str(serial_index),root/g['id']/'serial'/('wave-'+str(serial_index))).parent/run/'trajectory.json'
                    x=json.loads(serial_path.read_text());y=json.loads((root/g['id']/phase/run/'trajectory.json').read_text());row=compare(c,task_by_id(c,run),x,y)
                    if not row['passed']:raise ValueError('saved numeric gate failed')
                    comparisons.append(row)
        if d['numerical_comparisons']!=comparisons:raise ValueError('numeric summary not reproducible')
        if d['concurrency']>1 and (not all(wave_passed(v) for v in d['parallel']) or sum(v['elapsed'] for v in d['parallel'])>=sum(v['elapsed'] for v in serial)):raise ValueError('actual parallel resource/makespan')
    extra=recovery.extra_actual(r)if hasattr(recovery,'extra_actual')else dict(adam=0,backward=0,forward=0)
    for k in actual:actual[k]+=extra[k];reserved[k]+=extra[k]
    b=r['budget']
    if any(extra.values())and b.get('diagnostic_actual')!=extra:raise ValueError('registered diagnostic consumption cannot be hidden/reset')
    if b!=json.loads((root/'budget.json').read_text()) or b.get('caps')!=ctx['caps'] or b.get('reserved')!=reserved or b.get('actual')!=actual or b.get('refund')is not False or any(reserved[k]>ctx['caps'][k] or actual[k]>reserved[k] for k in actual):raise ValueError('saved budget/reserved/actual/caps')
    return r


def group_artifact_namespace(c,model):
    """Count exact owned run directories; reject foreign/staging entries before handoff."""
    ctx=scope.context(c);tasks=[t for t in c['tasks'] if t['model']==model]
    root=ctx['result_root']/('formal-'+model)
    if model not in ctx['models'] or root.is_symlink() or root.resolve()!=root or not root.is_dir():raise ValueError('exact completed model result namespace')
    entries=list(root.iterdir())
    if {p.name for p in entries}!={t['id'] for t in tasks} or any(p.is_symlink() or not p.is_dir() for p in entries):raise ValueError('missing/foreign/duplicate/staging model artifact')
    return tasks


def checkpoint_files(out):
    """Presence/ownership/path only; never deserialize or compare best/last values."""
    paths=[Path(out)/name for name in ('best.pt','last.pt')]
    for p in paths:
        if p.is_symlink() or p.resolve()!=p or not p.is_file():raise ValueError('missing/shared checkpoint path')
        st=p.stat()
        if st.st_nlink!=1 or st.st_uid!=os.getuid():raise ValueError('shared/foreign checkpoint owner')
    return paths


def technical_group(c,model,a):
    """JSON/accounting/saved-best hash gate only; no checkpoint load or test replay."""
    from m6_remaining_entry import same
    from utils.ch3_m6 import verify_result
    ctx=scope.context(c);root=ctx['control']/model
    if same(json.loads((root/'controller.json').read_text())) or any((root/name).exists() for name in ('STOP','failure.json')):raise ValueError('group original live/failure/STOP')
    tasks=group_artifact_namespace(c,model);ids=[t['id'] for t in tasks];done=json.loads((root/'complete.json').read_text())
    if done!=dict(task_ids=ids,commit=a['commit'],protocol_sha=digest(c),result_review='pending'):raise ValueError('exact actual group completion')
    sources={};totals=dict(epochs=0,adam=0,forward=0,backward=0);workers={}
    def load(p):
        sources[str(p)]=source.ref(p)
        return source.bound(sources[str(p)])
    for t in tasks:
        out=scope.result_path(c,t);checkpoint_files(out);p=profile(c,t);ar=step_arithmetic(c,t);manifest=load(out/'manifest.json');result=verify_result(c,t['id'],load(out/'result.json'));b=load(out/'budget.json');runtime=load(out/'runtime.json');cfg=load(out/'config.json');i=manifest['identity']
        expected=dict(run_id=t['id'],profile_sha=digest(p),data_sha=digest(manifest['metadata']),commit=a['commit'],protocol_sha=digest(c),source=a['code'],time_mark=p.get('time_mark'),time_mark_protocol=c['native_time_mark']['id'])
        if any(i.get(k)!=v for k,v in expected.items()) or manifest['task']!=t or manifest['profile']!=p or a['data_bindings'][t['dataset']][t['id']]!=i['data_sha'] or cfg['approval']!=a or cfg['resume']is not False:raise ValueError('actual manifest/profile/data/source')
        if 'native_replacement' not in c:
            from utils.ch3_m_summary import verify_result as verify_m
            verify_m(c,t,result)
        elif i.get('parent_run_id')!=t['parent_run_id'] or result.get('parent_run_id')!=t['parent_run_id'] or result.get('revision')!='native-time-mark-v1':raise ValueError('replacement lineage identity')
        hp=out/'history.jsonl';sources[str(hp)]=source.ref(hp);history=[json.loads(line) for line in hp.read_text().splitlines()];best=BestState(p['training']['patience'])
        if not 1<=len(history)<=p['training']['epochs']:raise ValueError('epoch bound')
        channels=p['C'] if p.get('task')=='M' else 1
        for n,row in enumerate(history,1):
            val=row['validation'];elements=ar['validation_windows']*p['pred_len']*channels
            if best.stopped or row['epoch']!=n or row['steps']!=n*ar['train_batches'] or val['elements']!=elements or any(not math.isfinite(val[k]) for k in ('mse','mae','sse','sae')) or val['mse']!=val['sse']/elements or val['mae']!=val['sae']/elements:raise ValueError('history/accounting/selection arithmetic')
            best.update(val['mse'],n)
            if row['best_epoch']!=best.epoch:raise ValueError('tie keeps earliest best')
        if best.epoch!=result['best_epoch'] or (len(history)<p['training']['epochs'] and not best.stopped):raise ValueError('best/stopping')
        steps=history[-1]['steps'];fwd=len(history)*(ar['train_batches']+math.ceil(ar['validation_windows']/p['training']['eval_batch']))+math.ceil(ar['test_windows_arithmetic_only']/p['training']['eval_batch']);counts=dict(adam=steps,backward=steps,forward=fwd)
        if b['counts']!=counts or runtime['error']is not None or b['by_pid']!={str(runtime['pid']):counts}:raise ValueError('actual worker accounting/runtime')
        test=result.get('final_test',{})
        if test.get('calls')!=1 or test.get('selected')!='best.pt' or test.get('epoch')!=best.epoch or test.get('sha256')!=source.sha(out/'best.pt'):raise ValueError('validation-selected one final best test binding')
        workers[t['id']]=str(runtime['pid']);totals['epochs']+=len(history)
        for k in counts:totals[k]+=counts[k]
    report=source.bound(a['resource_report']);waves=scope.formal_waves(c,report,model)
    for n,ids in enumerate(waves):
        d=root/('wave-'+str(n));pr=load(d/'process.json')
        identities=wave_resource_identities(pr,a,d/'memory.jsonl',ids)
        if not wave_passed(pr) or set(identities)!={workers[r]for r in ids}:raise ValueError('formal wave lifecycle/resource')
        if a.get('resource_mode')is None and set(pr['process_peaks'])!={workers[r]for r in ids}:raise ValueError('historical formal process peaks')
        for run in ids:
            pid=workers[run];ticks=identities[pid]['start_ticks']
            if a.get('resource_mode')is not None and identities[pid]['task_id']!=run:raise ValueError('formal control PID/task pairing')
            if same(dict(pid=int(pid),start_ticks=str(ticks))):raise ValueError('formal worker original still active')
    if totals['epochs']>sum(profile(c,t)['training']['epochs'] for t in tasks) or totals['adam']>sum(step_arithmetic(c,t)['max_optimizer_steps'] for t in tasks):raise ValueError('formal total budget')
    return dict(model=model,status='technical-complete',task_ids=[t['id'] for t in tasks],technical_complete=True,result_review='pending',totals=totals,sources=sources,checkpoint_audit='deferred; saved best byte selection bound, no weight load')


def run_group(c,a,model):
    from ch3_runner import GPULock,dump
    from m6_remaining_entry import identity
    from m5_formal_entry import make_config,run_configs
    ctx=scope.context(c);root=ctx['control']/model
    if model not in ctx['models'] or root.exists():raise ValueError('exact fresh successor model group')
    reasons=authorization_reasons(c,a)
    if reasons:raise PermissionError('; '.join(reasons))
    with GPULock(c):
        root.mkdir(parents=True,exist_ok=False);dump(root/'controller.json',dict(**identity(os.getpid()),commit=a['commit'],protocol_sha=digest(c),model=model))
        completed=[];n=0;report=source.bound(a['resource_report'])
        try:
            for ids in scope.formal_waves(c,report,model):
                if (ctx['control']/'STOP').exists():raise InterruptedError('successor formal STOP')
                if authorization_reasons(c,a):raise ValueError('dynamic formal binding changed')
                configs=[make_config(c,'ch3_formal',scope.result_path(c,task_by_id(c,r)),task=r,approval=a) for r in ids]
                measured=run_configs(configs,root/('wave-'+str(n)),monitor=True)
                if not wave_passed(measured):raise RuntimeError('formal technical wave failure; stop remaining')
                completed+=ids;dump(root/'progress.json',dict(wave=n,completed=completed));n+=1
            expected=[t['id'] for t in c['tasks'] if t['model']==model]
            if len(completed)!=len(expected) or set(completed)!=set(expected):raise ValueError('formal exact dispatched set')
            dump(root/'complete.json',dict(task_ids=expected,commit=a['commit'],protocol_sha=digest(c),result_review='pending'))
        except BaseException as exc:dump(root/'failure.json',dict(error=repr(exc),completed=completed,automatic_retry=False));raise
