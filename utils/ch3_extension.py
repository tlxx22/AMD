"""M6 epf4-timemixer-v1: additive identities, fixed queues, read-only joined index."""
import copy,json,math,hashlib,os,signal
from pathlib import Path
from utils.ch3_contract import digest,profile,step_arithmetic,MODELS
BATCH='epf4-timemixer-v1'
MARKETS=('NP','BE','FR','DE')
PRIMARY=('AMD','J','DLinear','PatchTST','iTransformer','TimeMixer','ModernTCN','TimeXer')
APPEND=('DLinear','iTransformer','ModernTCN','TimeXer')
CATCHUP=('AMD','J','PatchTST','TimeMixer')
ORIGINAL=('UrbanEV','PJM','ETTh1','Weather','ECL','Exchange')

def task(model,domain,fold,h):
    v='F4' if domain=='UrbanEV' else 'MS';g=f'{model}-{domain}-{v}'
    return dict(id=f'{g}-f{fold}-h{h}-s2024',model=model,dataset=domain,input_variant=v,fold=fold,h=h,group=g,profile=g+f'-h{h}',seed=2024)

def expected_tasks(c):
    original=[]
    for model in MODELS:
        domains=['UrbanEV'] if model in ('N','S') else list(ORIGINAL[2:]) if model=='TimeMixer' else list(ORIGINAL)
        for d in domains:
            variants=['F1','F2','F3','F4'] if d=='UrbanEV' and model in ('AMD','J') else ['F4'] if d=='UrbanEV' else ['MS']
            for v in variants:
                for f in range(1,7) if d=='UrbanEV' else [1]:
                    for h in c['datasets'][d]['horizons']:
                        t=task(model,d,f,h);t.update(input_variant=v,group=f'{model}-{d}-{v}',id=f'{model}-{d}-{v}-f{f}-h{h}-s2024',profile=f'{model}-{d}-{v}-h{h}');original.append(t)
    extra=[task(m,d,1,24) for m in PRIMARY for d in MARKETS]
    extra += [task('TimeMixer','PJM',1,24)]+[task('TimeMixer','UrbanEV',f,h) for f in range(1,7) for h in (3,6,9,12)]
    return original+extra

def validate_extension(c):
    from utils.ch3_contract import CONTRACT,generate_groups
    x=c['extension'];tasks=expected_tasks(c)
    if c['contract']!=CONTRACT or x['id']!=BATCH or c['tasks']!=tasks:raise ValueError('extension task registration mismatch')
    ids=[t['id'] for t in tasks]
    if len(ids)!=552 or len(set(ids))!=552 or x['original_ids']!=ids[:495] or x['new_ids']!=ids[495:]:raise ValueError('495+57 partition')
    caught=[t['id'] for m in CATCHUP for d in list(MARKETS)+(['PJM','UrbanEV'] if m=='TimeMixer' else []) for t in tasks[495:] if (t['model'],t['dataset'])==(m,d)]
    app={m:[t['id'] for t in tasks[495:] if t['model']==m] for m in APPEND}
    if x['catchup_ids']!=caught or x['append_ids']!=app or len(caught)!=41:raise ValueError('41/16 queues mismatch')
    if set(caught)&set(sum(app.values(),[])) or set(caught+sum(app.values(),[]))!=set(ids[495:]):raise ValueError('overlapping extension queues')
    from utils.ch3_revision import validate_revision_profiles
    validate_revision_profiles(c)
    for t in tasks[:495]:
        if t['id']not in c.get('timemixer_revision',{}).get('task_ids',[]) and digest(profile(c,t))!=x['original_profile_hashes'].get(t['id']):raise ValueError('original scientific profile changed '+t['id'])
    if sum(profile(c,t)['training']['epochs'] for t in tasks)!=6240:raise ValueError('epoch cap')
    for a,b in zip(c['groups'],generate_groups(tasks)):
        if any(a[k]!=b[k] for k in ('id','task_ids','representatives','q','waves')):raise ValueError('fixed group/wave mismatch')
    if len(c['groups'])!=88:raise ValueError('88 additive groups')
    for d in MARKETS:
        z=c['datasets'][d];n=z['declared_rows']
        if z['endpoints']!=[7*n//10,n-2*n//10,n] or z['T']!=168 or z['horizons']!=[24] or z['target']!='OT' or len(z['features'])!=3 or z['features'][2]!='OT':raise ValueError('EPF split/field contract')
    for t in tasks[495:]:
        p=profile(c,t);tr=p['training'];src=profile(c,next(z for z in tasks[:495] if z['model']==('AMD' if t['model']=='TimeMixer' else t['model']) and z['dataset']=='PJM'))
        if t['dataset']!='UrbanEV' and (p['T'],p['pred_len'],tr['epochs'],tr['patience'])!=(168,24,20,5):raise ValueError('EPF training contract')
        if any(tr[k]!=src['training'][k] for k in ('optimizer','betas','eps','weight_decay','scheduler','dtype','accumulation','workers','threads','seed','search','from_scratch')):raise ValueError('unauthorized training mutation')
    return c

def queue_ids(c,model=None):
    if model is None:return list(c['extension']['catchup_ids'])
    if model not in APPEND:raise ValueError('original completed/active group cannot restart via appended-model entry')
    return [t['id'] for t in c['tasks'][:495] if t['model']==model]+c['extension']['append_ids'][model]

def result_path(c,t):
    if t['id']in c.get('timemixer_revision',{}).get('task_ids',[]):return Path(c['timemixer_revision']['result_root'])/t['id']
    base=Path(c['execution']['evidence'])
    if t['id'] in c['extension']['new_ids']:base=Path(c['extension']['result_root'])
    return base/('formal-'+t['model'])/t['id']

def fixed_waves(c,ids,decisions):
    from ch3_runner import verified_waves
    out=[];pending=list(ids)
    while pending:
        t=next(v for v in c['tasks'] if v['id']==pending[0]);g=next(v for v in c['groups'] if v['id']==t['group']);runs=[]
        while pending and pending[0] in g['task_ids']:runs.append(pending.pop(0))
        decision=decisions.get(g['id'],{});q=decision.get('concurrency')
        if t['dataset'] in MARKETS+('PJM',) and q!=1:raise ValueError('single EPF fit cannot be replicated')
        all_waves=verified_waves(g,decision)
        selected=[w for w in all_waves if all(r in runs for r in w)]
        if [r for w in selected for r in w]!=runs:raise ValueError('queue is not exact fixed-wave subsequence')
        out.extend(selected)
    return out

def joined_index(c,records=()):
    """No filesystem result reads; reviewed receipts required for success/metrics."""
    from utils.ch3_result_index import VerifiedReceipts
    if records and not isinstance(records,VerifiedReceipts):raise ValueError('source-bound accepted receipt loader required')
    by={};valid={t['id']:t for t in c['tasks']}
    for row in records:
        run=row['id']
        if run not in valid or run in by:raise ValueError('unknown/duplicate scientific identity')
        t=valid[run]
        if run in c.get('timemixer_revision',{}).get('task_ids',[]) and (row.get('execution_revision')!=c['timemixer_revision']['id'] or row.get('attempt')!=c['timemixer_revision']['attempt']):raise ValueError('old TimeMixer receipt cannot populate revision')
        for k,want in dict(model=t['model'],dataset=t['dataset'],h=t['h'],fold=t['fold'],input_variant=t['input_variant'],path=str(result_path(c,t))).items():
            if k in row and row[k]!=want:raise ValueError('receipt identity/path override: '+k)
        identity_profile=row.get('expected_profile_sha') if row.get('status')=='accepted-complete-binding-incomplete' else row.get('profile_sha')
        if identity_profile!=digest(profile(c,t)) or row.get('batch')!=('extension' if run in c['extension']['new_ids'] else 'original'):raise ValueError('profile/batch identity mismatch')
        if row.get('status')=='accepted-complete-binding-incomplete' and not row.get('missing_actual_bindings'):raise ValueError('partial identity requires explicit missing fields')
        if row.get('status')=='success':
            if row.get('reviewed') is not True or not all(row.get(k) for k in ('commit','protocol_sha','data_sha','result_sha256')):raise ValueError('unaudited success')
            if row['protocol_sha'] not in (digest(c),c['extension']['parent_protocol_sha']):raise ValueError('foreign protocol')
            if run in c['extension']['new_ids'] and row['protocol_sha']!=digest(c):raise ValueError('old license cannot cover extension')
            if any(not math.isfinite(row[k]) for k in ('mse','mae')):raise ValueError('nonfinite metrics')
        by[run]=row
    result=[dict(id=t['id'],model=t['model'],dataset=t['dataset'],h=t['h'],fold=t['fold'],input_variant=t['input_variant'],batch='extension' if t['id'] in c['extension']['new_ids'] else 'original',path=str(result_path(c,t)),profile_sha=by.get(t['id'],{}).get('profile_sha',digest(profile(c,t))),**{k:v for k,v in by.get(t['id'],dict(status='not-run' if t['id'] in c['extension']['new_ids']+c.get('timemixer_revision',{}).get('task_ids',[]) else 'unverified')).items() if k not in ('id','batch','profile_sha')})for t in c['tasks']]
    for row in result:
        if row['id'] in c.get('timemixer_revision',{}).get('task_ids',[]):
            row.update(execution_revision=c['timemixer_revision']['id'],attempt=c['timemixer_revision']['attempt'])
    return result

def summarized_index(rows):
    groups={}
    for row in rows:groups.setdefault((row['model'],row['dataset'],row['input_variant']),[]).append(row)
    result=[]
    for (model,domain,inp),bank in groups.items():
        complete=all(v['status']=='success' for v in bank);item=dict(model=model,dataset=domain,input_variant=inp,runs=len(bank),status='reviewed-complete' if complete else 'incomplete-or-unreviewed')
        if complete:
            byh={}
            for r in bank:byh.setdefault(r['h'],[]).append(r)
            item.update({k:sum(sum(v[k] for v in vals)/len(vals) for vals in byh.values())/len(byh) for k in ('mse','mae')})
        result.append(item)
    return result

def controller_live(path):
    try:
        s=json.loads(Path(path).read_text());p=Path('/proc')/str(s['pid'])/'stat';ticks=p.read_text().rsplit(')',1)[1].split()[19]
        return ticks==str(s['start_ticks'])
    except FileNotFoundError:return False

def reject_existing(c,ids,control):
    if Path(control).exists():raise FileExistsError('controller/staging retained; audit before resume')
    for run in ids:
        t=next(t for t in c['tasks'] if t['id']==run)
        if result_path(c,t).exists():raise FileExistsError('existing success/failure/staging: '+run)

def extension_reasons(c,approval,model=None):
    x=c['extension'];reasons=[]
    if model is None:
        from utils.ch3_revision import completion_audit_reasons
        reasons+=completion_audit_reasons(c,(approval or {}).get('revision_completion_audit'))
    for d in Path(c['execution']['evidence']).glob('formal-*/controller.json'):
        if controller_live(d):reasons.append('original formal group live: '+d.parent.name)
    if not approval:return reasons+['extension review, safe boundary, closure and resource approval missing']
    if approval.get('reviewed')is not True or approval.get('execution_permitted')is not True:reasons.append('extension template is not executable')
    if approval.get('protocol_sha')!=digest(c):reasons.append('extension protocol mismatch')
    if approval.get('extension_batch')!=BATCH or approval.get('parent_protocol_sha')!=x['parent_protocol_sha']:reasons.append('extension/parent authorization mismatch')
    ids=queue_ids(c,model)
    if not set(ids)<=set(approval.get('authorized_task_ids',[])):reasons.append('exact extension/append task authorization missing')
    from utils.ch3_revision import boundary_reasons
    reasons.extend(boundary_reasons(c,approval.get('safe_boundary',{})))
    if approval.get('source_explanations_accepted') is not True:reasons.append('EPF source/version/field limitations not reviewed')
    return reasons

def execute_waves(c,waves,control,approval,execute=None):
    """Use existing guarded formal workers/monitor. One failure stops the suffix."""
    from ch3_runner import dump
    from utils.ch3_m6 import run_formal_waves
    execute=execute or run_formal_waves;control=Path(control)
    for number,wave in enumerate(waves):
        if (control/'STOP').exists():raise InterruptedError('supplement stop before next wave')
        t=next(t for t in c['tasks'] if t['id']==wave[0]);out=result_path(c,t).parent
        execute(c,[wave],control/f'wave-{number:03d}',out,approval)
        dump(control/'progress.json',dict(wave=number,completed=wave,total_waves=len(waves)))

def safe_stop(control):
    control=Path(control);s=json.loads((control/'controller.json').read_text())
    if not controller_live(control/'controller.json'):raise RuntimeError('owned controller exited/PID reused; no signal sent')
    (control/'STOP').touch(exist_ok=False)
    os.kill(s['pid'],signal.SIGTERM)
