"""Narrow remaining-six-model queue contract. No models or observation reads."""
import ast
import copy
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from utils.ch3_contract import ROOT, digest, profile, task_by_id, step_arithmetic, BestState
from utils.ch3_extension import queue_ids, fixed_waves, result_path, controller_live

QUEUE='m6-remaining-models-v1'
MODELS=('DLinear','iTransformer','ModernTCN','TimeXer','N','S')
PACKAGE=ROOT.parent/'amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc'
PREPARATION=PACKAGE/'remaining-models-v1'
BASE='22b968e741bf45e02c1f7c8ba96a8a91d53e7fb7'
PARENT=PACKAGE/'urban-numeric-confirmation-v1/reviewed-merged-admission.json'
PARENT_SHA='4cf01b8fdd33a0b37f0c35e5b67cf5c695ffc804bb83f3e0023b5ddb62ceb184'
RESOURCE_PURPOSE='m6_remaining_resource_carry_forward_v1'
PYTHON='/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin/python'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ref(p):return dict(path=str(p),sha256=sha(p))
def bound(r):
    p=Path(r['path'])
    if p.is_symlink() or sha(p)!=r['sha256']:raise ValueError('bound source mismatch: '+str(p))
    d=read(p)
    if sha(p)!=r['sha256']:raise ValueError('bound source changed')
    return d

def control_root(c):return Path(c['execution']['evidence'])/'queues/remaining-models-v1'
def model_ids(c,m):
    if m not in MODELS:raise ValueError('model outside six-group scope')
    return queue_ids(c,m) if m in MODELS[:4] else [t['id']for t in c['tasks']if t['model']==m]
def group_control(c,m):
    return Path(c['extension']['result_root'])/'controllers'/('model-'+m) if m in MODELS[:4] else Path(c['execution']['evidence'])/('formal-'+m)

def plan(c,decisions):
    groups=[]
    for m in MODELS:
        ids=model_ids(c,m);tasks=[task_by_id(c,r)for r in ids]
        waves=fixed_waves(c,ids,decisions)
        groups.append(dict(model=m,task_ids=ids,waves=waves,control=str(group_control(c,m)),
          result_paths={t['id']:str(result_path(c,t))for t in tasks},
          profile_shas={t['id']:digest(profile(c,t))for t in tasks},
          runs=len(ids),run_epochs=sum(profile(c,t)['training']['epochs']for t in tasks),
          max_optimizer_steps=sum(step_arithmetic(c,t)['max_optimizer_steps']for t in tasks)))
    ids=sum([x['task_ids']for x in groups],[])
    if len(ids)!=len(set(ids)) or len(ids)!=228:raise ValueError('228 exact unique tasks')
    out=dict(queue_id=QUEUE,protocol_sha=digest(c),order=list(MODELS),groups=groups,task_ids=ids,
       runs=228,run_epochs=sum(x['run_epochs']for x in groups),max_optimizer_steps=sum(x['max_optimizer_steps']for x in groups),waves=sum(len(x['waves'])for x in groups),control=str(control_root(c)))
    if (out['run_epochs'],out['max_optimizer_steps'],out['waves'])!=(2640,7438600,73):raise ValueError('remaining budget/wave mismatch')
    if [len(x['waves'])for x in groups]!=[15,15,16,15,6,6]:raise ValueError('model wave order')
    return out

def child_reasons(c,m,a):
    """Pure worker-side scope check; parent checks bound files before dispatch."""
    if not a or a.get('queue_id')!=QUEUE:return ['remaining child authorization missing']
    ids=model_ids(c,m);ts=[task_by_id(c,r)for r in ids]
    expected=dict(runs=len(ids),run_epochs=sum(profile(c,t)['training']['epochs']for t in ts))
    checks=[(a.get('models')==[m] and a.get('authorized_task_ids')==ids,'exact child task/model scope'),
      (a.get('reviewed')is True and a.get('execution_permitted')is True,'child is non-executable template'),
      (a.get('run_budget')==expected and a.get('max_optimizer_steps')==sum(step_arithmetic(c,t)['max_optimizer_steps']for t in ts),'child budget'),
      (a.get('seed_list')==[2024] and a.get('additional_search')==0,'seed/search'),
      (a.get('profile_shas')=={t['id']:digest(profile(c,t))for t in ts},'child profiles'),
      (set(sum([list(v)for v in a.get('data_bindings',{}).values()],[]))==set(ids),'exact child data subset'),
      (a.get('extension_batch')==c['extension']['id'] if m in MODELS[:4] else 'extension_batch'not in a,'N/S extension separation')]
    return [s for ok,s in checks if not ok]

def worker_scope(c,s):
    a=s.get('approval')or{}
    if a.get('queue_id')!=QUEUE:return
    t=task_by_id(c,s['task']);reasons=child_reasons(c,t['model'],a)
    if reasons:raise PermissionError('; '.join(reasons))
    if s.get('resume') or Path(s['output'])!=result_path(c,t) or Path(s.get('artifact_root',''))!=result_path(c,t):raise PermissionError('remaining worker exact fresh output scope')

def source_proof(c,proof,code):
    """Bind reviewed scheduling diff; preserve every pre-existing compute function."""
    if proof.get('base_commit')!=BASE or proof.get('protocol_sha')!=digest(c) or proof.get('after_code')!=code:raise ValueError('source delta binding')
    old=read(PARENT)['code']
    if proof.get('before_code')!=old or proof.get('profile_shas')!={t['id']:digest(profile(c,t))for t in c['tasks']}:raise ValueError('parent source/profile binding')
    allowed={'ch3_runner.py':{'code_binding','validate_probe_report','preflight'},'tools/restricted_regression/m5_formal_entry.py':{'validate_config'}}
    for name,h in old.items():
        raw=subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':'+name])
        if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('base source mismatch')
        if code.get(name)==h:continue
        if name=='tools/restricted_regression/bundle.sha256':continue
        if name not in allowed:raise ValueError('unexpected computation/source change '+name)
        a=ast.parse(raw);b=ast.parse((ROOT/name).read_text())
        for tree in (a,b):tree.body=[n for n in tree.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))or n.name not in allowed[name]]
        if ast.dump(a)!=ast.dump(b):raise ValueError('non-routing AST changed '+name)
    additions=set(code)-set(old)
    if additions!={'utils/ch3_remaining.py','m6_remaining_entry.py','scripts/ch3/start_remaining_models.sh','tests/test_m6_remaining.py'}:raise ValueError('unreviewed extra source set')

def validate_resources(c,r):
    from ch3_runner import code_binding,environment_binding,hardware_binding
    if r.get('purpose')!=RESOURCE_PURPOSE or r.get('reviewed')is not True or r.get('synthetic_fixture')is not False:raise ValueError('reviewed remaining carry-forward required')
    if r.get('parent')!={'path':str(PARENT),'sha256':PARENT_SHA}:raise ValueError('exact accepted 88-group parent required')
    parent=bound(r['parent']);proof=bound(r['proof']);code=code_binding()
    if proof.get('reviewed')is not True or not proof.get('review_source'):raise ValueError('scheduling delta actual-byte review required')
    source_proof(c,proof,code)
    if parent.get('reviewed')is not True or r.get('decisions')!=parent['decisions']or r.get('lineage')!=parent['lineage']:raise ValueError('old resource decisions/lineage rewritten')
    if r.get('code')!=code or r.get('protocol_sha')!=digest(c):raise ValueError('current code/protocol')
    if any(r.get(k)!=parent[k]for k in ('environment','hardware')) or r['environment']!=environment_binding()or r['hardware']!=hardware_binding():raise ValueError('resource conditions changed')
    from ch3_runner import git
    if r.get('commit')!=git('rev-parse','HEAD'):raise ValueError('resource closure mismatch')
    plan(c,r['decisions'])
    return r

def catchup_audit(c,r):
    a=bound(r)
    if a.get('purpose')!='m6_catchup41_completion_audit' or a.get('reviewed')is not True or a.get('status')!='Passed':raise ValueError('41-run completion audit/review required')
    original=bound(a['review']['original_report'])
    if {k:v for k,v in a.items()if k not in ('reviewed','review')}!={k:v for k,v in original.items()if k!='reviewed'}or not a['review'].get('source'):raise ValueError('completion review derivation')
    ids=queue_ids(c)
    if a['task_ids']!=ids or a['protocol_sha']!=digest(c)or set(a['runs'])!=set(ids):raise ValueError('catchup audit scope')
    complete=bound(a['complete'])
    if a['complete']['path']!=str(Path(c['extension']['result_root'])/'controllers/catchup-41/complete.json')or complete['task_ids']!=ids or complete['protocol_sha']!=digest(c):raise ValueError('catchup completion path/protocol')
    for run in ids:
        t=task_by_id(c,run);b=a['runs'][run]
        for key in ('manifest','result'):
            if b[key]['path']!=str(result_path(c,t)/(key+'.json')):raise ValueError('audit result path')
            obj=bound(b[key])
            if key=='manifest':
                i=obj['identity']
                if i['run_id']!=run or i['profile_sha']!=digest(profile(c,t))or i['protocol_sha']!=digest(c)or i['commit']!=a['commit']or i['data_sha']!=b['data_sha']:raise ValueError('audit actual identity')
    for r in a['sources'].values():bound(r)
    return a

def reject_existing(c,p):
    if control_root(c).exists():raise FileExistsError('queue completed/failed/staging retained; no fresh retry')
    for g in p['groups']:
        if Path(g['control']).exists():raise FileExistsError('group controller/staging retained '+g['model'])
        if any(Path(v).exists() or Path(v).is_symlink()for v in g['result_paths'].values()):raise FileExistsError('existing task output '+g['model'])

def authorization(c,a,check_existing=True):
    if not a:raise PermissionError('remaining reviewed total + six child permits required')
    if a.get('queue_id')!=QUEUE or a.get('purpose')!='m6_remaining_queue' or a.get('reviewed')is not True or a.get('execution_permitted')is not True:raise PermissionError('remaining template not executable')
    from ch3_runner import code_binding,environment_binding,hardware_binding,git,preflight
    if git('status','--porcelain','--untracked-files=all')or a.get('commit')!=git('rev-parse','HEAD'):raise PermissionError('matching clean reviewed closure required')
    for k,v in dict(code=code_binding(),protocol_sha=digest(c),environment=environment_binding(),hardware=hardware_binding()).items():
        if a.get(k)!=v:raise ValueError('total '+k+' mismatch')
    resource=bound(a['resource_report']);validate_resources(c,resource);p=plan(c,resource['decisions'])
    if bound(a['plan'])!=p:raise ValueError('fixed plan mismatch')
    if a.get('task_ids')!=p['task_ids']or a.get('run_budget')!={'runs':228,'run_epochs':2640}or a.get('max_optimizer_steps')!=7438600:raise ValueError('total task/budget')
    catchup_audit(c,a['catchup_completion_audit'])
    if set(a.get('children',{}))!=set(MODELS):raise ValueError('six permits required')
    children={}
    for m in MODELS:
        child=bound(a['children'][m]);reasons=child_reasons(c,m,child)
        if child.get('plan_sha256')!=a['plan']['sha256']or child.get('probe_report_sha')!=a['resource_report']['sha256']:reasons.append('child plan/resource binding')
        if any(child.get(k)!=a[k]for k in ('code','commit','protocol_sha','environment','hardware')):reasons.append('child source differs from total')
        reasons+=preflight(c,m,child)
        if m in MODELS[:4]:
            from utils.ch3_extension import extension_reasons
            reasons+=extension_reasons(c,child,m)
        if reasons:raise PermissionError(m+': '+'; '.join(reasons))
        children[m]=child
    if check_existing:reject_existing(c,p)
    return p,children

def technical_handoff(c,g,a):
    """JSON-only technical gate. No checkpoint reads or scientific result review."""
    from utils.ch3_m6 import verify_result
    control=Path(g['control']);sources={};totals=dict(epochs=0,adam=0,forward=0,backward=0)
    def load(p):
        r=ref(p);v=bound(r);sources[str(p)]=r;return v
    if controller_live(control/'controller.json'):raise ValueError('child controller still live')
    controller=load(control/'controller.json')
    if controller.get('model')!=g['model']:raise ValueError('child controller model identity')
    if any(x.name.startswith(('staging','recovery','failed','running'))or x.name in ('failure.json','STOP')for x in control.iterdir()):raise ValueError('unexplained group failure/staging')
    complete=load(control/'complete.json')
    if complete['task_ids']!=g['task_ids']or complete['protocol_sha']!=digest(c):raise ValueError('group complete identity')
    pids={}
    for run in g['task_ids']:
        t=task_by_id(c,run);p=profile(c,t);d=result_path(c,t);m=load(d/'manifest.json');i=m['identity'];r=load(d/'result.json');b=load(d/'budget.json');runtime=load(d/'runtime.json');conf=load(d/'config.json')
        if any(x.name.startswith(('staging','recovery'))or x.name in ('failure.json','STOP')for x in d.iterdir()):raise ValueError('unexplained run failure/staging')
        if m['task']!=t or m['profile']!=p or i.get('run_id')!=run or i.get('profile_sha')!=digest(p)or i.get('data_sha')!=digest(m['metadata'])or i['data_sha']!=a['data_bindings'][t['dataset']][run]or i.get('commit')!=a['commit']or i.get('protocol_sha')!=digest(c)or i.get('source')!=a['code']:raise ValueError('run actual identity')
        if conf.get('approval')!=a or conf.get('resume')is not False or conf.get('task')!=run or conf.get('output')!=str(d):raise ValueError('executed worker authorization')
        verify_result(c,run,r)
        hp=d/'history.jsonl';sources[str(hp)]=ref(hp);hist=[json.loads(v)for v in hp.read_text().splitlines()]
        if sha(hp)!=sources[str(hp)]['sha256']:raise ValueError('history changed')
        ar=step_arithmetic(c,t);best=BestState(p['training']['patience'])
        if not 1<=len(hist)<=p['training']['epochs']:raise ValueError('epoch count')
        for ep,row in enumerate(hist,1):
            v=row['validation']
            if best.stopped or row['epoch']!=ep or row['steps']!=ep*ar['train_batches']:raise ValueError('history cumulative steps/stopping')
            if any(not math.isfinite(v[k])or v[k]<0 for k in ('mse','mae','sse','sae'))or v['elements']!=ar['validation_windows']*p['pred_len']or v['mse']!=v['sse']/v['elements']or v['mae']!=v['sae']/v['elements']:raise ValueError('validation finite/aggregation')
            best.update(v['mse'],ep)
            if row['best_epoch']!=best.epoch:raise ValueError('best tie/selection')
        if r['best_epoch']!=best.epoch or (len(hist)<p['training']['epochs']and not best.stopped):raise ValueError('final best/early stop')
        steps=hist[-1]['steps'];fwd=len(hist)*(ar['train_batches']+math.ceil(ar['validation_windows']/p['training']['eval_batch']))+math.ceil(ar['test_windows_arithmetic_only']/p['training']['eval_batch'])
        counts=dict(adam=steps,forward=fwd,backward=steps)
        if b['counts']!=counts or runtime['error']is not None or len(b['by_pid'])!=1 or str(runtime['pid'])not in b['by_pid']:raise ValueError('actual counter/runtime')
        if b['by_pid'][str(runtime['pid'])]!=counts:raise ValueError('per-worker counter differs from aggregate')
        pids[run]=runtime['pid'];totals['epochs']+=len(hist)
        for k in counts:totals[k]+=counts[k]
    expected_waves={f'wave-{n:03d}'for n in range(len(g['waves']))}
    if {p.name for p in control.glob('wave-*')}!=expected_waves:raise ValueError('exact wave coverage')
    for n,wave in enumerate(g['waves']):
        d=control/f'wave-{n:03d}'
        if g['model'] in MODELS[:4]:d=d/'wave-000'
        pr=load(d/'process.json')
        if pr['returncodes']!=[0]*len(wave)or not pr['resource_admission']or not pr['exit_transitions_resolved']or pr['failure']is not None or set(pr['process_peaks'])!={str(pids[r])for r in wave}:raise ValueError('wave resource/exit/scope')
        with (d/'memory.jsonl').open()as f:first=json.loads(next(f))
        for run in wave:
            pid=pids[run];ticks=first['owned_pid_metadata'][str(pid)]['start_ticks'];proc=Path('/proc')/str(pid)/'stat'
            try:actual=proc.read_text().rsplit(')',1)[1].split()[19]
            except FileNotFoundError:continue
            if actual==str(ticks):raise ValueError('worker still alive')
    return dict(status='technical-complete',result_review='pending',model=g['model'],task_ids=g['task_ids'],totals=totals,sources=sources,checkpoint_audit='deferred until whole queue completion',test_recomputed=False)
