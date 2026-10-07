"""One user-authorized synthetic eight-worker diagnostic, never a formal permit.

The existing restricted bootstrap, counters, GPU group lock and owned resource
monitor execute the work. No fallback, retry, dataset observation or checkpoint.
"""
import copy,json,os,subprocess,time
from pathlib import Path
from utils.ch3_contract import ROOT,digest,profile,numeric_probe_policy,task_by_id
from utils.ch3_native_recovery_records import sha,ref,bound,exclusive

P=ROOT.parent/'amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/baseline-unified-v3-ms-seal-m128-recovery1'
PACKAGE=P/'moderntcn-etth1-numeric-diagnosis-v1'
OUTPUT=PACKAGE/'short-confirmation'
PLAN=PACKAGE/'short-confirmation-plan.json'
CONFIG=ROOT/'configs/ch3_round2_m_batch128_recovery1.json'
CONFIG_SHA='7f4720a723e31e3f5b291466c5b483021d480e0152c4317871f8d8097b4b14a5'
BASE='e73545ecfb717284023bc14f00993b91b2c75bf1'
PURPOSE='ch3_moderntcn_etth1_diagnostic'
HORIZONS=(96,192,336,720)
CAPS=dict(adam=48,backward=56,forward=88)
CANDIDATE=dict(state_atol=2e-4,loss_atol=1e-6,metric_atol=1e-6,rtol=0,loss_rtol=0,equal_nan=False)

def read_profile():
    if sha(CONFIG)!=CONFIG_SHA:raise ValueError('diagnostic frozen M_BASE config changed')
    return json.loads(CONFIG.read_text())

def target_profile(c,t):
    if task_by_id(c,t['id'])!=t or (t['model'],t['dataset'],t.get('task'),t['h'])!=('ModernTCN','ETTh1','M',t['h']) or t['h']not in HORIZONS:
        raise ValueError('only the exact ModernTCN ETTh1 four-H M tasks')
    p=profile(c,t);tr=p['training'];r=numeric_probe_policy(c,t)
    if (p['T'],p['C'],p['pred_len'],tr['batch'],tr['eval_batch'],tr['epochs'],tr['patience'],tr['seed'])!=(96,7,t['h'],128,128,10,None,2024):
        raise ValueError('diagnostic scientific profile changed')
    if r['kind']!='full_float_state' or r['state_atol']!=1e-4 or r['metric_atol']!=1e-6 or r['rtol']!=0 or r['equal_nan']is not False:
        raise ValueError('original collection policy required')
    return p

def tasks(c):
    result=[t for t in c['tasks']if t['model']=='ModernTCN'and t['dataset']=='ETTh1']
    if [t['h']for t in result]!=list(HORIZONS):raise ValueError('four-H order/coverage')
    for t in result:target_profile(c,t)
    return result

def code_files():
    from m5_formal_entry import repository_files
    b=repository_files()
    for rel in ('utils/ch3_moderntcn_etth1_diagnostic.py','tests/test_m6_moderntcn_etth1_diagnostic.py'):
        b[str(ROOT/rel)]=sha(ROOT/rel)
    return b

def specification(c):
    from ch3_runner import environment_binding,hardware_binding
    ids=[t['id']for t in tasks(c)]
    return dict(purpose=PURPOSE,producer_parent_commit=BASE,config_ref=ref(CONFIG),protocol_sha=digest(c),
        profiles={t['id']:digest(target_profile(c,t))for t in tasks(c)},
        waves=[dict(phase='serial',tasks=[r])for r in ids]+[dict(phase='q4',tasks=ids)],
        caps=CAPS,candidate=CANDIDATE,workers=8,repetitions_per_mode=4,
        output=str(OUTPUT),code=code_files(),environment=environment_binding(),hardware=hardware_binding(),
        offline_ref=ref(PACKAGE/'offline-state-review.json'),owned_exit_ref=ref(PACKAGE/'owned-exit.json'),
        authorization_basis='user authorized one fixed-seed2024 synthetic four-H serial/q4 confirmation and up to four frozen Conv1d backward replays per mode; no formal/test authorization is materialized here',
        no_retry=True,no_fallback=True,no_real_data=True)

def plan(c):
    v=bound(ref(PLAN))
    if v!=specification(c):raise ValueError('exact diagnostic plan/code/environment binding changed')
    old=bound(v['offline_ref']);exited=bound(v['owned_exit_ref'])
    if not all(old.get(k)is True for k in ('source_identity_exact','all_exact_state_equal','all_finite')):
        raise ValueError('initial offline exclusions incomplete')
    if exited.get('signals_sent')!=0:raise ValueError('read-only ownership proof required')
    from utils.ch3_type1_upstream import assert_owned_exited
    assert_owned_exited(exited['instances'])
    return v

def limits(t,phase):
    replay=phase=='serial'and t['h']==96
    return dict(adam=6,backward=14 if replay else 6,forward=18 if replay else 10,seconds=1800)

def validate_worker(c,s):
    v=plan(c);t=task_by_id(c,s['task']);target_profile(c,t);phase=s.get('phase')
    replay=phase=='serial'and t['h']==96
    exact=dict(purpose=PURPOSE,protocol_sha=digest(c),protocol_file=str(CONFIG),
        session_root=str(OUTPUT),fixture_root=str(OUTPUT/'fixtures'),output=str(OUTPUT/phase/t['id'])if phase in ('serial','q4')else None,
        artifact_root=s['output'],ids=[t['id']],limits=limits(t,phase),bound_files=code_files(),
        author_files=c['sources']['ModernTCN']['files'],prefix_files={},metadata_files={},resume=False,
        device='cuda:0',kernel_probe=replay,plan_ref=ref(PLAN))
    if any(s.get(k)!=x for k,x in exact.items()):raise ValueError('diagnostic worker scope/path/source/ceiling mismatch')
    if phase not in ('serial','q4')or os.environ.get('TMPDIR')!=str(OUTPUT/'fixtures')or not (OUTPUT/'fixtures').is_dir():
        raise ValueError('diagnostic exact wave/fixture required')
    if any(s.get(k)for k in ('approval','successor_scope','unified_scope','type1_scope','recovery_scope')):
        raise ValueError('diagnostic is not a formal/probe permit')
    for p,h in s['author_files'].items():
        if sha(p)!=h:raise ValueError('author source changed')

def validate_wave(c,configs,out):
    if not configs:raise ValueError('empty diagnostic wave')
    for s in configs:validate_worker(c,s)
    phase=configs[0]['phase'];ids=[s['task']for s in configs]
    if any(s['phase']!=phase for s in configs)or dict(phase=phase,tasks=ids)not in plan(c)['waves'] or Path(out)!=OUTPUT/phase/('wave-'+ids[0]):
        raise ValueError('only five fixed diagnostic waves; no fallback or extra workers')
    return OUTPUT/'STOP'

def make_config(c,t,phase):
    from resource_budget import initialize
    out=OUTPUT/phase/t['id'];out.mkdir(parents=True,exist_ok=False)
    s=dict(version='restricted-regression-minimal-v3',repo=str(ROOT),tool_root=str(ROOT/'tools/restricted_regression'),
        purpose=PURPOSE,task=t['id'],case=None,ids=[t['id']],protocol_file=str(CONFIG),protocol_sha=digest(c),phase=phase,
        session_root=str(OUTPUT),fixture_root=str(OUTPUT/'fixtures'),output=str(out),artifact_root=str(out),
        audit_log=str(out/'audit.jsonl'),budget_file=str(out/'budget.json'),limits=limits(t,phase),
        bound_files=code_files(),author_roots=[v['repository']for v in c['sources'].values()],
        author_files=c['sources']['ModernTCN']['files'],prefix_files={},metadata_files={},forbidden_roots=[],
        device='cuda:0',resume=False,kernel_probe=phase=='serial'and t['h']==96,approval=None,plan_ref=ref(PLAN))
    for name in ('cache/torch/kernels','mpl','cuda-cache'):(out/name).mkdir(parents=True)
    exclusive(out/'config.json',s);initialize(s['budget_file'],PURPOSE,s['limits'])
    return s

def worker(c,s):
    from ch3_runner import probe_worker,dump
    out=Path(s['output']);started=time.time();error=None
    try:
        probe_worker(c,task_by_id(c,s['task']),out,m_confirmation=True,
            backend_name='model.downsample_layers.0.0'if s['kernel_probe']else None,backend_repetitions=4)
    except BaseException as exc:error=repr(exc);raise
    finally:dump(out/'runtime.json',dict(task=s['task'],pid=os.getpid(),started=started,finished=time.time(),error=error))

def run():
    """One exclusive claim, one project lock; children never acquire that lock."""
    c=read_profile();plan(c)
    if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=BASE:
        raise ValueError('unexpected diagnostic development parent')
    OUTPUT.mkdir(exist_ok=False);(OUTPUT/'fixtures').mkdir()
    exclusive(OUTPUT/'claim.json',dict(plan_ref=ref(PLAN),pid=os.getpid(),no_retry=True))
    from ch3_runner import GPULock
    from m5_formal_entry import run_configs
    reports=[]
    with GPULock(c):
        for w in plan(c)['waves']:
            configs=[make_config(c,task_by_id(c,i),w['phase'])for i in w['tasks']]
            r=run_configs(configs,OUTPUT/w['phase']/('wave-'+w['tasks'][0]),monitor=True)
            reports.append(dict(wave=w,measurement=r))
            exclusive(OUTPUT/'dispatch-results.json'if len(reports)==5 else OUTPUT/('dispatch-'+str(len(reports))+'.json'),reports)
            if r['failure']or any(r['returncodes'])or r['resource_admission']is not True:
                raise RuntimeError('diagnostic technical/resource failure; no retry or fallback')

def candidate_comparison(c,t,x,y):
    """Evaluation-only policy mapping, bound to this diagnostic's original policy.

    Collection JSON/schema remain immutable. The ordinary comparator first
    verifies their original policy SHA/id; then its full-state reader applies
    the predeclared state bound without changing the collection identity.
    """
    from ch3_runner import compare_probe_trajectories,_compare_full_numeric_files,_probe_validation_metrics
    r=numeric_probe_policy(c,t);target_profile(c,t)
    for key in ('scheduler_trace','time_mark','threads','affinity'):
        if x.get(key)!=y.get(key):raise ValueError('exact diagnostic state changed: '+key)
    for side in (x,y):
        if side.get('M_full_state_trace')!=side.get('full_numeric_trace'):
            raise ValueError('complete M state trace identity')
    old=compare_probe_trajectories(c,t,x,y)
    new=copy.deepcopy(r);new['state_atol']=CANDIDATE['state_atol']
    points=[_compare_full_numeric_files(new,a,b)for a,b in zip(x['full_numeric_trace'],y['full_numeric_trace'])]
    failures=[f for i,row in enumerate(points,1)for f in row['failures']]
    if old['loss_max_abs']>1e-6 or old['validation_max_abs']>1e-6:failures.append('original loss/step2 metric limit')
    metric_diffs={}
    for side in (x,y):
        obs=side['M_confirmation']['evaluations'];a,b=obs['2'],obs['6']
        if a['metrics']!=side['validation']or a['batch_ids']!=b['batch_ids']or b['before']!=b['after']or b['mode_restored']is not True:
            raise ValueError('endpoint changed trajectory/RNG or cached batches')
        if b['before']!=dict(rng=side['final_rng'],model=side['final'],optimizer=side['trajectory'][-1]['optimizer']):
            raise ValueError('not the fixed step6 endpoint')
    for step in ('2','6'):
        a,b=[side['M_confirmation']['evaluations'][step]for side in (x,y)]
        if a['batch_ids']!=b['batch_ids']:raise ValueError('endpoint input identity')
        u,v=[_probe_validation_metrics(c,t,e['metrics'],x['validation_tail'])for e in (a,b)]
        if u['elements']!=v['elements']:raise ValueError('endpoint elements')
        metric_diffs[step]={k:abs(u[k]-v[k])for k in ('mse','mae')}
        if any(d>1e-6 for d in metric_diffs[step].values()):failures.append('endpoint metric '+step)
    return dict(passed=not failures,collection_policy_sha=digest(r),evaluation_candidate=CANDIDATE,
        original_comparison=old,state_max_abs=max(p['state_max_abs']for p in points),
        loss_max_abs=old['loss_max_abs'],evaluation_diffs=metric_diffs,failures=failures,
        exact_residual_state=all(p['exact']for p in points))
