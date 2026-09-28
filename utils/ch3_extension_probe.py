"""Exact EPF/TimeMixer supplement probe wiring; no valid license is supplied here."""
import copy,hashlib,json,os,signal,time
from pathlib import Path
from utils.ch3_contract import digest,profile,task_by_id
from utils.ch3_extension import BATCH,controller_live
PACKAGE=Path('/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc')
PRODUCTION=Path('/public/home/yueweiting/大论文/AMD')
ROOT=PACKAGE/'extension-probe-v1'
PLAN=PACKAGE/'plans/probe-coverage.json'
PARENT=Path('/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-formal-launch-dhozikhu/runtime-probe-admission.json')
CAPS={'adam':360,'forward':512,'backward':360}
PREPARATION={'adam':0,'forward':18,'backward':0}

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def specification(c):
    plan_bytes=PLAN.read_bytes();plan=json.loads(plan_bytes);plan_sha=hashlib.sha256(plan_bytes).hexdigest();extra=[g for g in c['groups'] if set(g['task_ids'])<=set(c['extension']['new_ids'])]
    if plan['protocol_sha']!=digest(c) or len(extra)!=34 or sum(g['q']for g in extra)!=37:raise ValueError('extension probe plan protocol/coverage')
    listed=plan['extension_groups']
    if [g['id']for g in extra]!=[g['group']for g in listed]:raise ValueError('extension group order')
    for g,x in zip(extra,listed):
        if (g['representatives'],g['task_ids'],g['q'])!=(x['representatives'],x['tasks'],x['q']):raise ValueError('extension scope mismatch')
        if x['profile_shas']!={r:digest(profile(c,task_by_id(c,r)))for r in g['representatives']}:raise ValueError('probe profile changed')
    inherit=[g['group']for g in listed if g['status']=='eligible_for_reviewed_q1_carry_forward'];fresh=[g['id']for g in extra if g['id']not in inherit]
    if len(inherit)!=26 or len(fresh)!=8:raise ValueError('26/8 probe partition')
    original=[g['id']for g in c['groups']if g['id']not in [v['id']for v in extra]+c.get('timemixer_revision',{}).get('changed_group_ids',[])]
    revised=c.get('timemixer_revision',{}).get('changed_group_ids',[])
    partition=original+inherit+fresh+revised
    if (len(original),len(inherit),len(fresh),len(revised))!=(51,26,8,3) or len(set(partition))!=88 or set(partition)!={g['id']for g in c['groups']}:raise ValueError('51/26/8/3 admission partition')
    return dict(batch=BATCH,plan_sha256=plan_sha,protocol_sha=digest(c),original_groups=[g['id']for g in c['groups']if g['id']not in [v['id']for v in extra]+c.get('timemixer_revision',{}).get('changed_group_ids',[])],excluded_original_groups=c.get('timemixer_revision',{}).get('changed_group_ids',[]),new_groups=[g['id']for g in extra],proposed_inheritance=inherit,new_measurements=fresh,
      representatives={g['id']:g['representatives']for g in extra},profile_sha={r:digest(profile(c,task_by_id(c,r)))for g in extra for r in g['representatives']},parent_report_sha=plan['parent_report_sha256'],root=str(ROOT),caps=CAPS,already_charged=PREPARATION,worker=dict(adam=6,forward=8,backward=6,validation=2),fallbacks=[4,2,1])

def scope_module(a):
    scope=(a or {}).get('probe_scope')
    if scope=='urban-numeric-diagnostic-v1':
        from utils import ch3_urban_diagnostic
        return ch3_urban_diagnostic
    if scope=='timemixer-revision-numeric-v1':
        from utils import ch3_revision_probe
        return ch3_revision_probe
    if scope not in (None,'extension-probe-v1'):raise ValueError('unknown exact probe scope')
    import sys
    return sys.modules[__name__]

def authorization_reasons(c,a):
    spec=specification(c);reasons=[]
    if not a:return ['extension probe review/actual closure authorization missing']
    if a.get('purpose')!='ch3_resource_probe' or a.get('reviewed')is not True or a.get('execution_permitted')is not True:reasons.append('non-executable/template/old probe approval')
    if a.get('probe_scope') not in (None,'extension-probe-v1'):reasons.append('foreign probe scope')
    if a.get('extension_batch')!=BATCH or a.get('scope_sha')!=digest(spec) or a.get('protocol_sha')!=digest(c):reasons.append('extension scope/protocol binding mismatch')
    if a.get('plan_sha256')!=spec['plan_sha256']:reasons.append('complete probe PLAN SHA mismatch')
    if a.get('parent_protocol_sha')!=c['extension']['parent_protocol_sha'] or a.get('parent_report_sha')!=spec['parent_report_sha']:reasons.append('parent evidence mismatch')
    if a.get('caps')!=CAPS or a.get('already_charged')!=PREPARATION:reasons.append('aggregate budget/debit mismatch')
    mode=a.get('carry_forward_mode');inherit=spec['proposed_inheritance']if mode=='reviewed_26'else[]
    expected=spec['new_measurements']if mode=='reviewed_26'else spec['new_groups']
    if mode not in ('reviewed_26','none')or a.get('inherited_group_ids')!=inherit or a.get('execute_group_ids')!=expected:reasons.append('exact new scope/inheritance approval missing')
    allowed=[r for g in expected for r in spec['representatives'].get(g,[])]
    if a.get('authorized_task_ids')!=allowed:reasons.append('representative task IDs mismatch')
    from utils.ch3_revision import boundary_reasons,revision_report_reasons
    reasons+=boundary_reasons(c,a.get('safe_boundary',{}))
    reasons+=revision_report_reasons(c,a.get('revision_resource_report'),a)
    for path in Path(c['execution']['evidence']).glob('formal-*/controller.json'):
        if controller_live(path):reasons.append('original group live: '+path.parent.name)
    if a.get('meter_receipt_sha')!=sha(PACKAGE/'meter-closeout-v1/meter-ledger.json'):reasons.append('CPU meter receipt mismatch')
    elif read(PACKAGE/'meter-closeout-v1/meter-ledger.json').get('overall')!='Passed':reasons.append('CPU meter not accepted')
    return reasons

def locations():return str(ROOT),str(ROOT/'fixture')

def validate_worker(c,s):
    if s['task']not in s.get('approval',{}).get('authorized_task_ids',[]):raise ValueError('worker outside exact extension scope')
    if s['session_root']!=str(ROOT)or s['fixture_root']!=str(ROOT/'fixture')or s.get('prefix_files'):raise ValueError('probe session/prefix boundary')
    if not Path(s['output']).resolve().is_relative_to(ROOT):raise ValueError('extension output escape')

def readiness(c,a):
    reasons=authorization_reasons(c,a)
    if ROOT.exists():reasons.append('retained extension probe output; no duplicate start')
    from ch3_runner import ROOT as repo
    if repo.resolve()!=PRODUCTION:reasons.append('external candidate is not deployed reviewed production')
    if not reasons:
        from ch3_runner import preflight
        reasons+=preflight(c,None,a,probe=True) # actual clean Git/code/config/environment/hardware; no bypass
    return list(dict.fromkeys(reasons))

class Budget:
    def __init__(self,caps=CAPS,charged=PREPARATION):self.caps=dict(caps);self.counts=dict(charged);self.attempts=[]
    def reserve(self,configs):
        addition={k:sum(s['limits'][k]for s in configs)for k in self.caps}
        if any(self.counts[k]+addition[k]>self.caps[k]for k in self.caps):raise RuntimeError('aggregate budget before launch')
        # Reserve full bounded attempts. Missing/abnormal worker ledgers are NOT a refund.
        for k in self.caps:self.counts[k]+=addition[k]
        self.attempts.append(dict(configs=[s['output']for s in configs],reserved=addition))
    def record(self,configs):
        actual=[]
        for s in configs:
            p=Path(s['budget_file']);row=read(p)if p.exists()else None
            actual.append(dict(output=s['output'],ledger=str(p),counts=row.get('counts')if row else None))
        self.attempts[-1]['actual']=actual

def parallel_wave(launch,representatives,directory,q):
    """Fail closed unless the existing monitor explicitly classified resource failure."""
    if q not in (4,2) or len(representatives)!=4:raise ValueError('fixed four-task parallel wave only')
    measured=[];observed=[]
    for offset in range(0,4,q):
        m,t=launch(representatives[offset:offset+q],directory)
        measured.append(m)
        if m['failure'] or any(m['returncodes']) or not m['resource_admission']:
            if m.get('failure_kind')!='resource':
                raise RuntimeError('unconfirmed/business/numeric/identity/guard failure: stop, no concurrency fallback')
            return measured,observed,True
        observed+=t
    return measured,observed,False

def execute(c,a):
    reasons=readiness(c,a)
    if reasons:raise PermissionError('; '.join(reasons))
    from ch3_runner import GPULock,dump,code_binding,environment_binding,hardware_binding,compare_probe_trajectories
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/restricted_regression'))
    from m5_formal_entry import make_config,run_configs,gpu_sample,resource_assessment
    spec=specification(c)
    if sha(PARENT)!=spec['parent_report_sha']:raise ValueError('parent report bytes changed')
    parent=read(PARENT);plan_bytes=PLAN.read_bytes()
    if hashlib.sha256(plan_bytes).hexdigest()!=a.get('plan_sha256') or digest(spec)!=a.get('scope_sha'):raise ValueError('reviewed execution proof changed before launch')
    plan=json.loads(plan_bytes);budget=Budget()
    report=dict(protocol_sha=digest(c),code=code_binding(),environment=environment_binding(),hardware=hardware_binding(),Q=sum(g['q']for g in c['groups']),decisions={},inheritance={},steps=0,scope_sha=digest(spec),scope=spec)
    # The actual closure review must approve the unchanged original scope and the explicit 26 transfer proofs.
    for gid in spec['original_groups']:
        report['decisions'][gid]=copy.deepcopy(parent['decisions'][gid]);report['inheritance'][gid]=dict(path=str(PARENT),sha256=sha(PARENT),kind='original scope unchanged, not reexecuted')
    for gid in a['inherited_group_ids']:
        item=next(g for g in plan['extension_groups']if g['group']==gid);old=parent['decisions'][item['parent_group']]
        if old['status']!='Passed' or old['concurrency']!=1 or set(item['profile_differences'])-{'dataset','features'}:raise ValueError('not eligible q1 transfer')
        d=copy.deepcopy(old);d.update(representatives=spec['representatives'][gid],scope='reviewed computational q1 transfer; new data independently bound')
        report['decisions'][gid]=d;report['inheritance'][gid]=dict(path=str(PARENT),sha256=sha(PARENT),source_group=item['parent_group'],proof=item)
    if spec.get('excluded_original_groups'):
        revision=read(a['revision_resource_report']['path'])
        for gid in spec['excluded_original_groups']:
            report['decisions'][gid]=copy.deepcopy(revision['decisions'][gid]);report['inheritance'][gid]=dict(path=a['revision_resource_report']['path'],sha256=a['revision_resource_report']['sha256'],kind='new LR separately admitted; NOT inherited from old original scope')
    def launch(runs,directory):
        configs=[]
        if (ROOT/'STOP').exists():raise InterruptedError('owned probe safe-stop')
        ceilings=[dict(limits=spec['worker'],output=str(directory/r))for r in runs]
        # capacity checked even before new worker directories/configs
        if any(budget.counts[k]+sum(v['limits'][k]for v in ceilings)>CAPS[k]for k in CAPS):raise RuntimeError('insufficient aggregate budget')
        for r in runs:configs.append(make_config(c,'ch3_probe',directory/r,task=r,approval=a))
        budget.reserve(configs);dump(ROOT/'budget.json',vars(budget))
        try:measured=run_configs(configs,directory/('monitor-'+runs[0]),monitor=True)
        finally:budget.record(configs);dump(ROOT/'budget.json',vars(budget))
        if measured['failure']or any(measured['returncodes']):return measured,[]
        traces=[read(Path(s['output'])/'trajectory.json')for s in configs]
        for t,r in zip(traces,runs):
            if t['id']!=r or t['profile_sha']!=spec['profile_sha'][r]or t['steps']!=6 or t['finite']is not True:raise ValueError('worker result identity/finite')
        return measured,traces
    def stop(sig,frame):raise InterruptedError('owned probe interrupted')
    prior=signal.signal(signal.SIGTERM,stop);owned_output=False
    try:
      with GPULock(c):
        ROOT.mkdir(exist_ok=False);owned_output=True;(ROOT/'fixture').mkdir();dump(ROOT/'controller.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],scope_sha=digest(spec)))
        for gid in a['execute_group_ids']:
            g=next(x for x in c['groups']if x['id']==gid);base=ROOT/gid;serial=[];reference=[]
            for r in g['representatives']:
                m,t=launch([r],base/'serial');serial.append(m)
                if m['failure']or any(m['returncodes'])or not m['resource_admission']:raise RuntimeError('single worker technical/resource failure; stop suffix')
                reference+=t
            decision=dict(status='Passed',concurrency=1,representatives=g['representatives'],serial=serial,scope='six updates only',parallel='Not tested')
            if g['q']>1:
                sample=gpu_sample([]);headroom=max(8*1024**3,.1*sample['total'])
                peaks=[max([t['reserved'],m['whole_card_peak']]+[v for v in m['process_peaks'].values()if v is not None])for m,t in zip(serial,reference)]
                for q in (4,2):
                    sample=gpu_sample([])
                    if not resource_assessment(sample,[])['admission']or sample['free']-sum(sorted(peaks,reverse=True)[:q])<headroom:decision[str(q)]='ResourceRejected before launch';continue
                    measured,observed,resource_failed=parallel_wave(launch,g['representatives'],base/f'q{q}',q)
                    if resource_failed:decision[str(q)]='ResourceFailed';continue
                    if len(observed)!=4 or any(any(x[k]!=y[k]for k in ('id','profile_sha','initial','initial_rng','batch_ids'))for x,y in zip(reference,observed)):raise ValueError('identity/RNG/batch mismatch')
                    comparisons=[compare_probe_trajectories(c,task_by_id(c,r),x,y)for r,x,y in zip(g['representatives'],reference,observed)]
                    if not all(v['passed']for v in comparisons):raise ValueError('numeric mismatch: existing rule retained')
                    speed=sum(x['elapsed']for x in serial)/sum(x['elapsed']for x in measured)
                    decision[str(q)]=dict(resource='Passed',numerical_comparisons=comparisons,makespan=sum(x['elapsed']for x in measured),speedup=speed)
                    if speed>1:decision.update(concurrency=q,parallel='Passed on same four tasks');break
                # Valid serial reference is q1 fallback; no duplicate q1 rerun or automatic repair.
            report['decisions'][gid]=decision;report['steps']=sum(x['reserved']['adam']for x in budget.attempts);dump(ROOT/'progress.json',report)
        from ch3_runner import validate_probe_report
        validate_probe_report(c,report);dump(ROOT/'complete.json',report)
    except BaseException as exc:
        if owned_output:dump(ROOT/'failure.json',dict(error=repr(exc),budget=vars(budget),partial=report))
        raise
    finally:signal.signal(signal.SIGTERM,prior)
