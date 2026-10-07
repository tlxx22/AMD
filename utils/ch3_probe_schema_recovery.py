"""One registered M_BASE schema failure; immutable old measurements, fresh attempt."""
import ast,copy,hashlib,hmac,json,os,subprocess,sys
from pathlib import Path
from utils.ch3_contract import ROOT,digest,profile,task_by_id,numeric_probe_policy
from utils.ch3_native_recovery_records import bound,ref,sha,exclusive

OLD_COMMIT='fe3def16a3e19928e7c538e223d766751721cbe2'
from utils.ch3_ms_seal_recovery import PACKAGE as ORIGINAL_PACKAGE,RESULT as OLD_RESULT
PACKAGE=ORIGINAL_PACKAGE/'probe-schema-repair-v1'
RESULT=OLD_RESULT.with_name(OLD_RESULT.name+'-probe-schema-r1')
ENTRY=ROOT/'m6_probe_schema_recovery_entry.py'
WRAPPER=ROOT/'scripts/ch3/start_probe_schema_recovery.sh'
SESSION='ch3-baseline-type1-followup-v3-recovery1-probe-schema-r1'
LOG=PACKAGE/'followup-launcher.log'
SOURCE_REF=dict(path=str(PACKAGE/'source-anchors.json'),sha256='70f9fae3b8aa3587d388be8524c99cf1eae9dd5c0725bc555cfded8df640a2a2')
# Filled only after the bounded offline source/number audit, then reviewed with code.
REUSE_REF=dict(path=str(PACKAGE/'probe-reuse-evidence.final.json'),sha256='c4478689612d3b6b1820daeaa01fbb4543a34ce3c4b2e0c8ef77a65146f103e3')
ACTIVE=False

def current_recovery():
    """Explicit public/worker activation selects one fixed, SHA-bound attempt."""
    from utils.ch3_type1_chain import PROBE_RECOVERY
    return PROBE_RECOVERY or sys.modules[__name__]

def activate_worker(value):
    from utils import ch3_amend_ett_identity_recovery as prefix
    if value==prefix.REUSE_REF:prefix.activate();return prefix
    from utils import ch3_moderntcn_etth1_recovery as numeric
    if value==numeric.REUSE_REF:numeric.activate();return numeric
    if value==REUSE_REF:activate();return sys.modules[__name__]
    raise PermissionError('unregistered worker recovery identity')

def activate():
    """Select this fixed attempt in its own entry/worker process, never in W."""
    global ACTIVE
    if ACTIVE:return
    from utils import ch3_type1_tasks as s,ch3_type1_chain as q,ch3_round2_amendment as amendment,ch3_ms_seal_recovery as ms
    ACTIVE=True;s.RESULT=RESULT;ms.RESULT=RESULT;amendment.RESULT=RESULT/'round2-amendment'
    original_context=s.context
    def attempt_context(c):
        ctx=original_context(c)
        return dict(ctx,fixture=PACKAGE/c['baseline_unified']['stage']/'fixtures')
    s.context=attempt_context
    q.CONTROL=RESULT/'queue/controller';q.LOG=LOG;q.SESSION=SESSION;q.ENTRY=ENTRY;q.WRAPPER=WRAPPER;q.PROBE_RECOVERY=sys.modules[__name__];q.QUEUE_LOCK=PACKAGE/'queue.lock'

def anchors():
    if REUSE_REF is None:raise PermissionError('review-bound probe reuse evidence absent')
    return dict(recovery='specific_M_BASE_validation_schema_failure',source_anchors_ref=SOURCE_REF,
        probe_reuse_ref=REUSE_REF,original_training_commit='df6a16403e10d51097db8c88829909c533d15652',
        retained_producer_commit=OLD_COMMIT,expected_runs=dict(MS=203,M=84,total=287),expected_remaining_formal=427)

def evidence():
    v=bound(REUSE_REF);source=bound(SOURCE_REF)
    if v.get('purpose')!='M_BASE_probe_schema_reuse_v1' or v.get('source_anchors_ref')!=SOURCE_REF or v.get('producer_commit')!=OLD_COMMIT:
        raise ValueError('specific reviewed reuse evidence/source mismatch')
    if v.get('comparator_sha')!=comparator_sha():raise ValueError('offline comparator review is for different code')
    if v.get('config_ref')!=source['refs']['config']:raise ValueError('reuse config identity changed')
    return v

def comparator_sha():
    """No commit recursion: exact AST of only the comparison functions reviewed offline."""
    from utils import ch3_native_execution as native
    paths={ROOT/'ch3_runner.py':{'_probe_validation_metrics','compare_probe_trajectories','_compare_full_numeric_files'},
           Path(native.__file__):{'compare'}}
    return digest({str(p.relative_to(ROOT)):{n.name:ast.dump(n,include_attributes=False)for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names}for p,names in paths.items()})

def verify_production_inheritance(permit):
    """Explicit reviewed plumbing delta; every computation-producing function stays exact."""
    allowed={'ch3_runner.py':{'code_binding','compare_probe_trajectories','_probe_validation_metrics'},
      'utils/ch3_native_execution.py':{'compare','run_probe','validate_probe_completion'},
      'utils/ch3_native_recovery_records.py':{'manifest_projection','scan_manifest'},
      'utils/ch3_type1_chain.py':{'upstream_anchors','upstream_status','wait_upstream','start_template','validate_permit','create_permit','wrapper_command','safe_stop','wait_owned','start'},
      'utils/ch3_type1_execution.py':{'read_config','metadata_files','make_config','validate_worker'}}
    deltas={}
    for name,expected in permit['code'].items():
        p=ROOT/name
        if sha(p)==expected:continue
        if name not in allowed:raise ValueError('unapproved producer/source change: '+name)
        old=subprocess.check_output(['git','-C',str(ROOT),'show',OLD_COMMIT+':'+name],text=True)
        def functions(text):return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        a,b=functions(old),functions(p.read_text());changed={k for k in set(a)|set(b)if a.get(k)!=b.get(k)}
        if not changed<=allowed[name]:raise ValueError('computation-producing AST changed: '+name+': '+str(changed))
        deltas[name]=sorted(changed)
    return deltas

def check_environment(c,permit):
    """Same source/data/environment contract, without training or checkpoint IO."""
    from utils import ch3_type1_chain as q
    now=q.dynamic(c)
    for key in ('protocol_sha','environment','hardware','source_states'):
        if permit.get(key)!=now[key]:raise ValueError('retained measurement applicability changed: '+key)

def verify_registered_source():
    """Light startup metadata and qualified exits; no training checkpoint scan."""
    from utils.ch3_type1_upstream import assert_owned_exited
    source=bound(SOURCE_REF);refs=source['refs']
    records={k:bound(refs[k])for k in ('controller','failure','child','probe_failure','producer_permit','MS_boundary','upstream_boundary','authorization','exit_verification','config','source_verification')}
    if source['producer_commit']!=OLD_COMMIT or records['authorization']['closure_commit']!=OLD_COMMIT or records['controller']['authorization']!=refs['authorization']:
        raise ValueError('retained actual producer/authorization mismatch')
    if records['failure']['error']!="RuntimeError('owned child technical failure: M_BASE-probe')" or records['probe_failure']['error']!="ValueError('validation schema changed')":
        raise ValueError('only the registered schema failure is recoverable')
    if records['producer_permit']['commit']!=OLD_COMMIT or records['producer_permit']['protocol_sha']!=digest(records['config']):raise ValueError('old probe protocol/source mismatch')
    ms=records['MS_boundary'];prior=records['upstream_boundary']
    if ms.get('technical_complete')is not True or ms.get('training_commit')!='df6a16403e10d51097db8c88829909c533d15652' or ms.get('seal_execution_commit')!=OLD_COMMIT or len(ms['task_ids'])!=203 or len(set(ms['task_ids']))!=203 or prior['boundaries']['MS']!=refs['MS_boundary']:
        raise ValueError('exact previously sealed original MS203 required')
    verification=records['source_verification']
    if ms['source_verification_ref']!=refs['source_verification'] or ms['task_ids']!=verification['task_ids'] or ms['receipts']!=verification['receipts'] or ms['protocol_sha']!=verification['training_protocol_sha']:raise ValueError('original MS verification binding')
    assert_owned_exited(records['exit_verification']['owned_exited'])
    for p in (OLD_RESULT/'queue/controller/STOP',OLD_RESULT/'probe/M_BASE/STOP'):
        if p.exists():raise ValueError('STOP is not this registered schema failure')
    return records

def status():
    verify_registered_source();evidence()
    return dict(state='MS203_SEAL_AND_PARTIAL_PROBE_REUSE_AWAITING_START',READY_FOR_GPU_EXECUTION=False,anchors=anchors(),result_review='pending')

def import_ms():
    """New owned lifecycle references the old seal, retaining both execution versions."""
    from utils import ch3_type1_chain as q
    q.stop_check();records=verify_registered_source();proof=evidence();verify_production_inheritance(records['producer_permit'])
    source=bound(SOURCE_REF);secret=os.environ.get(q.SECRET)
    if not secret:raise PermissionError('controlled lifecycle required')
    v=dict(state='ORIGINAL_MS203_VERIFIED_SEALED_AND_RELEASED',READY_FOR_GPU_EXECUTION=True,anchors=anchors(),
        boundaries=dict(MS=source['refs']['MS_boundary']),owned_exited=records['upstream_boundary']['owned_exited'],
        handoff_scope=q.s.ID,successor_owner=q.owner(),result_review='pending',reused_seal_ref=source['refs']['MS_boundary'],
        reuse_review_ref=REUSE_REF,adoption_commit=q.closure(),new_MS_training_calls=0,new_MS_test_calls=0)
    v['mac']=hmac.new(secret.encode(),digest(v).encode(),hashlib.sha256).hexdigest()
    return exclusive(q.CONTROL/'upstream-technical-boundary.json',v)

def validate_permit_link(c,a,probe):
    if a.get('execution_attempt')!='M_BASE-probe-schema-r1' or a.get('probe_recovery_ref')!=REUSE_REF:raise PermissionError('exact reviewed attempt/reuse permit required')

def retained_refs(report):
    if not report.get('probe_recovery_ref'):return {}
    if not ACTIVE or report['probe_recovery_ref']!=REUSE_REF:raise PermissionError('unregistered readonly probe root')
    v=evidence();return dict(v['artifacts'],**{r['path']:r for r in [v['producer_permit_ref']]+[e['process']for e in v['evidence'].values()]})

def retained_payload(c,point):
    current=current_recovery()
    if current is not sys.modules[__name__]:return current.retained_payload(c,point)
    if not ACTIVE or c['baseline_unified']['stage']!='M_BASE':return False
    allowed=retained_refs(dict(probe_recovery_ref=REUSE_REF))
    for key,hkey in (('schema_file','schema_sha'),('data_file','data_sha')):
        value=dict(path=point[key],sha256=point[hkey]);p=Path(value['path'])
        if p.is_symlink() or not p.resolve().is_relative_to(OLD_RESULT/'probe/M_BASE') or allowed.get(value['path'])!=value:return False
    return True

def load_seed(c,a):
    if not a.get('probe_recovery_ref'):return None
    if not ACTIVE or c['baseline_unified']['stage']!='M_BASE':raise PermissionError('reuse only on this M_BASE attempt')
    validate_permit_link(c,a,True);v=evidence();verify_registered_source();old_permit=bound(v['producer_permit_ref']);verify_production_inheritance(old_permit);check_environment(c,old_permit)
    if digest(c)!=v['protocol_sha']:raise ValueError('cannot reuse another scientific config')
    # One adoption scan. No tensor comparison, GPU computation or old MS scan.
    for path,value in retained_refs(dict(probe_recovery_ref=REUSE_REF)).items():
        if Path(path).is_symlink() or ref(path)!=value:raise ValueError('retained probe bytes changed before adoption: '+path)
    budget=dict(caps=a['caps'],reserved=copy.deepcopy(v['historical_reserved']),actual=copy.deepcopy(v['historical_actual']),
        historical_actual=copy.deepcopy(v['historical_actual']),new_actual=dict(adam=0,backward=0,forward=0),refund=False)
    return dict(budget=budget,decisions=copy.deepcopy(v['decisions']),evidence=copy.deepcopy(v['evidence']),artifacts=copy.deepcopy(v['artifacts']))

def location(report,key,default):
    if not report.get('probe_recovery_ref'):return Path(default)
    v=evidence()
    if key in v['evidence']:
        if report['evidence'].get(key)!=v['evidence'][key]:raise ValueError('retained process reference changed')
        return Path(v['evidence'][key]['process']['path']).parent
    return Path(default)

def producer(report,key,current):
    if report.get('probe_recovery_ref') and key in evidence()['evidence']:return bound(evidence()['producer_permit_ref'])
    return current

def self_review(report,path):
    if not report.get('probe_recovery_ref'):return False
    v=evidence();return str(path)in v['trajectory_reviews'] and report['artifacts'].get(str(path))==v['artifacts'][str(path)]

def group_review(report,g):
    if not report.get('probe_recovery_ref'):return None
    v=evidence()
    if g['id']in v['decisions']:
        if report['decisions'][g['id']]!=v['decisions'][g['id']]:raise ValueError('retained decision changed')
        return v['decisions'][g['id']]['numerical_comparisons']
    return None

def compile_reuse(c):
    """Preparation only: one bounded offline audit of exactly the registered prefix."""
    from utils import ch3_type1_tasks as s,ch3_native_execution as native
    from utils.ch3_m_execution import wave_passed
    from ch3_runner import _probe_validation_metrics,memory_growth_review
    source=bound(SOURCE_REF);refs=source['refs'];failure=bound(refs['probe_failure']);permit=bound(refs['producer_permit'])
    verify_registered_source();inheritance=verify_production_inheritance(permit);check_environment(c,permit)
    groups=s.probe_groups(c);prefix=groups[:12]
    if list(failure['decisions'])!=[g['id']for g in prefix]:raise ValueError('exact first twelve group prefix')
    pending=groups[12]['id']+'/serial/0';want={g['id']+'/serial/'+str(n):[t]for g in prefix for n,t in enumerate(g['representatives'])}
    want.update({g['id']+'/q4/0':g['representatives']for g in prefix});want[pending]=[groups[12]['representatives'][0]]
    if set(failure['evidence'])!=set(want):raise ValueError('exact 96 complete + one serial measurement required')
    assets=failure['artifacts'];reviews={};counts=dict(adam=0,backward=0,forward=0)
    for path,value in assets.items():
        p=Path(path)
        if p.is_symlink() or not p.resolve().is_relative_to(OLD_RESULT/'probe/M_BASE') or ref(path)!=value:raise ValueError('old probe artifact/source SHA changed: '+path)
    for key,ids in want.items():
        entry=failure['evidence'][key];group,phase,n=key.split('/');wave=OLD_RESULT/'probe/M_BASE'/group/phase/('wave-'+n)
        if entry!=dict(process=ref(wave/'process.json'),task_ids=ids):raise ValueError('actual resource/wave/member identity')
        measured=bound(entry['process'])
        if not wave_passed(measured) or measured.get('exit_transitions_resolved')is not True or measured.get('process_attribution')!='Measured':raise ValueError('retained measured resource/exit gate')
        for run in ids:
            out=wave.parent/run;t=task_by_id(c,run);p=profile(c,t);cfg=bound(assets[str(out/'config.json')]);b=bound(assets[str(out/'budget.json')]);tr=bound(assets[str(out/'trajectory.json')]);rt=bound(assets[str(out/'runtime.json')]);cost=s.worker_counts(c,t)
            if cfg['task']!=run or cfg['output']!=str(out) or cfg['approval']!=permit or cfg['protocol_sha']!=digest(c) or cfg['successor_phase']!=phase or cfg['limits']!=dict(**cost,seconds=1800) or cfg['prefix_files']:raise ValueError('retained worker task/config/producer')
            if cfg.get('purpose')!='ch3_probe' or cfg.get('resume')is not False or cfg.get('kernel_probe')is not False or cfg.get('device')!='cuda:0' or cfg.get('ids')!=[run] or cfg['unified_stage']!='M_BASE' or cfg['type1_scope']!=s.ID or cfg['successor_scope']!=s.context(c)['probe_scope'] or cfg['protocol_file']!=str(s.file('M_BASE')) or cfg['session_root']!=str(OLD_RESULT/'probe/M_BASE') or cfg['artifact_root']!=str(out):raise ValueError('retained guarded compute/namespace identity')
            for name,h in permit['code'].items():
                if cfg['bound_files'].get(str(ROOT/name))!=h:raise ValueError('retained worker did not bind producer bytes')
            if cfg['author_files']!=c['sources'].get(t['model'],{}).get('files',{}):raise ValueError('retained author/input implementation')
            for path,h in dict(cfg['metadata_files'],**cfg['author_files']).items():
                if sha(path)!=h:raise ValueError('retained data/metadata/source binding changed: '+path)
            if tr['id']!=run or tr['profile_sha']!=digest(p) or tr.get('time_mark')!=p.get('time_mark') or tr.get('finite')is not True or tr['steps']!=6 or len(tr['batch_ids'])!=6:raise ValueError('retained trajectory identity/coverage')
            if b['counts']!=cost or b.get('by_pid')!={str(rt['pid']):cost} or rt.get('task')!=run or rt.get('error')is not None or str(rt['pid'])not in measured['process_peaks']:raise ValueError('retained runtime/accounting/ownership')
            if tr['threads']!=4 or tr['affinity']!=permit['hardware']['cpu_affinity'] or tr['memory_review']!=memory_growth_review(tr['memory']) or tr['memory_review']['blocked'] or tr['memory_review']['needs_long_window']:raise ValueError('retained memory/thread gate')
            if any(json.loads(line).get('event')=='denied'for line in (out/'audit.jsonl').read_text().splitlines()):raise ValueError('retained guard denial')
            _probe_validation_metrics(c,t,tr['validation'],tr['validation_tail'])
            for point in tr['M_full_state_trace']:
                if any(assets.get(point[k])!=dict(path=point[k],sha256=point[h])for k,h in (('schema_file','schema_sha'),('data_file','data_sha'))):raise ValueError('retained full-state coverage/checksum binding')
            if len(tr['M_full_state_trace'])!=6:raise ValueError('retained six-state coverage')
            reviews[str(out/'trajectory.json')]=dict(validation_schema='valid M aggregate and diagnostics',producer_commit=OLD_COMMIT)
            for k in counts:counts[k]+=b['counts'][k]
    decisions=copy.deepcopy(failure['decisions']);comparisons=0
    for g in prefix:
        d=decisions[g['id']]
        if any(numeric_probe_policy(c,task_by_id(c,t))is not None for t in g['representatives']):raise ValueError('prefix is specifically original exact branch')
        serial=[bound(failure['evidence'][g['id']+'/serial/'+str(n)]['process'])for n in range(4)]
        parallel=bound(failure['evidence'][g['id']+'/q4/0']['process'])
        if d['status']!='Passed' or d['concurrency']!=4 or d['serial']!=serial or d['parallel']!=[parallel] or d['attempts']!=[dict(q=4,waves=[parallel],resource_failed=False)] or d['coverage']!=g['coverage'] or parallel['elapsed']>=sum(x['elapsed']for x in serial):raise ValueError('retained decision/resource/makespan provenance')
        rows=[]
        for run in g['representatives']:
            x=bound(assets[str(OLD_RESULT/'probe/M_BASE'/g['id']/'serial'/run/'trajectory.json')]);y=bound(assets[str(OLD_RESULT/'probe/M_BASE'/g['id']/'q4'/run/'trajectory.json')]);row=native.compare(c,task_by_id(c,run),x,y);comparisons+=1
            if not row['passed']:raise ValueError('retained exact numerical comparison failed')
            rows.append(row)
        if rows!=d['numerical_comparisons']:raise ValueError('retained exact comparison semantics changed')
    run=want[pending][0];path=OLD_RESULT/'probe/M_BASE'/groups[12]['id']/'serial'/run/'trajectory.json';tr=bound(assets[str(path)])
    row=native.compare(c,task_by_id(c,run),tr,tr);comparisons+=1
    if not row['passed']:raise ValueError('TimeMixer serial self-check failed')
    reviews[str(path)]['self_comparison']=row
    if counts!=failure['budget']['actual'] or counts!=failure['budget']['reserved'] or counts!=dict(adam=582,backward=582,forward=776):raise ValueError('retained actual consumption cannot be reset/guessed')
    return dict(purpose='M_BASE_probe_schema_reuse_v1',source_anchors_ref=SOURCE_REF,producer_commit=OLD_COMMIT,
        producer_permit_ref=refs['producer_permit'],config_ref=refs['config'],protocol_sha=digest(c),comparator_sha=comparator_sha(),
        production_AST_deltas=inheritance,decisions=decisions,evidence=failure['evidence'],artifacts=assets,
        trajectory_reviews=reviews,historical_actual=counts,historical_reserved=failure['budget']['reserved'],
        retained_groups=12,retained_trajectories=97,offline_numeric_comparisons=comparisons,
        fresh_nominal_workers=71,formal_runs=427,new_forward=0,new_backward=0,new_adam=0,new_test_calls=0,
        whole_M_probe_admission=False,result_review='pending')
