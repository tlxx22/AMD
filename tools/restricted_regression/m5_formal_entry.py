"""Narrow CH3 acceptance and fixed-group resource admission entry."""
import argparse
import hashlib
import json
import os
import re
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

TOOL=Path(__file__).resolve().parent
REPO=TOOL.parents[1]
sys.path.insert(0,str(REPO))
from utils.ch3_contract import read_profiles,validate_manifest,digest,task_by_id,profile
from ch3_runner import dump,code_binding,preflight,GPULock,dispatch,audit_resume,hardware_binding,environment_binding,validate_probe_report


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def inside(path,root):return path==root or path.startswith(root+os.sep)


def repository_files():
    paths=set(code_binding())|{'configs/ch3_formal_profiles.json','tests/test_ch3_formal.py','models/__init__.py','utils/__init__.py'}
    return {str(REPO/p):sha(REPO/p) for p in paths}


def validate_config(s):
    from run_restricted import verify_bundle,ensure_unique_exact
    verify_bundle();c=validate_manifest(read_profiles())
    if s['protocol_sha']!=digest(c) or s['bound_files']!=repository_files():raise ValueError('CH3 code/config binding mismatch')
    if s['session_root']!=c['execution']['evidence'] or s['fixture_root']!=c['execution']['fixture']:
        raise ValueError('CH3 execution/fixture binding')
    if os.environ.get('TMPDIR')!=s['fixture_root'] or not Path(s['fixture_root']).is_dir():raise ValueError('fixture environment mismatch')
    purpose=s['purpose']
    if purpose=='ch3_cpu':expected=c['acceptance']['cpu_ids']
    elif purpose=='ch3_model_acceptance':
        expected=[c['acceptance']['model_cases'][s['case']]['id']]
        if s['task']!=c['acceptance']['model_cases'][s['case']]['run_id']:raise ValueError('model acceptance task mismatch')
        from utils.ch3_contract import training_blockers
        if training_blockers(c,task_by_id(c,s['task'])):raise ValueError('unresolved baseline training source')
    elif purpose=='ch3_step_diagnostic':
        cases=c['execution'].get('diagnostics',{}).get('cases',{})
        case=cases.get(s.get('case'))
        if case is None or s.get('task')!=case['run_id'] or s.get('diagnostic')!='current-profile-six-step':
            raise ValueError('exact current-package diagnostic capability required')
        if s['limits']['seconds']!=180:raise ValueError('diagnostic time bound')
        from utils.ch3_contract import training_blockers
        if training_blockers(c,task_by_id(c,s['task'])):raise ValueError('diagnostic unresolved source')
        expected=['ch3.diagnostic.'+s['case']]
    elif purpose in ('ch3_probe','ch3_formal'):
        t=task_by_id(c,s['task']);expected=[t['id']]
        if purpose=='ch3_probe' and t['id'] not in [x for g in c['groups'] for x in g['representatives']]:raise ValueError('not a representative worker')
        reasons=preflight(c,t['model'],s.get('approval'),probe=purpose=='ch3_probe')
        if reasons:raise PermissionError('startup prerequisites: '+'; '.join(reasons))
    elif purpose=='ch3_prefix':
        domain=s.get('case')
        if domain not in ('Weather','PJM') or s.get('diagnostic')!='train_validation_connectivity' or s.get('task'):
            raise ValueError('exact Weather/PJM connectivity capability required')
        limit={'Weather':42157,'PJM':41933}[domain]
        expected=['ch3.prefix.'+domain+'.train_validation']
        if s['prefix_files']!={c['datasets'][domain]['path']:limit}:raise ValueError('exact validation prefix required')
        if c['datasets'][domain]['endpoints'][1]!=limit:raise ValueError('prefix endpoint binding mismatch')
    elif purpose=='ch3_placeholder':expected=['ch3.placeholder']
    elif purpose=='ch3_resource_diagnostic':
        expected=['ch3.resource_diagnostic']
        if s.get('diagnostic')!='owned_cuda_tensor_16mib' or s.get('task') or s['limits']['seconds']!=180:
            raise ValueError('exact resource diagnostic capability required')
        if c['execution'].get('resource_diagnostic')!={'max_processes':4,'tensor_bytes':16777216,'seconds':180,'attempts':2}:
            raise ValueError('this package has no exact diagnostic authorization')
    else:raise ValueError('unknown CH3 purpose')
    ensure_unique_exact(s['ids'],expected)
    source=c['sources'].get(task_by_id(c,s['task'])['model']) if s.get('task') else None
    if s['author_files']!=(source['files'] if source else {}):raise ValueError('purpose/source closure mismatch')
    ceilings={'ch3_cpu':(0,0,0),'ch3_prefix':(0,0,0),'ch3_placeholder':(0,0,0),
              'ch3_model_acceptance':(2,6,2),'ch3_probe':(6,8,6),'ch3_step_diagnostic':(6,8,6),'ch3_resource_diagnostic':(0,0,0)}
    if purpose in ceilings:
        if tuple(s['limits'][k] for k in ('adam','forward','backward'))!=ceilings[purpose]:raise ValueError('exact operation budget mismatch')
    if purpose not in ('ch3_prefix','ch3_formal') and s['prefix_files']:raise ValueError('real prefix forbidden for this purpose')
    if not inside(s['output'],s['session_root']):raise ValueError('output outside current evidence')
    for p,h in s['author_files'].items():
        if sha(p)!=h:raise ValueError('author binding changed: '+p)
    if purpose in ('ch3_cpu','ch3_prefix','ch3_placeholder') and os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only purpose')


def check_access(real,writing,s,deny,permitted_prefix):
    execution=inside(real,s['session_root']) or inside(real,s['fixture_root'])
    if real in s.get('prefix_files',{}) and not writing:
        if permitted_prefix!=real:deny('read',real,'prefix parser capability missing')
        return True
    if inside(real,str(REPO.parent/'amd-execution-evidence')) and not inside(real,s['session_root']):deny('access',real,'old evidence forbidden')
    if not execution and Path(real).suffix.lower() in {'.csv','.pt','.pth','.ckpt','.npy','.npz','.pkl','.h5','.parquet','.safetensors'}:
        deny('access',real,'unapproved observation/checkpoint')
    if inside(real,str(REPO)) and not inside(real,str(TOOL)) and real not in s['bound_files']:
        deny('read',real,'unbound project source')
    for root in s['author_roots']:
        if inside(real,root) and real not in s['author_files']:deny('read',real,'unbound author source')
    return False


def bootstrap(s):
    from m5_entry import BoundFinder
    modules={}
    for path in s['bound_files']:
        p=Path(path);rel=p.relative_to(REPO)
        if p.suffix=='.py':
            name='.'.join(rel.with_suffix('').parts)
            if name.endswith('.__init__'):name=name[:-9]
            if name.startswith('tests.'):name=name[6:]
            modules[name]=str(p)
    c=read_profiles()
    if s.get('task'):
        task=task_by_id(c,s['task'])
        if task['model'] in c['sources']:modules.update(c['sources'][task['model']]['modules'])
    sys.meta_path.insert(0,BoundFinder(modules))
    if s['purpose']=='ch3_placeholder':return
    import torch
    torch.set_num_threads(4)
    from resource_budget import install,counted,instrument_module_calls
    install(s)
    if s['purpose'] in ('ch3_cpu','ch3_prefix'):
        def forbidden(*a,**k):raise PermissionError('CH3 CPU purpose forbids model/optimizer/GPU construction')
        torch.nn.Module.__init__=forbidden;torch.optim.Optimizer.__init__=forbidden;torch.cuda._lazy_init=forbidden
    elif s['purpose']=='ch3_resource_diagnostic':
        def forbidden(*a,**k):raise PermissionError('resource diagnostic forbids models/optimizers/autograd')
        torch.nn.Module.__init__=forbidden;torch.optim.Optimizer.__init__=forbidden
        torch.autograd.backward=forbidden;torch.autograd.grad=forbidden
        # The sole SHA-bound worker below allocates exactly one tensor, no model.
    else:
        instrument_module_calls(torch.nn.Module)
        torch.autograd.backward=counted('backward','backward',torch.autograd.backward)
        torch.optim.Adam.step=counted('adam','Adam',torch.optim.Adam.step)
        if s.get('device')=='cuda:0':
            free,total=torch.cuda.mem_get_info(0);reserve=max(8*1024**3,.1*total)
            if free<=reserve:raise RuntimeError('resource headroom unavailable')
            torch.cuda.set_per_process_memory_fraction(min(.95,(free-reserve)/total),0)


def resource_assessment(sample, pids, baseline=None):
    """Fail closed on unknown competitors; N/A process memory never becomes zero."""
    import math
    reliable=(sample.get('uuid') and all(isinstance(sample.get(k),(int,float)) and math.isfinite(sample[k]) and sample[k]>=0
              for k in ('total','used','free','driver_reserved')) and sample['total']>0
              and abs(sample['total']-sample['used']-sample['free']-sample['driver_reserved'])<=2*1024**2
              and sample.get('process_table_reliable') is True)
    processes=sample.get('nvml_processes',{});matched=set(sample.get('owned_host_pids',[]))
    initial=baseline.get('nvml_processes',{}) if baseline else {}
    unknown=set(processes)-matched-set(initial)
    # With no owned processes this is an observed external-occupancy baseline.
    own=sample.get('process_gpu',{})
    mapped=sample.get('owned_pid_metadata',{})
    external_known=not pids or (baseline is not None and not unknown and baseline.get('uuid')==sample.get('uuid')
                               and all(str(p) in own or mapped.get(str(p),{}).get('host_pid') is not None for p in pids))
    attribution='Measured' if pids and all(own.get(str(p)) is not None for p in pids) else 'Not verified'
    reserve=max(8*1024**3,.1*sample.get('total',0))
    return dict(card_reliable=bool(reliable),external_occupancy_known=external_known,
                process_attribution=attribution,unknown_pids=sorted(unknown) if pids else [],
                admission=bool(reliable and external_known and sample['free']>=reserve),
                measurement='process' if attribution=='Measured' else 'whole-card only',reserve=reserve)


def owned_pid_metadata(pid):
    """Read only the caller's owned PID; sched exposes the kernel PID here.

    Keep starttime stable across reads. Never infer ownership from a new GPU
    table entry or from a memory delta. Missing metadata remains unknown.
    """
    root=Path(f'/proc/{pid}')
    try:
        before=(root/'stat').read_text().rsplit(')',1)[1].split()[19]
        status=(root/'status').read_text();sched=(root/'sched').read_text().splitlines()[0]
        after=(root/'stat').read_text().rsplit(')',1)[1].split()[19]
        match=re.search(r'\((\d+), #threads: \d+\)$',sched)
        if before!=after:raise ValueError('PID lifetime changed during sampling')
        return dict(pid=pid,host_pid=int(match[1]) if match else None,start_ticks=before,sched=sched,
                    nspid=next((x.split()[1:] for x in status.splitlines() if x.startswith('NSpid:')),[]),
                    namespace=os.readlink(root/'ns/pid'),
                    rss=next((int(x.split()[1])*1024 for x in status.splitlines() if x.startswith('VmRSS:')),None))
    except (FileNotFoundError,ProcessLookupError):return dict(pid=pid,host_pid=None,state='exited_during_sample')
    except (PermissionError,ValueError,IndexError):return dict(pid=pid,host_pid=None,state='metadata_unavailable')


def gpu_sample(pids):
    raw=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.total,memory.used,memory.free,memory.reserved','--format=csv,noheader,nounits'],text=True)
    uuid,*memory=raw.strip().split(',');total,used,free,driver_reserved=[float(x.strip())*1024**2 for x in memory]
    processes=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    own={};all_processes={};pid_candidates={};matched=[];table_reliable=True
    metadata={str(pid):owned_pid_metadata(pid) for pid in pids}
    for pid in pids:
        host=metadata[str(pid)]['host_pid']
        pid_candidates[pid]={host} if host is not None else set()
    for line in processes.splitlines():
        parts=line.split(',')
        if len(parts)==2 and parts[0].strip().isdigit():
            host_pid=int(parts[0])
            try:memory=float(parts[1].strip())*1024**2
            except ValueError:memory=None
            if memory is not None and (not math.isfinite(memory) or memory<0):memory=None;table_reliable=False
            all_processes[str(host_pid)]=memory
            for pid,candidates in pid_candidates.items():
                if host_pid in candidates:own[str(pid)]=memory;matched.append(str(host_pid))
        elif line.strip():table_reliable=False
    hosts=[m['host_pid'] for m in metadata.values() if m['host_pid'] is not None]
    if len(set(hosts))!=len(hosts):table_reliable=False
    rss={pid:m['rss'] for pid,m in metadata.items() if m.get('rss') is not None}
    return dict(time=time.monotonic(),uuid=uuid.strip(),total=total,used=used,free=free,driver_reserved=driver_reserved,
                process_gpu=own,cpu_rss=rss,nvml_processes=all_processes,
                owned_host_pids=matched,pid_candidates={str(k):sorted(v) for k,v in pid_candidates.items()},
                process_table_reliable=table_reliable,owned_pid_metadata=metadata,
                raw_gpu=raw,raw_processes=processes,
                process_states={p:('registered' if p in own else m.get('state','alive_not_registered')) for p,m in metadata.items()})


class ExitObservation:
    """A bounded wait for observed own exits, never a cached ownership grant.

    No sample containing a stale PID can grant admission. A later clean sample
    must resolve it; PID reuse, UUID change and unrelated competitors fail closed.
    """
    def __init__(self, baseline):
        self.uuid=baseline['uuid'];self.known={};self.pending={}
    def classify(self,sample,owned,active):
        if sample['uuid']!=self.uuid:raise ValueError('GPU UUID changed')
        now=sample['time'];active={str(p) for p in active};owned={str(p) for p in owned}
        for pid,m in sample.get('owned_pid_metadata',{}).items():
            if pid not in owned:raise ValueError('foreign PID metadata')
            if m.get('host_pid') is not None:
                identity=(str(m['host_pid']),m['start_ticks'])
                if pid in self.known and self.known[pid]!=identity:raise ValueError('owned PID lifetime changed')
                self.known[pid]=identity
        observed=set(sample.get('nvml_processes',{}));waiting=[]
        for pid,(host,start) in self.known.items():
            metadata=sample.get('owned_pid_metadata',{}).get(pid,{})
            if (pid not in active or metadata.get('host_pid') is None) and host in observed:
                deadline=self.pending.setdefault((pid,host,start),now+3.0)
                if now>deadline:raise ValueError('owned exit observation did not settle within 3 seconds')
                waiting.append(host)
            elif pid in active and metadata.get('start_ticks')==start:
                self.pending.pop((pid,host,start),None)
        for key in list(self.pending):
            if key[1] not in observed:del self.pending[key]
        return waiting


def make_config(c,purpose,out,*,task=None,case=None,approval=None,artifact_root=None,resume=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    src=c['sources'].get(task_by_id(c,task)['model']) if task else None
    prefix={}
    if purpose=='ch3_prefix':
        if case not in ('Weather','PJM'):raise ValueError('Weather/PJM prefix only')
        prefix={c['datasets'][case]['path']:{'Weather':42157,'PJM':41933}[case]}
    elif purpose=='ch3_formal':
        t=task_by_id(c,task);d=c['datasets'][t['dataset']]
        if t['dataset']=='UrbanEV':prefix={str(Path(d['path'])/f):c['urban_folds'][t['fold']-1][2] for f in ['volume.csv','e_price.csv','s_price.csv','weather_central.csv']}
        else:prefix[d['path']]=d['endpoints'][2]
    ids=(c['acceptance']['cpu_ids'] if purpose=='ch3_cpu' else
         [c['acceptance']['model_cases'][case]['id']] if purpose=='ch3_model_acceptance' else
         ['ch3.diagnostic.'+case] if purpose=='ch3_step_diagnostic' else
         [task] if task else ['ch3.resource_diagnostic' if purpose=='ch3_resource_diagnostic' else 'ch3.prefix.'+case+'.train_validation' if purpose=='ch3_prefix' else 'ch3.placeholder'])
    limits=dict(adam=0,backward=0,forward=0,seconds=1200)
    if purpose=='ch3_model_acceptance':limits.update(adam=2,forward=6,backward=2)
    elif purpose=='ch3_probe':limits.update(adam=6,backward=6,forward=8,seconds=1800)
    elif purpose=='ch3_step_diagnostic':limits.update(adam=6,backward=6,forward=8,seconds=180)
    elif purpose=='ch3_formal':limits=dict(seconds=None,adam=None,forward=None,backward=None)
    elif purpose=='ch3_resource_diagnostic':limits['seconds']=180
    config=dict(version='restricted-regression-minimal-v3',repo=str(REPO),tool_root=str(TOOL),
                purpose=purpose,case=case,task=task,ids=ids,protocol_sha=digest(c),
                diagnostic='current-profile-six-step' if purpose=='ch3_step_diagnostic' else 'owned_cuda_tensor_16mib' if purpose=='ch3_resource_diagnostic' else 'train_validation_connectivity' if purpose=='ch3_prefix' else None,
                session_root=c['execution']['evidence'],fixture_root=c['execution']['fixture'],
                audit_log=str(out/'audit.jsonl'),budget_file=str(out/'budget.json'),
                output=str(out),limits=limits,bound_files=repository_files(),
                author_roots=[s['repository'] for s in c['sources'].values()],
                author_files=src['files'] if src else {},prefix_files=prefix,forbidden_roots=[],
                device='cuda:0' if purpose in ('ch3_probe','ch3_model_acceptance','ch3_step_diagnostic','ch3_formal','ch3_resource_diagnostic') else 'cpu',approval=approval,
                artifact_root=str(artifact_root or out),resume=resume)
    for dirname in ('cache/torch/kernels','mpl','cuda-cache'):(out/dirname).mkdir(parents=True,exist_ok=True)
    dump(out/'config.json',config)
    from resource_budget import initialize
    initialize(config['budget_file'],purpose,limits)
    return config


def spawn(config):
    out=Path(config['output']);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
       PYTHONPATH=str(TOOL),AMD_RR_CONFIG=str(out/'config.json'),AMD_RR_CONFIG_SHA256=sha(out/'config.json'),
       TMPDIR=config['fixture_root'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
       XDG_CACHE_HOME=str(out/'cache'),MPLCONFIGDIR=str(out/'mpl'),CUDA_CACHE_PATH=str(out/'cuda-cache'))
    if config['device']=='cpu':env['CUDA_VISIBLE_DEVICES']=''
    c=read_profiles();command=[c['execution']['python'],str(Path(__file__).resolve()),'worker']
    dump(out/'command.json',dict(argv=command,env={k:v for k,v in env.items() if os.environ.get(k)!=v},config_sha=sha(out/'config.json')))
    handle=(out/'worker.log').open('x',encoding='utf-8')
    process=subprocess.Popen(command,env=env,cwd=out,stdout=handle,stderr=subprocess.STDOUT)
    return process,handle


def run_configs(configs,out,monitor=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();children=[];samples=[];failure=None;baseline=None
    stop_file=Path(read_profiles()['execution']['evidence'])/'probe'/'STOP'
    def terminate_owned(sig,frame):raise InterruptedError('safe-stop own process tree')
    previous=signal.signal(signal.SIGTERM,terminate_owned);exit_observation=None
    try:
        if monitor:
            baseline=gpu_sample([])
            if not resource_assessment(baseline,[])['admission']:raise MemoryError('whole-card admission unavailable before launch')
            exit_observation=ExitObservation(baseline)
        children=[spawn(c) for c in configs]
        with (out/'memory.jsonl').open('x',encoding='utf-8') as log:
            while any(p.poll() is None for p,_ in children) or (monitor and exit_observation.pending):
                if stop_file.exists():raise InterruptedError('safe-stop; own workers only')
                if time.monotonic()-start>max(c['limits']['seconds'] or 1e12 for c in configs):raise TimeoutError('worker wall time limit')
                if monitor:
                    active=[p.pid for p,_ in children if p.poll() is None]
                    sample=gpu_sample(active)
                    after=[p.pid for p,_ in children if p.poll() is None]
                    waiting=exit_observation.classify(sample,[p.pid for p,_ in children],after)
                    # Exited children are not silently reattributed. During this
                    # bounded transition, check the card and all OTHER unknowns.
                    sample['assessment']=resource_assessment(sample,active or [p.pid for p,_ in children],baseline)
                    unknown=set(sample['assessment']['unknown_pids'])-set(waiting)
                    transient=bool(waiting or active!=after)
                    sample['exit_pending']=waiting;sample['admission_deferred']=transient
                    if transient:sample['assessment']['admission']=False
                    elif not after and not unknown:
                        sample['assessment']=resource_assessment(sample,[],baseline)
                    samples.append(sample);log.write(json.dumps(sample)+'\n');log.flush()
                    if not sample['assessment']['card_reliable']:raise MemoryError('whole-card sampling unreliable')
                    if sample['free']<sample['assessment']['reserve']:raise MemoryError('whole-card headroom crossed')
                    if unknown or (len(configs)>1 and not transient and after and not sample['assessment']['external_occupancy_known']):
                        raise MemoryError('external occupancy unknown; concurrency not admitted')
                time.sleep(.1)
    except (MemoryError,TimeoutError,InterruptedError,subprocess.CalledProcessError,ValueError) as exc:
        failure=('whole-card sampling failed: ' if isinstance(exc,(subprocess.CalledProcessError,ValueError)) else '')+str(exc)
    finally:
        for p,h in children:
            if p.poll() is None:p.terminate()
            p.wait();h.close()
        signal.signal(signal.SIGTERM,previous)
    codes=[p.returncode for p,_ in children]
    intervals=[b['time']-a['time'] for a,b in zip(samples,samples[1:])]
    result=dict(returncodes=codes,elapsed=time.monotonic()-start,failure=failure,
                process_peaks={str(p.pid):max((s['process_gpu'][str(p.pid)] for s in samples if s['process_gpu'].get(str(p.pid)) is not None),default=None) for p,_ in children},
                cpu_peaks={str(p.pid):max((s['cpu_rss'].get(str(p.pid),0) for s in samples),default=None) for p,_ in children},
                sampling='sampled peaks, not continuous maxima',requested_interval=.1,
                actual_interval_min=min(intervals,default=None),actual_interval_max=max(intervals,default=None),
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip() if Path('/sys/fs/cgroup/cpu.max').exists() else
                    {k:Path('/sys/fs/cgroup/cpu/'+k).read_text().strip() for k in ('cpu.cfs_quota_us','cpu.cfs_period_us')})
    result['whole_card_peak']=max((s['used'] for s in samples),default=None)
    settled=[s for s in samples if not s.get('admission_deferred')]
    result['exit_transitions_resolved']=exit_observation is not None and not exit_observation.pending
    result['resource_admission']=bool(settled) and not failure and result['exit_transitions_resolved'] and all(s['assessment']['admission'] for s in settled)
    result['baseline']=baseline
    result['process_attribution']='Measured' if result['process_peaks'] and all(v is not None for v in result['process_peaks'].values()) else 'Not verified'
    failed_logs=[(Path(c['output'])/'worker.log').read_text() for c,code in zip(configs,codes) if code]
    result['failure_kind']=('resource' if failure and any(x in failure for x in ('headroom','whole-card','occupancy')) else
                            'resource' if failed_logs and all('CUDA out of memory' in text for text in failed_logs) else
                            'business' if any(codes) or failure else None)
    dump(out/'process.json',result)
    for c in configs:
        audit=Path(c['audit_log'])
        if audit.exists() and any(json.loads(l)['event']=='denied' for l in audit.read_text().splitlines()):raise RuntimeError('shared guard failure: stop all execution')
    if any(code==86 for code in codes):raise RuntimeError('bootstrap/source identity failed: stop all execution')
    return result


def audit_prefixes(c,out):
    from utils.ch3_data import load,prefix_csv,batches
    results={}
    for domain in c['datasets']:
        try:
            rows=[]
            if domain=='PJM' and c['datasets'][domain]['endpoints'] is None:
                header,times,_,info=prefix_csv(c['datasets'][domain]['path'],1,c['datasets'][domain]['features'])
                results[domain]=dict(header=header,first_time=times[0],prefix=info,fit_policy='F=1 confirmed',blocked='version row count/endpoints, units, forecast as-of')
                continue
            for task in c['tasks']:
                if task['model']!='AMD' or task['dataset']!=domain:continue
                if domain!='UrbanEV' and task['h']!=96:continue
                if domain=='UrbanEV' and task['h']!=3:continue
                ds,meta=load(c,task);p=profile(c,task)
                for split in ['train','validation']:
                    x,y=next(iter(batches(ds[split],p,split)))
                    if x.device.type!='cpu' or y.device.type!='cpu':raise AssertionError('data must stay on CPU')
                meta['run_id']=task['id']
                meta['host_rss']={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith(('VmRSS:','VmHWM:'))}
                meta['all_H_validation_windows']={str(h):((c['urban_folds'][task['fold']-1][1]-c['urban_folds'][task['fold']-1][0]-p['T']-h+1)*275
                    if domain=='UrbanEV' else c['datasets'][domain]['endpoints'][1]-c['datasets'][domain]['endpoints'][0]-h+1) for h in c['datasets'][domain]['horizons']}
                rows.append(meta)
                results[domain]=rows;dump(Path(out)/'prefix-results.json',results)
                del ds,x,y
            results[domain]=rows
        except (ValueError,FileNotFoundError) as exc:
            results[domain]=dict(status='Blocked',reason=str(exc),test_observations_accessed=False)
        dump(Path(out)/'prefix-results.json',results)
    dump(Path(out)/'prefix-results.json',results)


def worker():
    from restricted_io_guard import require_installed
    s=require_installed();c=read_profiles();out=Path(s['output']);purpose=s['purpose']
    if purpose in ('ch3_cpu','ch3_model_acceptance'):
        import unittest
        from run_restricted import execute_cases,flatten
        tests=list(flatten(unittest.defaultTestLoader.loadTestsFromNames(s['ids'])))
        with (out/'ledger.jsonl').open('x',encoding='utf-8') as log:
            result=execute_cases(tests,s['ids'],{},journal=log,per_method_seconds=300,failfast=True)
        dump(out/'result.json',result)
        if not result['success']:raise SystemExit(1)
    elif purpose=='ch3_resource_diagnostic':
        import torch
        start=time.monotonic()
        dump(out/'cpu-ready.json',dict(pid=os.getpid(),metadata=owned_pid_metadata(os.getpid()),time=start))
        while not (out/'GO').exists():
            if time.monotonic()-start>170:raise TimeoutError('diagnostic readiness timeout')
            time.sleep(.05)
        tensor=torch.empty(4*1024*1024,dtype=torch.float32,device='cuda:0')
        tensor.fill_(1.);torch.cuda.synchronize()
        dump(out/'cuda-ready.json',dict(pid=os.getpid(),tensor_bytes=tensor.numel()*tensor.element_size(),
             metadata=owned_pid_metadata(os.getpid()),time=time.monotonic(),allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved()))
        while not (out/'RELEASE').exists():
            if time.monotonic()-start>170:raise TimeoutError('diagnostic hold timeout')
            time.sleep(.05)
        dump(out/'exiting.json',dict(pid=os.getpid(),time=time.monotonic()))
    elif purpose=='ch3_prefix':
        from utils.ch3_data import load,batches,Windows
        import torch
        from torch.utils.data import Subset
        domain=s['case'];task=next(t for t in c['tasks'] if t['model']=='AMD' and t['dataset']==domain)
        datasets,metadata=load(c,task);p=profile(c,task)
        checks={}
        for split in ('train','validation'):
            x,y=next(iter(batches(datasets[split],p,split,torch.Generator().manual_seed(2024))))
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('nonfinite prefix batch')
            checks[split]=dict(input_shape=list(x.shape),target_shape=list(y.shape),finite=True)
        val=datasets['validation'];tail=len(val)%p['training']['eval_batch']
        subset=Subset(val,range(len(val)-tail,len(val))) if tail else val
        x,y=next(iter(batches(subset,p,'validation')))
        checks['validation_tail']=dict(rows=len(x),finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all()))
        if not checks['validation_tail']['finite'] or len(x)!=(tail or p['training']['eval_batch']):raise ValueError('tail mismatch')
        train_end,val_end,test_end=c['datasets'][domain]['endpoints']
        arithmetic={}
        for h in c['datasets'][domain]['horizons']:
            counts=dict(train=train_end-p['T']-h+1,validation=val_end-train_end-h+1,test=test_end-val_end-h+1)
            arithmetic[str(h)]={k:dict(windows=n,full_batches=n//p['training']['batch'],remainder=n%p['training']['batch']) for k,n in counts.items()}
        dump(out/'prefix-result.json',dict(success=True,metadata=metadata,batches=checks,window_arithmetic=arithmetic,
             test_arithmetic_only=True,models=0,adam=0,forward=0,backward=0,gpu=0))
    elif purpose=='ch3_step_diagnostic':
        from ch3_runner import probe_worker
        case=c['execution']['diagnostics']['cases'][s['case']]
        probe_worker(c,task_by_id(c,s['task']),out,capture_states=case['capture_states'])
    elif purpose=='ch3_probe':
        from ch3_runner import probe_worker
        probe_worker(c,task_by_id(c,s['task']),out)
    elif purpose=='ch3_formal':
        from ch3_runner import formal_worker
        formal_worker(c,task_by_id(c,s['task']),Path(s['artifact_root']),s['approval'],s['resume'])
    elif purpose=='ch3_placeholder':
        # Exercise exactly the same fixed-wave dispatch and lock, no torch.
        with GPULock(c):
            try:
                with GPULock(c):raise AssertionError('second model acquired GPU lock')
            except RuntimeError:pass
            dispatch([['a','b'],['c']],lambda run:[sys.executable,'-c','print("placeholder '+run+'")'],out)
        (out/'STOP').write_text('stop before next wave\n')
        try:dispatch([['must_not_start']],lambda run:[sys.executable,'-c','raise AssertionError()'],out)
        except InterruptedError:pass
        else:raise AssertionError('stop flag ignored')
        dump(out/'result.json',dict(success=True,normal_lifecycle=True,second_lock_refused=True,stop_prevents_next_wave=True))


def inherited_trajectory_path(c,parent_path,parent,gid,run):
    """Resolve a reviewed group's direct or already hash-bound ancestor JSON."""
    if parent['decisions'][gid]['status']!='Passed':raise ValueError('failed parent cannot be inherited')
    direct=Path(parent_path).parent/gid/'serial'/run/'trajectory.json'
    if direct.exists():return direct
    record=parent.get('inheritance',{}).get('groups',{}).get(gid,{})
    if not isinstance(record,dict):raise ValueError('ancestor reference is missing')
    ref=record.get('references',{}).get(run)
    if not isinstance(ref,dict):raise ValueError('ancestor run reference is missing')
    path=Path(ref['path']).resolve()
    roots=(REPO.parent/'amd-execution-evidence/m5',Path(c['execution']['fixture']))
    if (path.name!='trajectory.json' or path.parent.name!=run or gid not in path.parts
            or not any(inside(str(path),str(root.resolve()))for root in roots)):
        raise ValueError('ancestor reference outside exact evidence scope')
    if sha(path)!=ref.get('sha256'):raise ValueError('ancestor trajectory changed')
    return path


def probe_all(c,approval):
    reasons=preflight(c,None,approval,probe=True)
    if reasons:raise RuntimeError('; '.join(reasons))
    from utils.ch3_contract import followup_limits
    maximum=followup_limits(c,approval)
    followup=c['execution'].get('followup')
    report=dict(protocol_sha=digest(c),code=code_binding(),hardware=hardware_binding(),environment=environment_binding(),Q=sum(g['q'] for g in c['groups']),decisions={},steps=0)
    if followup:
        parent_path=Path(followup['parent_complete'])
        if sha(parent_path)!=followup['parent_sha256']:raise ValueError('parent evidence changed')
        parent=read(parent_path)
        report['inheritance']=dict(parent=str(parent_path),sha256=followup['parent_sha256'],groups={})
        for gid in followup['inherit_groups']:
            decision=parent['decisions'][gid]
            if decision['status']!='Passed':raise ValueError('cannot inherit failed parent decision')
            group=next(g for g in c['groups'] if g['id']==gid);checks={}
            for run in group['representatives']:
                path=inherited_trajectory_path(c,parent_path,parent,gid,run)
                trajectory=read(path)
                if trajectory['profile_sha']!=digest(profile(c,task_by_id(c,run))):
                    raise ValueError('inherited effective profile differs: '+run)
                checks[run]=dict(path=str(path),sha256=sha(path),profile_sha=trajectory['profile_sha'])
            report['decisions'][gid]=decision
            report['inheritance']['groups'][gid]=dict(status='Reviewed equivalent computation; original result retained',references=checks)
        groups=[g for g in c['groups'] if g['id'] in followup['retest_groups']]
        report['allowance']=dict(original_remaining=followup['remaining_first_adam'],approved_extra=followup['approved_extra_adam'],effective_max=maximum)
    else:groups=c['groups']
    # Scope, source, approval, extra budget and inheritance all checked before
    # creating any output. No rejected preflight can leave an empty run root.
    root=Path(c['execution']['evidence'])/'probe'
    with GPULock(c):
        root.mkdir(exist_ok=False)
        dump(root/'controller.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21]))
        for group in groups:
            group_max=6*group['q']*(1 if group['q']==1 else 3)
            if report['steps']+group_max>maximum:raise RuntimeError('insufficient remaining budget before group launch')
            if group['dataset']=='PJM' and c['datasets']['PJM']['endpoints'] is None:
                report['decisions'][group['id']]=dict(status='Blocked',reason='version endpoints/actual validation tail Not verified',executed_workers=0)
                dump(root/'progress.json',report);continue
            gdir=root/group['id'];gdir.mkdir();serial=[];references=[];blocked=False
            for run in group['representatives']:
                config=make_config(c,'ch3_probe',gdir/'serial'/run,task=run,approval=approval)
                measured=run_configs([config],gdir/'serial-process'/run,monitor=True)
                report['steps']+=read(config['budget_file'])['counts']['adam']
                if measured['failure'] or any(measured['returncodes']):blocked=True;break
                references.append(read(Path(config['output'])/'trajectory.json'));serial.append(measured)
            decision=dict(status='Blocked' if blocked else 'Passed',concurrency=1,representatives=group['representatives'],
                          serial=serial,scope='six updates only; no full-epoch guarantee',parallel='Not tested')
            if not blocked and not all(x['resource_admission'] for x in serial):
                blocked=True
                decision.update(status='ResourceNotVerified',parallel='external occupancy or whole-card sampling uncertain; no permission')
            if not blocked and group['q']>1:
                sample=gpu_sample([])
                # Whole-card serial peaks are conservative upper bounds when
                # container PID namespaces prevent per-process attribution.
                peaks=[max([ref['reserved']]+[v for v in x['process_peaks'].values() if v is not None]+[x['whole_card_peak']])
                       for x,ref in zip(serial,references)]
                if any(v is None for v in peaks):raise RuntimeError('resource measurement binding unavailable')
                need=sum(peaks)
                decision['resource_estimator']='max(torch peak reserved, NVML process/whole-card sampled peaks); whole-card includes external occupancy, attribution separately reported'
                decision['process_attribution']=[x['process_attribution'] for x in serial]
                headroom=max(8*1024**3,.1*sample['total'])
                try_q=group['q'] if resource_assessment(sample,[])['admission'] and sample['free']-need>=headroom else None
                if try_q is None:decision['parallel']='ResourceRejected before launch'
                for concurrency in ([try_q,2] if try_q else [2]):
                    if concurrency is None:continue
                    if concurrency==2 and group['q']<=2 and try_q is not None:break
                    subset_peak=sum(sorted(peaks,reverse=True)[:concurrency])
                    if not resource_assessment(sample,[])['admission'] or sample['free']-subset_peak<headroom:
                        decision[str(concurrency)]='ResourceRejected';continue
                    times=[];observed=[];resource_failure=False
                    for offset in range(0,group['q'],concurrency):
                        runs=group['representatives'][offset:offset+concurrency]
                        configs=[make_config(c,'ch3_probe',gdir/f'q{concurrency}'/run,task=run,approval=approval) for run in runs]
                        measured=run_configs(configs,gdir/f'q{concurrency}-wave{offset}',monitor=True);times.append(measured['elapsed'])
                        for config in configs:report['steps']+=read(config['budget_file'])['counts']['adam']
                        if measured['failure'] or any(measured['returncodes']) or not measured['resource_admission']:resource_failure=True;break
                        observed += [read(Path(x['output'])/'trajectory.json') for x in configs]
                    if resource_failure and measured['failure_kind']!='resource':
                        decision.update(status='Blocked',parallel='model/numerical/lifecycle failure; no lower-concurrency masking');break
                    if resource_failure:
                        decision[str(concurrency)]='ResourceFailed; inspect retained logs';continue
                    fields=['id','profile_sha','initial','initial_rng','batch_ids','trajectory','validation','validation_tail','final','final_rng']
                    if len(observed)!=len(references) or any(any(a[k]!=b[k] for k in fields[:5]) for a,b in zip(references,observed)):
                        raise RuntimeError('shared worker identity/initial RNG mismatch; stop all probe')
                    from ch3_runner import compare_probe_trajectories
                    comparisons=[compare_probe_trajectories(c,task_by_id(c,run),a,b)
                                 for run,a,b in zip(group['representatives'],references,observed)]
                    dump(gdir/f'q{concurrency}-numerical-comparison.json',comparisons)
                    if not all(row['passed'] for row in comparisons):
                        decision.update(status='Blocked',parallel='numerical equivalence failed; no lower-concurrency masking',
                                        numerical_comparisons=comparisons);break
                    speed=sum(x['elapsed'] for x in serial)/sum(times)
                    decision[str(concurrency)]=dict(resource='Passed',trajectory='bounded_numeric' if any(x['mode']=='bounded_numeric' for x in comparisons) else 'exact',
                        numerical_comparisons=comparisons,makespan=sum(times),speedup=speed)
                    if speed>1:
                        decision.update(concurrency=concurrency,parallel='Passed on this one short group');break
            report['decisions'][group['id']]=decision
            dump(root/'progress.json',report)
            if report['steps']>maximum:raise RuntimeError('probe budget exceeded')
        dump(root/'complete.json',report)


def resource_diagnostic(c,attempt,repair_reason):
    """One bounded 1 -> 2 -> 4 live-process diagnostic, not a probe permit."""
    root=Path(c['execution']['evidence'])/f'resource-diagnostic-attempt{attempt}'
    root.mkdir(exist_ok=False);start=time.monotonic();children=[];samples=[];failure=None
    def sample(phase,baseline):
        pids=[p.pid for p,h,s in children if p.poll() is None]
        value=gpu_sample(pids);value.update(phase=phase,assessment=resource_assessment(value,pids,baseline))
        samples.append(value)
        with (root/'samples.jsonl').open('a') as log:log.write(json.dumps(value)+'\n')
        if not value['assessment']['admission']:raise RuntimeError('diagnostic resource admission refused: '+json.dumps(value['assessment']))
        return value
    def wait_ready(path):
        while not path.exists():
            if time.monotonic()-start>=175:raise TimeoutError('diagnostic overall deadline')
            if any(p.poll() is not None for p,h,s in children):raise RuntimeError('diagnostic worker exited before release')
            time.sleep(.05)
    try:
        baseline=gpu_sample([]);dump(root/'baseline.json',baseline)
        if not resource_assessment(baseline,[])['admission'] or baseline['nvml_processes']:
            raise RuntimeError('diagnostic requires safe margin and no external compute processes')
        with GPULock(c):
            for target in (1,2,4):
                for index in range(len(children),target):
                    s=make_config(c,'ch3_resource_diagnostic',root/f'worker-{index+1}')
                    p,h=spawn(s);children.append((p,h,s));out=Path(s['output'])
                    wait_ready(out/'cpu-ready.json');sample(f'cpu-ready-{index+1}',baseline)
                    (out/'GO').write_text('one 16MiB tensor\n');wait_ready(out/'cuda-ready.json')
                for index in range(3):
                    sample(f'cuda-live-{target}',baseline);time.sleep(.1)
            for p,h,s in children:(Path(s['output'])/'RELEASE').write_text('exit owned process\n')
            for p,h,s in children:p.wait(timeout=max(1,180-(time.monotonic()-start)))
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:failure=str(exc)
    finally:
        for p,h,s in children:
            if p.poll() is None:p.terminate()
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            h.close()
    intervals=[b['time']-a['time'] for a,b in zip(samples,samples[1:])]
    result=dict(success=failure is None and len(children)==4 and all(p.returncode==0 for p,h,s in children),
        failure=failure,attempt=attempt,repair_reason=repair_reason,instances=len(children),
        returncodes=[p.returncode for p,h,s in children],pids=[p.pid for p,h,s in children],
        elapsed=time.monotonic()-start,requested_interval=.1,actual_intervals=intervals,
        phases=[v['phase'] for v in samples],sampled_not_continuous=True,
        model=0,adam=0,forward=0,backward=0,tensor_bytes_per_process=16777216,
        process_attribution='Measured' if samples and all(v['assessment']['process_attribution']=='Measured' for v in samples if v['phase'].startswith('cuda-live')) else 'Not verified')
    dump(root/'result.json',result);print(json.dumps(result))
    if not result['success']:raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['worker','cpu-tests','model-tests','resource-diagnostic','weather-prefix','pjm-prefix','placeholder','preflight','dry-run','start','status','logs','complete','safe-stop','formal-worker'])
    parser.add_argument('--approval');parser.add_argument('--run-id');parser.add_argument('--output');parser.add_argument('--attempt',type=int,default=1);parser.add_argument('--repair-reason');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();c=validate_manifest(read_profiles());e=Path(c['execution']['evidence'])
    if args.action=='worker':return worker()
    if args.action in ['status','logs','complete','safe-stop']:
        root=e/'probe'
        if args.action=='safe-stop':(root/'STOP').write_text('user requested safe stop\n')
        complete=(root/'complete.json').exists()
        if complete:validate_probe_report(c,read(root/'complete.json'))
        print(json.dumps(dict(root=str(root),complete=complete,progress=str(root/'progress.json'))));return
    approval=read(args.approval) if args.approval else None
    if args.action in ['preflight','dry-run']:
        reasons=preflight(c,None,approval,True)
        print(json.dumps(dict(groups=len(c['groups']),Q=sum(g['q'] for g in c['groups']),blocked=reasons),indent=2))
        if args.action=='preflight' and reasons:raise SystemExit(2)
        return
    if args.action=='start':return probe_all(c,approval)
    if args.attempt not in (1,2) or args.attempt==2 and not args.repair_reason:raise ValueError('one documented mechanical retry only')
    if args.action=='resource-diagnostic':return resource_diagnostic(c,args.attempt,args.repair_reason)
    if args.action=='model-tests':
        all_results=[];totals=dict(adam=0,forward=0,backward=0)
        for ledger in e.glob('model-attempt*-ledger.json'):
            for k,v in read(ledger)['totals'].items():totals[k]+=v
        prior=dict(totals)
        for name,case in c['acceptance']['model_cases'].items():
            if any(totals[k]+v>c['execution']['model_first'][k] for k,v in dict(adam=2,forward=6,backward=2).items()):
                raise RuntimeError('shared first+retry acceptance allowance exhausted before launch')
            config=make_config(c,'ch3_model_acceptance',e/f'model-attempt{args.attempt}'/name,task=case['run_id'],case=name)
            result=run_configs([config],Path(config['output'])/'process',monitor=True);all_results.append(result)
            for k,v in read(config['budget_file'])['counts'].items():totals[k]+=v
            if any(totals[k]>c['execution']['model_first'][k] for k in totals):raise RuntimeError('shared acceptance budget')
            if result['failure'] or any(result['returncodes']):break
        dump(e/f'model-attempt{args.attempt}-ledger.json',dict(results=all_results,totals={k:totals[k]-prior[k]for k in totals},cumulative=totals,repair_reason=args.repair_reason));return
    purpose={'cpu-tests':'ch3_cpu','weather-prefix':'ch3_prefix','pjm-prefix':'ch3_prefix','placeholder':'ch3_placeholder','formal-worker':'ch3_formal'}[args.action]
    if purpose=='ch3_formal':
        task=task_by_id(c,args.run_id);reasons=preflight(c,task['model'],approval)
        if reasons:raise RuntimeError('; '.join(reasons))
    artifact=Path(args.output or e/f'{args.action}-attempt{args.attempt}')
    if args.resume:
        if purpose!='ch3_formal' or audit_resume(artifact,approval,args.run_id)!='resume':raise PermissionError('explicit legal formal recovery only')
    config=make_config(c,purpose,artifact/('recovery-'+str(time.time_ns())) if args.resume else artifact,
                       task=args.run_id,case={'weather-prefix':'Weather','pjm-prefix':'PJM'}.get(args.action),approval=approval,artifact_root=artifact,resume=args.resume)
    measured=run_configs([config],Path(config['output'])/'process',monitor=purpose=='ch3_formal')
    print(json.dumps(measured))
    if measured['failure'] or any(measured['returncodes']):raise SystemExit(1)


if __name__=='__main__':main()
