"""One owned supervisor, fixed task list, reviewed q<=4, no mandatory probe."""
import fcntl
import json
import os
import re
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import uuid
from .training_config import ROOT,validate,digest,sha
from .gate_policy import validate_policy, validate_config_policy

def proc_start(pid):
    text=Path('/proc/'+str(pid)+'/stat').read_text();return text[text.rfind(')')+2:].split()[19]

def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True,timeout=30).strip()

def batch(config,ids):
    if not ids or len(ids)!=len(set(ids)) or not set(ids)<={t['id'] for t in config['tasks']}:raise ValueError('explicit unique nonempty subset required')
    tasks={t['id']:t for t in config['tasks']};groups={}
    for id in ids:
        t=tasks[id];domain='EPF' if t['dataset'] in ('PJM','NP','BE','FR','DE') else t['dataset'];groups.setdefault((domain,t['input_variant'],t['arm']),[]).append(id)
    return [members[i:i+4] for members in groups.values() for i in range(0,len(members),4)]

def validate_authorization(config,config_path,ids,authorization):
    from .worker import source_binding
    if authorization.get('schema')!='timemixer-uv-launch-v2' or authorization.get('review_approved') is not True:raise PermissionError('actual post-closure reviewed dual-policy launch binding required')
    try:
        validate_config_policy(config);validate_policy(authorization.get('gate_policy'))
        if 'main_metric' in authorization or 'gate_main_metric' in authorization or authorization['gate_policy']!=config['gate_policy']:raise ValueError('legacy/mismatched policy')
    except ValueError as error:raise PermissionError('dual gate policy startup contract') from error
    if authorization.get('commit')!=git('rev-parse','HEAD') or git('status','--porcelain=v1'):raise PermissionError('reviewed clean commit required')
    head=git('rev-parse','HEAD')
    if git('rev-parse','origin/M6-Timemixer')!=head or git('ls-remote','--heads','origin','refs/heads/M6-Timemixer').split()[0]!=head:raise PermissionError('three-end source binding')
    if authorization.get('config_sha256')!=sha(config_path) or authorization.get('task_ids')!=ids or authorization.get('source_binding')!=source_binding():raise PermissionError('exact config/tasks/source binding')
    resource=authorization.get('resource_plan',{})
    if resource.get('q_max')!=4 or resource.get('waves')!=batch(config,ids) or resource.get('mandatory_probe') is not False or not resource.get('reviewed_evidence_refs'):raise PermissionError('reviewed exact q4 wave/resource evidence binding required')
    for ref in resource['reviewed_evidence_refs']:
        if sha(ref['path'])!=ref['sha256']:raise PermissionError('resource evidence changed')
    if config.get('source_binding')!=source_binding():raise PermissionError('resolved configuration source changed')
    from .source import verify_author
    verify_author()
    for task in config['tasks']:
        if task['id'] not in ids:continue
        for name,expected in task['data_stat'].items():
            if {k:getattr(Path(name).stat(),k) for k in expected}!=expected:raise PermissionError('startup data stat changed')
    return authorization

def gpu_startup(authorization):
    # Called only by a real, post-review supervisor, never during CPU acceptance.
    last=None
    for n in range(3):
        try:
            lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free','--format=csv,noheader,nounits'],text=True,timeout=60).splitlines()
            fields=next(x.split(',') for x in lines if x.split(',')[0].strip()=='0');index,uid,total,free=[v.strip() for v in fields]
            if uid!=authorization['gpu_uuid'] or float(free)<max(8192,float(total)*.1):raise PermissionError('startup GPU identity/free-memory gate')
            return dict(index=index,uuid=uid,total_MiB=float(total),free_MiB=float(free))
        except subprocess.TimeoutExpired as error:
            last=error
            if n<2:time.sleep(2)
    raise last

def validate_worker_context(config,task_id,context):
    from .worker import source_binding
    if not context:raise PermissionError('owned supervisor runtime required')
    path=Path(context['path'])
    if sha(path)!=context['sha256']:raise PermissionError('runtime changed')
    runtime=json.loads(path.read_text());pid=runtime['owner_pid']
    if proc_start(pid)!=runtime['owner_start'] or os.getppid()!=pid or task_id not in runtime['task_ids']:raise PermissionError('expired/foreign owner or task')
    if runtime['config_sha']!=digest(config) or runtime['source_binding']!=source_binding() or runtime['commit']!=git('rev-parse','HEAD') or git('status','--porcelain=v1'):raise PermissionError('runtime scientific source changed')
    grant=runtime['authorization_ref']
    if sha(grant['path'])!=grant['sha256']:raise PermissionError('authorization changed')
    authorization=json.loads(Path(grant['path']).read_text())
    if authorization.get('schema')!='timemixer-uv-launch-v2' or authorization.get('review_approved') is not True or authorization.get('commit')!=runtime['commit'] or authorization.get('task_ids')!=runtime['task_ids']:raise PermissionError('runtime/grant identity')
    try:
        validate_config_policy(config);validate_policy(runtime.get('gate_policy'));validate_policy(authorization.get('gate_policy'))
        if runtime.get('schema')!='timemixer-uv-owned-runtime-v2' or 'main_metric' in authorization or 'gate_main_metric' in runtime or runtime['gate_policy']!=config['gate_policy'] or authorization['gate_policy']!=runtime['gate_policy']:raise ValueError('runtime policy mismatch')
    except ValueError as error:raise PermissionError('runtime dual gate policy') from error
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or runtime['gpu_identity']['index']!='0' or runtime['gpu_identity']['uuid']!=authorization['gpu_uuid']:raise PermissionError('runtime CUDA mapping/GPU UUID identity')
    if sha(runtime['config_ref']['path'])!=runtime['config_ref']['sha256']:raise PermissionError('runtime config bytes changed')
    return runtime

def prepare_resume(folder,ids,authorization,synthetic=False,*,archive=True,config=None):
    """Only an explicitly reviewed same-attempt epoch-boundary recovery."""
    folder=Path(folder)
    if not folder.is_dir() or folder.is_symlink() or (folder/'complete.json').exists():raise PermissionError('incomplete owned attempt required')
    owner=json.loads((folder/'owner.json').read_text())
    try:live=proc_start(owner['pid'])==owner['start']
    except (FileNotFoundError,ProcessLookupError):live=False
    if live:raise PermissionError('previous owner still running')
    if not synthetic:
        if authorization.get('resume_review_approved') is not True or authorization.get('resume_task_ids')!=ids:raise PermissionError('explicit reviewed resume task subset required')
        ref=authorization['previous_runtime_ref']
        if sha(ref['path'])!=ref['sha256']:raise PermissionError('previous runtime evidence changed')
        previous=json.loads(Path(ref['path']).read_text())
        if previous.get('schema')!='timemixer-uv-owned-runtime-v2':raise PermissionError('resume requires dual policy runtime v2')
        try:
            validate_policy(previous.get('gate_policy'));validate_policy(authorization.get('gate_policy'))
            if previous['gate_policy']!=authorization['gate_policy']:raise ValueError('resume policy mismatch')
        except ValueError as error:raise PermissionError('resume dual gate policy') from error
        if previous['commit']!=authorization['commit'] or not set(ids)<=set(previous['task_ids']):raise PermissionError('resume source/task identity')
        if config is not None:
            from .worker import source_binding
            if owner['runtime_ref']!=ref or previous['owner_pid']!=owner['pid'] or previous['owner_start']!=owner['start'] or previous['config_sha']!=digest(config) or previous['source_binding']!=source_binding():raise PermissionError('resume previous runtime/config/source/owner identity')
    for path in (folder/'runs').glob('*/manifest.json'):
        if json.loads(path.read_text())['task_id'] in ids:raise PermissionError('resume subset includes an already completed task')
    saved={}
    for name in ('owner.json','progress.json','children.json','failure.json','STOP'):
        path=folder/name
        if path.exists():
            if not path.is_file() or path.is_symlink():raise PermissionError('regular resume control files required')
            saved[name]=(path.read_bytes(),path.stat().st_mode & 0o777)
    if archive:archive_resume(folder,saved)
    return saved

def archive_resume(folder,saved):
    """Archive validated old control bytes after the supervisor claims ownership."""
    folder=Path(folder)
    history=folder/'control-history'/uuid.uuid4().hex;history.mkdir(parents=True)
    for name,(content,mode) in saved.items():
        (history/name).write_bytes(content);(history/name).chmod(mode)
    if (folder/'STOP').exists():os.rename(folder/'STOP',history/'STOP')
    if (folder/'failure.json').exists():os.rename(folder/'failure.json',history/'failure-current.json')

def controls_attempt(folder,claim):
    if claim is None:return False
    try:return json.loads((Path(folder)/'owner.json').read_text())==claim and proc_start(claim['pid'])==claim['start']
    except (OSError,ValueError,KeyError):return False

def verify_child_receipt(child,config,folder,runtime,runtime_ref):
    """Audit self-consistency AND bind completion to this exact assignment."""
    from .artifacts import audit
    from .spec import StructuralSpec
    receipt=json.loads(child['receipt'].read_text());path=Path(receipt['path']);runs=Path(folder)/'runs'
    if not path.is_absolute() or path.parent!=runs or runs.is_symlink() or path.resolve().parent!=runs.resolve():raise ValueError('receipt outside assigned attempt publish directory')
    verified=audit(path)
    if verified!=receipt:raise ValueError('wave completion receipt mismatch')
    task=next(t for t in config['tasks'] if t['id']==child['expected_task_id']);m=receipt['manifest'];identity=m['identity']
    if runtime.get('schema')!='timemixer-uv-owned-runtime-v2':raise ValueError('receipt requires dual policy runtime v2')
    if m['task_id']!=child['expected_task_id'] or identity['task']!=task or receipt['metrics']['task_id']!=child['expected_task_id']:raise ValueError('receipt assigned task/arm/F/fold/H/seed mismatch')
    p=task['structural_source_profile'];spec=StructuralSpec.from_profile(p,use_u=task['use_u'],use_v=task['use_v'],task=task['task'],fold=task['fold']).value
    if identity['config_sha']!=digest(config) or identity['profile']!=p or identity['structural_spec']!=spec or identity['source']!=runtime['source_binding'] or identity['producer_commit']!=runtime['commit']:raise ValueError('receipt configuration/source/producer mismatch')
    validate_policy(identity.get('gate_policy'));validate_config_policy(config);validate_policy(runtime.get('gate_policy'))
    if identity.get('schema')!='timemixer-uv-run-identity-v2' or 'gate_main_metric' in identity or identity['purpose']!=config['purpose'] or identity['training_protocol']!=config['training_protocol'] or identity['candidate_id']!=config['candidate_id'] or identity['reference_original'] is not False or identity['gate_policy']!=runtime['gate_policy'] or identity['gate_policy']!=config['gate_policy']:raise ValueError('receipt run identity mismatch')
    expected_device=dict(device='cpu' if runtime['synthetic'] else 'cuda:0',gpu_uuid=None if runtime['synthetic'] else runtime['gpu_identity']['uuid'])
    if identity['execution_device']!=expected_device:raise ValueError('receipt execution device identity mismatch')
    provenance=None if runtime['synthetic'] else dict(runtime_ref=runtime_ref,authorization_ref=runtime['authorization_ref'])
    if m['launch_provenance']!=provenance or identity['environment']['threads']!=p['training']['threads']:raise ValueError('receipt runtime/environment mismatch')
    expected_name=task['dataset']+'-'+task['input_variant']+'-f'+str(task['fold'])+'-h'+str(task['h'])+'-'+task['arm']+'-'+m['identity_sha'][:12]
    if path.name!=expected_name:raise ValueError('receipt publish name identity')
    return m['task_id']

def supervise(config_path,ids,root,attempt,authorization_path=None,*,empty_synthetic=False,synthetic_fixture_path=None,resume=False):
    from .training_state import atomic_json
    from .worker import source_binding,RUN_NAMESPACE,synthetic_root_allowed
    from .artifacts import audit
    root=Path(root).resolve();folder=root/attempt
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,90}',attempt):raise ValueError('attempt syntax')
    if empty_synthetic:
        if ids or not attempt.startswith('synthetic-') or not root.is_relative_to(ROOT.parent/'amd-execution-evidence/m6/M6-Timemixer-runner-integration-r1'):raise ValueError('empty supervisor fixture only')
        config=None;waves=[]
    elif synthetic_fixture_path is not None:
        config=validate(json.loads(Path(config_path).read_text()))
        if config['purpose']!='synthetic' or not synthetic_root_allowed(config['synthetic_root']) or not root.is_relative_to(Path(config['synthetic_root'])) or not Path(synthetic_fixture_path).resolve().is_relative_to(Path(config['synthetic_root'])):raise ValueError('owned CPU fixture namespace required')
        waves=batch(config,ids)
    else:
        config=validate(json.loads(Path(config_path).read_text()))
        if config['purpose']!='development' or not root.is_relative_to(RUN_NAMESPACE):raise PermissionError('independent real namespace/config required')
        authorization=json.loads(Path(authorization_path).read_text())
        validate_authorization(config,config_path,ids,authorization);waves=batch(config,ids)
    if folder.exists() and not resume:raise FileExistsError('fresh supervisor attempt only')
    if resume and empty_synthetic:raise ValueError('empty fixture has no training state to resume')
    # Reuse the existing global GPU mutex; no GPU query occurs for empty fixtures.
    real=config is not None and synthetic_fixture_path is None
    lock=open(config['global_lock'],'a') if real else open(os.devnull,'r')
    children=[];claim=None;completed_ids=[]
    try:
        if real:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        saved=prepare_resume(folder,ids,authorization if real else {},synthetic=not real,archive=False,config=config) if resume else None
        if real:
            if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise PermissionError('real supervisor requires explicit CUDA_VISIBLE_DEVICES=0')
            startup=gpu_startup(authorization)
        runtime=dict(schema='timemixer-uv-owned-runtime-v2',owner_pid=os.getpid(),owner_start=proc_start(os.getpid()),nonce=uuid.uuid4().hex,
                     task_ids=ids,waves=waves,commit=git('rev-parse','HEAD'),config_sha=digest(config) if config else None,source_binding=source_binding(),
                     authorization_ref=dict(path=str(authorization_path),sha256=sha(authorization_path)) if real else None,synthetic=not real,
                     config_ref=dict(path=str(config_path),sha256=sha(config_path)) if config else None,gate_policy=validate_config_policy(config) if config else None,gpu_identity=startup if real else None)
        runtime_path=folder/('runtime-'+runtime['nonce']+'.json')
        runtime_content=json.dumps(runtime,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        import hashlib
        runtime_sha=hashlib.sha256(runtime_content.encode()).hexdigest();runtime_ref=dict(path=str(runtime_path),sha256=runtime_sha)
        if not resume:folder.mkdir(parents=True)
        new_owner=dict(pid=os.getpid(),start=proc_start(os.getpid()),synthetic=not real,runtime_ref=runtime_ref)
        atomic_json(folder/'owner.json',new_owner);claim=new_owner
        if saved is not None:archive_resume(folder,saved)
        atomic_json(runtime_path,runtime)
        (folder/'logs').mkdir(exist_ok=resume);(folder/'receipts').mkdir(exist_ok=resume)
        if real:atomic_json(folder/'startup-hardware.json',startup)
        if empty_synthetic:
            # Persistent empty owned daemon; STOP exercises guardian lifetime.
            atomic_json(folder/'progress.json',dict(state='EMPTY_SYNTHETIC_WAIT',completed=0))
            while not (folder/'STOP').exists():time.sleep(.1)
            atomic_json(folder/'stopped.json',dict(synthetic=True,owned_children=0));return
        for number,wave in enumerate(waves):
            if (folder/'STOP').exists():raise InterruptedError('owned supervisor STOP')
            atomic_json(folder/'progress.json',dict(state='RUNNING',wave=number,task_ids=wave))
            children=[]
            for id in wave:
                receipt=folder/'receipts'/(digest(id)+'-'+runtime['nonce']+'.json');log=(folder/'logs'/(digest(id)+'-'+runtime['nonce']+'.log')).open('x')
                command=[sys.executable,'-B',str(ROOT/'timemixer_train_entry.py'),'worker','--config',str(config_path),'--task-id',id,'--root',str(root),'--attempt',attempt,'--runtime',str(runtime_path),'--runtime-sha',runtime_sha,'--receipt',str(receipt)]
                if resume:command+=['--resume']
                if synthetic_fixture_path is not None:
                    command += ['--synthetic-fixture-path',str(synthetic_fixture_path)]
                    task=next(t for t in config['tasks'] if t['id']==id)
                    if task.get('synthetic_fault'):command+=['--synthetic-fault',task['synthetic_fault']]
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
                children.append(dict(process=child,pid=child.pid,start=proc_start(child.pid),receipt=receipt,expected_task_id=id))
            atomic_json(folder/'children.json',[{k:v for k,v in c.items() if k in ('pid','start','expected_task_id')} for c in children])
            while any(c['process'].poll() is None for c in children):
                if (folder/'STOP').exists() or any(c['process'].poll() not in (None,0) for c in children):raise InterruptedError('STOP or worker failure; next wave forbidden')
                time.sleep(.2)
            wave_completed=[]
            for c in children:
                if c['process'].returncode!=0:raise RuntimeError('worker failed')
                wave_completed.append(verify_child_receipt(c,config,folder,runtime,runtime_ref))
            if len(wave_completed)!=len(set(wave_completed)) or set(wave_completed)!=set(wave) or set(wave_completed)&set(completed_ids):raise ValueError('exact wave completion set mismatch')
            completed_ids.extend(wave_completed)
            children=[]
        if len(completed_ids)!=len(ids) or set(completed_ids)!=set(ids):raise ValueError('exact batch completion set mismatch')
        from .summary import summarize
        result=summarize(config,root,attempt)
        atomic_json(folder/'summary.json',result);atomic_json(folder/'complete.json',dict(technical_complete=True,task_ids=completed_ids,waves=len(waves),scientific_gate=result['scientific_gate']))
    except BaseException as error:
        if not controls_attempt(folder,claim):raise
        if folder.exists():
            (folder/'STOP').touch(exist_ok=True);atomic_json(folder/'failure.json',dict(error=str(error),next_wave_dispatched=False))
        for c in children:
            try:
                if c['process'].poll() is None and proc_start(c['pid'])==c['start']:os.killpg(c['pid'],signal.SIGTERM)
            except (ProcessLookupError,FileNotFoundError):pass
        for c in children:
            try:c['process'].wait(timeout=15)
            except subprocess.TimeoutExpired:
                if proc_start(c['pid'])==c['start']:os.killpg(c['pid'],signal.SIGKILL)
                c['process'].wait()
        raise
    finally:lock.close()

def stop(root,attempt):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,90}',attempt):raise ValueError('attempt syntax')
    folder=Path(root)/attempt
    owner=json.loads((folder/'owner.json').read_text())
    if proc_start(owner['pid'])!=owner['start']:raise PermissionError('expired owned supervisor')
    (folder/'STOP').touch(exist_ok=True)

def status(root,attempt):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,90}',attempt):raise ValueError('attempt syntax')
    folder=Path(root)/attempt
    return {n:json.loads((folder/n).read_text()) for n in ('owner.json','progress.json','failure.json','stopped.json','complete.json') if (folder/n).is_file()}
