"""Approved q1/T2 execution resources; no model imports or GPU queries on import."""
import csv
import os
from pathlib import Path
import subprocess
from .runtime_io import read
from .imports import sha

AUTH_BOUND_BLOCKERS=['q1/T2 wiring engineering review pending', 'exact Git closure and actual AUTH pending']
THREAD_ENV={'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1',
            'NUMBA_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}


def worker_environment(base):
    return dict(base,**THREAD_ENV,PYTHONDONTWRITEBYTECODE='1')


def configure_threads(task,resource):
    if task['training']['threads']!=2 or task['training']['workers']!=0:
        raise PermissionError('approved task threads/loader required')
    if resource['threads_per_worker']!=2 or resource['interop_threads']!=1 or resource['thread_environment']!=THREAD_ENV:
        raise PermissionError('bound T2 resource required')
    if any(os.environ.get(k)!=v for k,v in THREAD_ENV.items()):raise PermissionError('worker thread environment mismatch')
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if (torch.get_num_threads(),torch.get_num_interop_threads())!=(2,1):raise PermissionError('actual torch thread mismatch')
    return dict(intra_op=2,inter_op=1,loader_workers=0,environment={k:os.environ[k] for k in THREAD_ENV},
                affinity=sorted(os.sched_getaffinity(0)),project_root=str(Path(__file__).resolve().parents[1]))


def apply_resource_selection(plan,path):
    selection=read(path)
    plan.update(resource_frozen=True,execution_authorized=False,resource_plan=selection,
                resource_selection_reference=dict(path=str(Path(path).resolve()),sha256=sha(path)),
                preparation_blockers=AUTH_BOUND_BLOCKERS+sorted({b for t in plan['tasks'] for b in t['readiness_blockers']}),
                budget_note='216 tasks / 16200 run-epochs; resource q1/T2 fixed; no formal wall-clock hard cap')
    plan['source_binding'][str(Path(path).resolve())]=sha(path)
    for reference in selection['evidence'].values():plan['source_binding'][reference['path']]=reference['sha256']


def validate_resources(plan):
    r=plan['resource_plan'];ids=[t['id'] for t in plan['tasks']]
    if r['waves']!=[[i] for i in ids] or r['test_task_ids']!=ids or r['concurrency']!=1:
        raise ValueError('serial exact original train/test order required')
    if (r['threads_per_worker'],r['interop_threads'],r['loader_workers'])!=(2,1,0) or r['thread_environment']!=THREAD_ENV:
        raise ValueError('q1/T2 resource threads mismatch')
    if r['formal_wall_clock_cap'] is not None or r['automatic_batch_change'] or r['additional_probes']:
        raise ValueError('formal resource boundary')
    if not r['visible_devices'].startswith('GPU-') or r['visible_devices']!=r['expected_gpu_uuid']:
        raise ValueError('bound allocated GPU required')
    for t in plan['tasks']:
        if t['training']['threads']!=2 or t['training']['workers']!=0:raise ValueError('task T2/loader mismatch')
        if [t['training']['batch'],t['training']['eval_batch']]!=r['domain_batches'][t['domain']]:raise ValueError('frozen domain batch mismatch')
    reference=plan['resource_selection_reference']
    if sha(reference['path'])!=reference['sha256'] or read(reference['path'])!=r:
        raise ValueError('resource config identity')
    for ref in r['evidence'].values():
        if plan['source_binding'].get(ref['path'])!=ref['sha256']:raise ValueError('unbound resource adoption evidence')


def verify_gpu_allocation(resource):
    """Startup only, reuse the reviewed single-container UUID/minor contract."""
    requested=os.environ.get('NVIDIA_VISIBLE_DEVICES','')
    if not requested or ',' in requested or requested in ('all','none','void'):
        raise PermissionError('single explicit allocated GPU required')
    def query(args):return subprocess.check_output(['nvidia-smi',*args],text=True,timeout=30).strip()
    rows=list(csv.reader(query(['--query-gpu=uuid','--format=csv,noheader']).splitlines()))
    if len(rows)!=1 or len(rows[0])!=1:raise PermissionError('ambiguous GPU allocation')
    uuid=rows[0][0].strip();matches=[]
    for path in Path('/proc/driver/nvidia/gpus').glob('*/information'):
        info=dict(line.split(':',1) for line in path.read_text().splitlines() if ':' in line)
        if info.get('GPU UUID','').strip()==uuid:matches.append(info)
    if len(matches)!=1 or requested not in (uuid,matches[0].get('Device Minor','').strip()):raise PermissionError('allocation UUID/minor mismatch')
    if uuid!=resource['expected_gpu_uuid'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=uuid:
        raise PermissionError('reviewed GPU identity mismatch')
    processes=query(['-i',uuid,'--query-compute-apps=pid','--format=csv,noheader,nounits'])
    if processes:raise PermissionError('GPU occupied before formal queue')
    free=float(query(['-i',uuid,'--query-gpu=memory.free','--format=csv,noheader,nounits']))
    if not free>=resource['startup_min_free_mib']:raise PermissionError('startup GPU headroom')
    return dict(uuid=uuid,allocation=requested,free_mib=free,compute_processes=[],scope='startup allocation/idle verification; not a resource probe')
