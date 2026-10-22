"""Owned wrapper around the unchanged second-round MS/M numerical loop."""
import os
import time
from pathlib import Path
from .contract import *
from .upstream import live


def guard(c,t,out,approval,resume=False):
    if resume:raise PermissionError('fresh C1 only; retained states require separate recovery review')
    from .launch import preflight
    auth=bound(approval['authorization_ref']);rt=bound(approval['runtime_ref']);preflight(CONFIG,auth,network=False,fresh=False)
    if rt['schema']!='moderntcn-C1-runtime-v1' or rt['commit']!=auth['commit']or rt['config_sha256']!=sha(CONFIG)or rt['source_binding']!=c['source_binding']or rt['authorization_ref']!=approval['authorization_ref']:raise PermissionError('C1 runtime identity')
    if os.getppid()!=rt['owner']['pid']or not live(rt['owner'])or t['id']not in rt['task_ids']:raise PermissionError('C1 owned worker only')
    expected=RESULT_ROOT/ATTEMPT/'staging'/t['id']
    if Path(out)!=expected or Path(out).is_symlink()or (expected/'manifest.json').exists()or (expected/'test-claim.json').exists():raise PermissionError('C1 exact fresh task path')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0'or rt['gpu_identity']['uuid']!=auth['gpu_uuid']:raise PermissionError('C1 GPU mapping')
    if approval['commit']!=auth['commit']or approval['data_bindings']!=c['data_bindings']:raise PermissionError('C1 permit/data mismatch')
    found=False
    for _ in range(50):
        children=read(RESULT_ROOT/ATTEMPT/'children.json')if (RESULT_ROOT/ATTEMPT/'children.json').exists()else []
        if any(x['pid']==os.getpid()and x['task_id']==t['id']and live(x)for x in children):found=True;break
        time.sleep(.1)
    if not found:raise PermissionError('C1 worker absent from owner assignments')

def identity(c,t,metadata,approval):
    return dict(purpose='ch3_formal',revision=ID,backbone_variant='paper_C1_r8_v1',run_id=t['id'],input_variant=t['input_variant'],task=t['task'],metric_scope=t['metric_scope'],protocol_sha=digest(c),profile_sha=digest(profile(c,t)),data_sha=digest(metadata),source=c['source_binding'],commit=approval['commit'],parent=c['parents'][t['id']],from_scratch=True,authorization_ref=approval['authorization_ref'],runtime_ref=approval['runtime_ref'])

def claim_test(out,ident,epoch):
    # Exclusive claim precedes the only native test-loader call. Never auto-retest uncertainty.
    write_new(Path(out)/'test-claim.json',dict(identity_sha=digest(ident),calls=1,best_epoch=epoch))

def run(task_id,runtime_path,runtime_sha):
    from .launch import preflight
    c=read(CONFIG);rt=bound(dict(path=runtime_path,sha256=runtime_sha));auth=bound(rt['authorization_ref']);preflight(CONFIG,auth,network=False,fresh=False)
    t=next(t for t in c['tasks']if t['id']==task_id);folder=RESULT_ROOT/ATTEMPT;out=folder/'staging'/task_id
    if out.exists()or (folder/'runs'/task_id).exists()or (RESULT_ROOT/'registry'/task_id).exists():raise FileExistsError('C1 state/success already exists')
    permit=dict(commit=auth['commit'],data_bindings=c['data_bindings'],authorization_ref=rt['authorization_ref'],runtime_ref=dict(path=runtime_path,sha256=runtime_sha))
    # Validate before data read, directory mutation, model, optimizer or CUDA initialization.
    guard(c,t,out,permit)
    out.mkdir(parents=True)
    for n in ('cache','mpl','cuda-cache'):
        dest=folder/'worker-cache'/task_id/n;dest.mkdir(parents=True,exist_ok=True)
        os.environ[{'cache':'XDG_CACHE_HOME','mpl':'MPLCONFIGDIR','cuda-cache':'CUDA_CACHE_PATH'}[n]]=str(dest)
    os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
    # Preload the exact project dependency closure before author package path adaptation.
    from utils import ch3_data,ch3_time_marks,ch3_m_tasks,ch3_onecycle,feature_schema,dataloader_urbanev
    from models.ch3_adapter import target_prediction
    from ch3_runner import formal_worker
    from .artifacts import publish
    import torch
    torch.set_num_threads(4)
    if torch.__version__!='2.0.1':raise ValueError('second-round torch identity')
    env=dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),dtype='float32',threads=torch.get_num_threads(),cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    write_new(out/'environment.json',env);write_new(out/'command.json',dict(argv=__import__('sys').argv,runtime_ref=permit['runtime_ref']))
    began=time.time()
    try:
        formal_worker(c,t,out,permit,resume=False)
        r=read(out/'result.json');h=[__import__('json').loads(l)for l in (out/'history.jsonl').read_text().splitlines()]
        r.update(revision=ID,backbone_variant='paper_C1_r8_v1',epochs=len(h),optimizer_updates=h[-1]['steps'],scheduler_updates=h[-1]['steps'],test_calls=1)
        atomic(out/'result.json',r)
        write_new(out/'runtime.json',dict(started=began,finished=time.time(),elapsed=time.time()-began,pid=os.getpid(),error=None,timing_scope='native full formal worker including load, training, validation, test, save'))
        receipt=publish(out,folder/'runs'/task_id,c,t,auth['commit'])
        (RESULT_ROOT/'registry').mkdir(exist_ok=True)
        write_new(RESULT_ROOT/'registry'/task_id,receipt);write_new(folder/'receipts'/(task_id+'.json'),receipt)
    except BaseException as exc:
        if out.exists():atomic(out/'failure.json',dict(error=repr(exc),elapsed=time.time()-began,automatic_retry=False))
        raise
