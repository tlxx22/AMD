"""M6 metadata binding and one-monitor-per-wave formal orchestration.
No model execution at import; no final-test parsing during preparation.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
from utils.ch3_contract import digest,profile,step_arithmetic,task_by_id

FREEZE_ID='el-amd-s2-thls-v1'

def source_states(c):
    paths=[]
    for name,d in c['datasets'].items():
        if name=='UrbanEV':paths += [str(Path(d['path'])/f) for f in ('volume.csv','e_price.csv','s_price.csv','weather_central.csv')]
        else:paths.append(d['path'])
    result={}
    for path in paths:
        s=Path(path).stat()
        result[path]=dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
    return result


def metadata_for_horizon(c,task,base,windows):
    """Same loaded/scaled arrays, exact per-H Windows semantics; no new read."""
    from utils.ch3_data import Windows
    p=profile(c,task);meta=copy.deepcopy(base);counts={}
    for split,old in windows.items():
        counts[split]=len(Windows(old.x,old.y,old.start,old.end,p['T'],p['pred_len'],
                                 urban=old.urban,label_horizon=task['h'] if old.urban else None))
    meta['window_counts']=counts
    return meta


def build_data_bindings(c,out):
    """Read each dataset/fold/input prefix once; bind all equivalent model/H tasks."""
    from utils.ch3_data import load
    from ch3_runner import dump
    out=Path(out);out.mkdir(parents=True,exist_ok=True);before=source_states(c)
    banks={}
    for task in c['tasks']:
        key=(task['dataset'],task['fold'],task['input_variant'])
        banks.setdefault(key,[]).append(task)
    bindings={name:{} for name in c['datasets']};sources=[]
    for number,(key,tasks) in enumerate(banks.items()):
        base_task=min(tasks,key=lambda t:t['h']);windows,metadata=load(c,base_task)
        if set(windows)!={'train','validation'} or metadata['test_observations_accessed']:
            raise ValueError('binding preparation accessed test')
        path=out/f'prefix-{number:02d}.json';dump(path,metadata)
        for task in tasks:
            value=metadata_for_horizon(c,task,metadata,windows)
            if value['endpoints_read']!=metadata['endpoints_read']:raise ValueError('prefix boundary changed')
            bindings[task['dataset']][task['id']]=digest(value)
        sources.append(dict(key=list(key),base_task=base_task['id'],path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            tasks=[t['id'] for t in tasks],parsed_end=metadata['endpoints_read'][-1],test_parsed=False))
        del windows,metadata
        print('M6 prefix bound',key,len(tasks),flush=True)
    after=source_states(c)
    if before!=after:raise ValueError('data changed during binding')
    result=dict(protocol_sha=digest(c),data_bindings=bindings,source_states=after,read_groups=sources,
                tasks=sum(map(len,bindings.values())),test_observations_parsed=False,model_updates=0)
    if result['tasks']!=len(c['tasks']):raise ValueError('binding task coverage')
    dump(out/'data-bindings.json',result)
    return result


def approval_reasons(c,model,approval):
    if approval is None:return []
    reasons=[]
    freeze=c.get('structure_freeze',{})
    if freeze.get('id')!=FREEZE_ID or freeze.get('variant')!='J':reasons.append('frozen EL-AMD identity missing')
    if approval.get('freeze_id')!=FREEZE_ID:reasons.append('formal approval freeze identity mismatch')
    tasks=[t for t in c['tasks'] if t['model']==model]
    if model not in approval.get('models',[]):reasons.append('model outside formal authorization')
    for t in tasks:
        h=approval.get('data_bindings',{}).get(t['dataset'],{}).get(t['id'])
        if not isinstance(h,str) or len(h)!=64:reasons.append('missing task data binding: '+t['id'])
    try:
        if source_states(c)!=approval.get('source_states'):reasons.append('formal source file state changed')
    except OSError as exc:reasons.append('formal source stat unavailable: '+str(exc))
    return reasons


def verify_result(c,run,result):
    task=task_by_id(c,run);p=profile(c,task);arith=step_arithmetic(c,task)
    if (result.get('id')!=run or result.get('purpose')!='ch3_formal' or result.get('protocol_sha')!=digest(c)
            or result.get('input_variant')!=task['input_variant'] or result.get('seed')!=2024
            or result.get('metric_space')!='train-standardized'):raise ValueError('formal result identity mismatch')
    if type(result.get('best_epoch')) is not int or not 1<=result['best_epoch']<=p['training']['epochs']:
        raise ValueError('formal best epoch invalid')
    for k in ('mse','mae','sse','sae'):
        if not isinstance(result.get(k),(int,float)) or not math.isfinite(result[k]) or result[k]<0:
            raise ValueError('invalid formal metric')
    expected=arith['test_windows_arithmetic_only']*p['pred_len']
    if type(result.get('elements')) is not int or result['elements']!=expected:raise ValueError('formal full test element count mismatch')
    if result['mse']!=result['sse']/result['elements'] or result['mae']!=result['sae']/result['elements']:
        raise ValueError('formal metric aggregation mismatch')
    return result


def run_formal_waves(c,waves,controller_out,artifact_out,approval,recovery=None):
    """Only one run_configs coordinator owns all workers in each fixed wave."""
    from ch3_runner import dump
    import sys
    tool=Path(__file__).resolve().parents[1]/"tools/restricted_regression"
    if str(tool) not in sys.path:sys.path.insert(0,str(tool))
    from m5_formal_entry import make_config,run_configs
    import time
    controller_out=Path(controller_out);artifact_out=Path(artifact_out);recovery=recovery or {}
    for number,wave in enumerate(waves):
        if (artifact_out/'STOP').exists():raise InterruptedError('formal stop before next wave')
        configs=[]
        for run in wave:
            root=artifact_out/run
            execution=root/('recovery-'+str(time.time_ns())) if recovery.get(run)=='resume' else root
            configs.append(make_config(c,'ch3_formal',execution,task=run,approval=approval,
                                       artifact_root=root,resume=recovery.get(run)=='resume'))
        measured=run_configs(configs,controller_out/f'wave-{number:03d}',monitor=True)
        if measured['failure'] or any(measured['returncodes']) or not measured['resource_admission']:
            raise RuntimeError('formal wave failed; retained artifacts; audit before resume')
        for run in wave:verify_result(c,run,json.loads((artifact_out/run/'result.json').read_text()))
        dump(controller_out/'progress.json',dict(wave=number,completed_wave=wave,total_waves=len(waves),time=time.time()))
