"""One specific cache-only O completion; never consumes a GPU launch AUTH."""
import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from .contract import ROOT,read,sha,load_ref,write_new,verify_sources
from .queue import checked,proc_start
from .reporting import completed

BASE='16e45afe658bc5fdf0ec1ab2801fc893b7ed3866'
PY='/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin/python'
R=ROOT.parent/'amd-execution-evidence/m6'
OLD=R/'moderntcn-uv-v10-fullcoverage-reporting-runs/all275-4h-epf5-sixmodels-first-r1'
OUT=R/'moderntcn-uv-v10-fullcoverage-reporting-runs/all275-4h-epf5-sixmodels-offline-merge-r1'
E=R/'M6-ModernTCN-UV-v10-fullcoverage-offline-merge-repair-r1'


def ref(p):return dict(path=str(p),sha256=sha(p))


def owned_alive(x):
    try:return str(proc_start(x['pid']))==str(x['start'])
    except (FileNotFoundError,ProcessLookupError):return False


def original(binding):
    """Validate immutable original receipts using their actual producer, not new AUTH."""
    r=load_ref(binding['original_runtime_ref']);c=load_ref(r['config_ref']);a=load_ref(r['auth_ref'])
    if binding['original_runtime_ref']['path']!=str(OLD/'runtime.json') or r['commit']!=BASE or a['commit']!=BASE or a['binding']!=c:raise ValueError('exact original runtime/config/AUTH')
    if c['scope']!='m6_ch3_fullcoverage_reporting_v10_v1' or c['attempt']!=OLD.name or len(c['jobs'])!=40 or len(c['waves'])!=40:raise ValueError('exact original40 scope')
    if load_ref(binding['original_failure_ref'])!=read(OLD/'failure.json') or (OLD/'complete.json').exists():raise ValueError('original failed O must remain')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=BASE:raise ValueError('base commit changed')
    for p,h in c['source_binding'].items():
        if p=='m6_ch3_fullcoverage_v10/reporting.py':
            import hashlib
            if hashlib.sha256(subprocess.check_output(['git','show',BASE+':'+p],cwd=ROOT)).hexdigest()!=h:raise ValueError('old reporting source archive')
        elif sha(ROOT/p)!=h:raise ValueError('unrelated original source changed: '+p)
    for p,h in binding['repair_sources'].items():
        if sha(ROOT/p)!=h:raise ValueError('repair source changed: '+p)
    if owned_alive(read(OLD/'owner.json')):raise ValueError('original owner active')
    for p in OLD.glob('children-wave-*.json'):
        if any(owned_alive(x) for x in read(p)):raise ValueError('original child active')
    if {p.stem for p in (OLD/'receipts').glob('*.json')}!={j['id'] for j in c['jobs'][:-1]} or len(list((OLD/'claims').glob('*.json')))!=39:raise ValueError('original39 receipts/claims')
    for i,j in enumerate(c['jobs'][:-1],1):
        checked(c,OLD,j)
        if read(OLD/f'wave-{i}-complete.json')!=dict(jobs=[j['id']]):raise ValueError('original wave')
    verify_sources(load_ref(c['sources_ref']))
    return c


def verify_payload(folder,receipt):
    folder=Path(folder);data=folder/'data'
    if set(receipt['files'])!={str(p.relative_to(folder)) for p in data.rglob('*') if p.is_file()}:raise ValueError('offline payload membership')
    for p,h in receipt['files'].items():
        if not (folder/p).resolve().is_relative_to(data.resolve()) or sha(folder/p)!=h:raise ValueError('offline payload SHA/path')


def verify():
    binding=load_ref(read(OUT/'offline-execution.json')['repair_binding_ref']);c=original(binding)
    z=read(OUT/'complete.json');receipt=read(OUT/'receipts/O.json');verify_payload(OUT,receipt)
    if z.get('logical_jobs')!=40 or z.get('inherited_successful_jobs')!=39 or z.get('new_offline_jobs')!=1 or z.get('new_model_calls')!=0 or z.get('new_test_accesses')!=0:raise ValueError('separate original39/new O accounting')
    if receipt['job']!=c['jobs'][-1] or receipt['offline_execution_ref']!=ref(OUT/'offline-execution.json') or (OUT/'failure.json').exists():raise ValueError('offline receipt identity')
    refs=read(OUT/'inherited-receipts.json')
    if set(refs)!={j['id'] for j in c['jobs'][:-1]}:raise ValueError('inherited39 identity')
    for key,x in refs.items():
        if x!=ref(OLD/'receipts'/(key+'.json')):raise ValueError('original receipt reference')
    if read(OUT/'wave-complete.json')!=dict(execution_wave=1,original_wave=40,jobs=['O']):raise ValueError('one offline wave')
    if len(read(OUT/'data/source-ledger.json'))!=54 or len(read(OUT/'data/region-metrics-6600.json'))!=6600 or len(read(OUT/'data/EPF-position-all.json'))!=720:raise ValueError('complete data coverage')
    with (OUT/'data/region-wide-1100.csv').open() as f:
        if len(list(csv.DictReader(f)))!=1100:raise ValueError('wide row coverage')
    policy=read(OUT/'data/display-policy.json')
    if policy['regions_selected'] or policy['automatic_ranking'] or policy['final_region_models'] is not None or policy['final_EPF_models'] is not None:raise ValueError('no display selection')
    return z


def run():
    import torch
    if sys.executable!=PY or torch.cuda.is_initialized():raise PermissionError('fixed CPU-only interpreter; CUDA must remain uninitialized')
    binding=read(E/'repair-binding.json')
    if binding.get('authorization')!='user_direct_offline_repair_and_completion' or binding.get('new_model_calls')!=0 or binding.get('new_test_accesses')!=0:raise PermissionError('offline-only repair authority')
    c=original(binding);OUT.mkdir(exist_ok=False)
    write_new(OUT/'offline-execution.json',dict(kind='cache_only_technical_completion_not_GPU_runtime',base_commit=BASE,uncommitted_repair_sources=binding['repair_sources'],repair_binding_ref=ref(E/'repair-binding.json'),original_runtime_ref=binding['original_runtime_ref'],python=sys.executable,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),cuda_initialized=False,started=time.time(),new_model_calls=0,new_test_accesses=0))
    try:
        result=completed(c,OUT,source_folder=OLD)
        if torch.cuda.is_initialized():raise RuntimeError('unexpected CUDA initialization')
        write_new(OUT/'inherited-receipts.json',{j['id']:ref(OLD/'receipts'/(j['id']+'.json')) for j in c['jobs'][:-1]})
        write_new(OUT/'receipts/O.json',dict(job=c['jobs'][-1],result=result,offline_execution_ref=ref(OUT/'offline-execution.json'),original_runtime_ref=binding['original_runtime_ref'],files={str(p.relative_to(OUT)):sha(p) for p in (OUT/'data').rglob('*') if p.is_file()}))
        write_new(OUT/'wave-complete.json',dict(execution_wave=1,original_wave=40,jobs=['O']))
        write_new(OUT/'complete.json',dict(technical_complete=True,logical_jobs=40,logical_waves=40,inherited_successful_jobs=39,new_offline_jobs=1,new_offline_waves=1,new_model_calls=0,new_test_accesses=0,new_training=0,cuda_initialized=False,original_failure_preserved=True,original_failure_ref=binding['original_failure_ref'],logical_cells=54,region_cells=6600,region_wide_rows=1100,EPF_position_rows=720,display_selection='deferred',scientific_review='Pending',finished=time.time()))
        return verify()
    except BaseException:
        import traceback
        write_new(OUT/'failure.json',dict(traceback=traceback.format_exc(),new_model_calls=0,new_test_accesses=0));raise


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','verify']);a=p.parse_args()
    print(json.dumps(run() if a.action=='run' else verify(),ensure_ascii=False,indent=2))
