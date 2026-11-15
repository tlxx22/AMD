"""PDFormer native DTW/KShape fitted only to approved train prefixes."""
import argparse
import io
import json
import time
import os
import socket
import resource
import zipfile
from pathlib import Path
import numpy as np
from .prepare_data import ROOT,EVIDENCE,prefix,npz_prefix
from .imports import sha
from .native_graphs import train_dtw,fit_patterns
from .runtime_io import write_new,read,proc_start
from .pattern_candidates import validate_candidate_identity
from .kshape_rollback import POLICY


def verified_reference(reference):
    if sha(reference['path'])!=reference['sha256']:raise ValueError('cache reference identity changed')
    return Path(reference['path'])


def reuse_dtw(reference,record,nodes):
    spec=read(verified_reference(reference));path=verified_reference(spec['dtw'])
    original=read(verified_reference(spec['producer_start']))
    for key in ('domain','train_rows','metadata_sha256','node_order','steps_per_day','complete_train_days','dtw_radius'):
        if original[key]!=record[key]:raise ValueError('DTW reuse input/parameter mismatch: '+key)
    if original['source']['sha256']!=record['source']['sha256']:raise ValueError('DTW reuse source data mismatch')
    for path_before,expected in original['code_sha256'].items():
        snapshot=spec['producer_source_snapshots'][path_before]
        if sha(snapshot)!=expected:raise ValueError('DTW producer source snapshot mismatch')
    dtw=np.load(path,allow_pickle=False)
    if dtw.shape!=(nodes,nodes) or not np.isfinite(dtw).all() or (dtw<0).any() or not np.array_equal(dtw,dtw.T) or np.any(np.diag(dtw)!=0):
        raise ValueError('DTW reuse output schema/numeric identity')
    return spec['dtw']


def prepare(domain,destination,*,fixed_reference,candidate_reference,dtw_reference=None,threads=2):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();usage_start=resource.getrusage(resource.RUSAGE_SELF)
    if threads<1 or any(os.environ.get(k)!=str(threads) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS')):
        raise ValueError('explicit identical CPU thread limits required before process imports')
    fixed=read(verified_reference(fixed_reference))
    data=EVIDENCE/'data'/(domain+('-prefix-r1' if domain.startswith('PEMS') else '-prefix-r2'))
    meta=json.loads((data/'metadata.json').read_text());end=meta['endpoints'][0];nodes=meta['node_order']
    selection=validate_candidate_identity(candidate_reference,domain,data/'metadata.json')
    if fixed['identity']['domain']!=domain or fixed['identity']['node_order']!=nodes or fixed['identity']['metadata_sha256']!=sha(data/'metadata.json'):
        raise ValueError('fixed graph/data identity mismatch')
    if domain=='UrbanEV':
        from utils.ch3_data import prefix_csv
        cfg=json.loads((ROOT/'configs/ch3_baseline_ms_u96_oc01_v3.json').read_text())
        path=Path(cfg['datasets']['UrbanEV']['path'])/'volume.csv'
        _,_,raw,access=prefix_csv(path,end,nodes)
        source=dict(path=str(path),sha256=sha(path),access=access)
    elif domain.startswith('PEMS'):
        path=Path(meta['source_npz']['path'])
        if sha(path)!=meta['source_npz']['sha256']:raise ValueError('NPZ identity changed')
        values,access=npz_prefix(path,end);raw=values[...,0]
        source=dict(path=str(path),sha256=sha(path),access=access)
    else:
        path=Path(meta['source_archive']['path']);member=meta['member_base']+'/volume.csv'
        if sha(path)!=meta['source_archive']['sha256']:raise ValueError('archive identity changed')
        with zipfile.ZipFile(path) as z,z.open(member) as f,io.TextIOWrapper(f,encoding='utf-8-sig') as stream:
            frame=prefix(stream,end)
        if list(frame.columns[1:])!=nodes:raise ValueError('train target node identity')
        raw=frame.iloc[:,1:].to_numpy(dtype=float)
        source=dict(path=str(path),sha256=sha(path),member=member,decoded_rows=end)
    for key in ('y','valid'):
        if sha(data/(key+'.npy'))!=meta['files'][key+'.npy']:raise ValueError('train prefix identity changed')
    standardized=np.load(data/'y.npy',mmap_mode='r')[:end]
    valid=np.load(data/'valid.npy',mmap_mode='r')[:end]
    periods=288 if domain.startswith('PEMS') else 24;cand_days=21 if domain=='PEMS08' else 14
    # Urban H3/6/9/12 share the intersection defined by H12, including full train labels.
    legal=selection['legal_train_windows']
    record=dict(domain=domain,source=source,train_rows=end,train_only=True,validation_or_test_fitting=False,
                metadata_sha256=sha(data/'metadata.json'),node_order=nodes,steps_per_day=periods,
                complete_train_days=end//periods,dtw_radius=6,cand_key_days=cand_days,
                candidate_windows=min(cand_days*periods,legal),pattern_length=3,clusters=16,max_iter=5,seed=2024,
                raw_target_for_dtw=True,standardized_target_for_patterns=True,
                source_commit='f8c8f6ad007a04fad3baee958b89504711852ce9',
                fixed_graph_identity=fixed_reference,rollback_policy_version=POLICY,
                candidate_identity=candidate_reference,candidate_selection=selection,
                input_files={key:dict(path=str(data/(key+'.npy')),sha256=meta['files'][key+'.npy']) for key in ('y','valid','times')},
                time_first=str(np.load(data/'times.npy',mmap_mode='r')[0]),train_last=meta['train_last'],
                legal_train_windows=legal,threads=threads,hostname=socket.gethostname(),pid=os.getpid(),start_ticks=proc_start(os.getpid()),
                code_sha256={str(ROOT/'ch4'/name):sha(ROOT/'ch4'/name) for name in ('native_graphs.py','prepare_native_patterns.py','pattern_candidates.py','pattern_initialization.py','kshape_rollback.py')})
    snapshots=destination/'producer-source';snapshots.mkdir()
    for path in record['code_sha256']:
        (snapshots/Path(path).name).write_bytes(Path(path).read_bytes())
    write_new(destination/'pattern-start.json',record)
    def stage(key,value):
        np.save(destination/(key+'.npy'),value,allow_pickle=False)
        print(json.dumps(dict(domain=domain,stage=key,elapsed_seconds=time.monotonic()-started,shape=list(value.shape))),flush=True)
    def costs():
        usage=resource.getrusage(resource.RUSAGE_SELF)
        return dict(wall_seconds=time.monotonic()-started,cpu_user_seconds=usage.ru_utime-usage_start.ru_utime,
                    cpu_system_seconds=usage.ru_stime-usage_start.ru_stime,max_rss_kib=usage.ru_maxrss,cpu_seconds_are_gpu_hours=False)
    try:
        if not np.asarray(valid).all():raise ValueError('native source has no missing-target rule')
        if dtw_reference:
            dtw_ref=reuse_dtw(dtw_reference,record,len(nodes))
            print(json.dumps(dict(domain=domain,stage='reused_dtw',reference=dtw_ref)),flush=True)
        else:
            stage('dtw_matrix',train_dtw(raw,valid,periods))
            dtw_ref=dict(path=str(destination/'dtw_matrix.npy'),sha256=sha(destination/'dtw_matrix.npy'))
        def initialization_stage(centers,details):
            stage('initial_centers',centers)
            write_new(destination/'initialization-identity.json',dict(domain=domain,candidate_identity=candidate_reference,
                initialization=details,centers=dict(path=str(destination/'initial_centers.npy'),sha256=sha(destination/'initial_centers.npy'))))
            record['initialization_identity']=dict(path=str(destination/'initialization-identity.json'),sha256=sha(destination/'initialization-identity.json'))
        def policy_stage(details):
            write_new(destination/'kshape-policy.json',details)
            record['rollback_policy_identity']=dict(path=str(destination/'kshape-policy.json'),sha256=sha(destination/'kshape-policy.json'))
        centers,fit=fit_patterns(standardized,legal,periods,cand_key_days=cand_days,initialization_stage=initialization_stage,policy_stage=policy_stage)
        stage('pattern_keys',centers)
        from fastdtw import fastdtw
        record.update(costs=costs(),fastdtw_implementation=fastdtw.__module__,fit=fit,dtw_reuse=dtw_reference,
                      artifacts=dict(dtw_matrix=dtw_ref,pattern_keys=dict(path=str(destination/'pattern_keys.npy'),sha256=sha(destination/'pattern_keys.npy'))))
        write_new(destination/'pattern-identity.json',record)
        return dict(path=str(destination/'pattern-identity.json'),sha256=sha(destination/'pattern-identity.json'))
    except BaseException as error:
        write_new(destination/'pattern-failure.json',dict(type=type(error).__name__,error=str(error),costs=costs(),successful_fit=False,
                  rollback_policy_identity=record.get('rollback_policy_identity')))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('domain');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fixed-identity',type=Path,required=True);p.add_argument('--candidate-identity',type=Path,required=True)
    p.add_argument('--reuse-dtw',type=Path);p.add_argument('--threads',type=int,default=2)
    args=p.parse_args();reference=lambda path:dict(path=str(path.resolve()),sha256=sha(path))
    prepare(args.domain,args.output,fixed_reference=reference(args.fixed_identity),candidate_reference=reference(args.candidate_identity),
            dtw_reference=reference(args.reuse_dtw) if args.reuse_dtw else None,threads=args.threads)
