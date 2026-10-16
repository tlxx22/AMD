"""Explicitly synthetic v2 fixtures; no derivation or reads of real observations."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from timemixer_dev_v2.training_config import ROOT, validate, arithmetic, digest
from timemixer_dev_v2.spec import CANDIDATE_ID, REVISION
from timemixer_dev_v2.gate_policy import policy
from utils.ch3_onecycle import configuration

STORED = json.loads((ROOT/'configs/timemixer_uv_round2_resolved_v2.json').read_text())
EVIDENCE = ROOT.parent/'amd-execution-evidence/m6/M6-Timemixer-v2-implementation-r1'
CHILD_UPDATES = []


def run_cpu_supervisor(command, *, cwd, timeout):
    """Reviewed repair-r1 CPU fork fixture, adapted only to the v2 CLI path.

    The supervisor/worker execute the actual entry with runpy in forked CPU
    processes. This is not a cold exec queue claim; the cold failure is retained.
    """
    import multiprocessing
    import os
    import subprocess
    import sys
    import tempfile
    import time
    import traceback
    from unittest.mock import patch
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized() or '--synthetic-fixture-path' not in command or timeout!=45:
        raise AssertionError('bounded CPU fixture only')
    context=multiprocessing.get_context('fork');real_popen=subprocess.Popen
    def entry(argv,stdout,stderr,directory,new_session):
        try:
            if new_session:os.setsid()
            os.chdir(directory);os.dup2(stdout.fileno(),1);os.dup2(stderr.fileno(),2);sys.argv=argv[2:]
            import runpy
            runpy.run_path(argv[2],run_name='__main__')
        except SystemExit as error:
            if error.code not in (None,0):raise
        except BaseException:
            traceback.print_exc();raise
        finally:sys.stdout.flush();sys.stderr.flush()
    class WorkerProcess:
        def __init__(self,argv,**kwargs):
            stderr=kwargs['stdout'] if kwargs['stderr']==subprocess.STDOUT else kwargs['stderr']
            self.argv=argv;self.process=context.Process(target=entry,args=(argv,kwargs['stdout'],stderr,kwargs['cwd'],kwargs.get('start_new_session',False)))
            self.process.start();self.pid=self.process.pid
        @property
        def returncode(self):return self.process.exitcode
        def poll(self):self.process.join(0);return self.returncode
        def wait(self,timeout=None):
            self.process.join(timeout)
            if self.process.is_alive():raise subprocess.TimeoutExpired(self.argv,timeout)
            return self.returncode
    def factory(argv,**kwargs):
        if '--receipt' in argv and '--synthetic-fixture-path' in argv:return WorkerProcess(argv,**kwargs)
        return real_popen(argv,**kwargs)
    with tempfile.TemporaryFile() as out,tempfile.TemporaryFile() as err:
        def supervisor():
            with patch.object(subprocess,'Popen',side_effect=factory):entry(command,out,err,cwd,False)
        process=context.Process(target=supervisor);start=time.monotonic();process.start();process.join(timeout)
        if process.is_alive():
            process.kill();process.join();raise subprocess.TimeoutExpired(command,timeout)
        if time.monotonic()-start>timeout:raise subprocess.TimeoutExpired(command,timeout)
        out.seek(0);err.seek(0)
        return subprocess.CompletedProcess(command,process.exitcode,out.read().decode(),err.read().decode())


def profile(dataset='UrbanEV', variant='F4', H=3):
    return copy.deepcopy(next(t['structural_source_profile'] for t in STORED['tasks']
                             if (t['dataset'],t['input_variant'],t['h']) == (dataset,variant,H)))


def small_profile(dataset='UrbanEV', variant='F4', H=3):
    p = profile(dataset,variant,H)
    p['structure'].update(d_model=8,d_ff=16,e_layers=1)
    return p


def fixture(folder, dataset='UrbanEV', variant='F4', arm='UV', H=3, epochs=2, patience=None):
    folder=Path(folder)
    t=copy.deepcopy(next(t for t in STORED['tasks'] if (t['dataset'],t['input_variant'],t['h'])==(dataset,variant,H)))
    t.update(arm=arm,use_u=arm in ('U','UV'),use_v=arm in ('V','UV'))
    t['id']='synthetic-'+CANDIDATE_ID+'-'+dataset+'-'+variant+'-'+arm+'-h'+str(H)
    p=t['structural_source_profile'];p['structure'].update(d_model=8,d_ff=16,e_layers=1)
    tr=p['training'];tr.update(batch=(32-12-H+1) if dataset=='UrbanEV' else 4,
                              eval_batch=11 if dataset=='UrbanEV' else 4,epochs=epochs,patience=patience,threads=1)
    data=t['data_config'];d=data['datasets'][dataset]
    t['data_stat']={};t['baseline_prefix_metadata']={};t['baseline_record_prefix']={}
    folder.mkdir(parents=True,exist_ok=True)
    if dataset=='UrbanEV':
        data['urban_folds']=[[32,64,96]]*6;data['synthetic_nodes']=2;n=96
        clock=pd.date_range('2022-09-01',periods=n,freq='h');path=folder/'urban';path.mkdir(exist_ok=True)
        for name,cols in [('volume.csv',['n0','n1']),('e_price.csv',['n0','n1']),('s_price.csv',['n0','n1']),('weather_central.csv',['T','P','U'])]:
            rng=np.random.RandomState(22+len(name))
            frame=pd.DataFrame({c:np.arange(n)/30+np.sin(np.arange(n)/7+i)+rng.randn(n)*.1 for i,c in enumerate(cols)})
            frame.insert(0,'timestamp',clock.strftime('%Y-%m-%d %H:%M:%S'));frame.to_csv(path/name,index=False)
    else:
        a,b,c=(200,304,408) if p['pred_len']==96 else (129,161,193)
        d['endpoints']=[a,b,c];n=c
        clock=pd.date_range('2022-09-01',periods=n,freq='15min' if p['time_mark']['freq']=='t' else 'h')
        path=folder/'series.csv';rng=np.random.RandomState(381)
        frame=pd.DataFrame({name:np.sin(np.arange(n)/13+i)+np.arange(n)/100+rng.randn(n)*.2 for i,name in enumerate(p['features'])})
        frame.insert(0,'date',clock.strftime('%Y-%m-%d %H:%M:%S'));frame.to_csv(path,index=False)
    d['path']=str(path);d.pop('source_admission',None)
    tr['scheduler']=configuration(epochs,2);t['profile_sha']=digest(p);t['arithmetic']=arithmetic(data,p,t['fold'])
    c=dict(schema=STORED['schema'],candidate_id=CANDIDATE_ID,structural_revision=copy.deepcopy(REVISION),
           training_protocol='synthetic_round2_pipeline_v1',purpose='synthetic',synthetic_root=str(folder),
           tasks=[t],existing_B={},F4_reuse={},gate_policy=policy(),execution_authorized=False)
    validate(c)
    return c,t['id'],path


def nested_equal(test,a,b):
    if isinstance(a,torch.Tensor):
        test.assertTrue(torch.equal(a,b));test.assertEqual(a.dtype,b.dtype)
    elif isinstance(a,np.ndarray):test.assertTrue(np.array_equal(a,b))
    elif isinstance(a,dict):
        test.assertEqual(set(a),set(b))
        for k in a:nested_equal(test,a[k],b[k])
    elif isinstance(a,(tuple,list)):
        test.assertEqual(type(a),type(b));test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):nested_equal(test,x,y)
    else:test.assertEqual(a,b)


def live_state(model,opt,schedule,generator):
    import random
    return copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=schedule.state_dict(),
        python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),generator=generator.get_state()))
