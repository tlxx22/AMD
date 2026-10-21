"""v6 r4 deterministic synthetic fixtures only; no real observations/checkpoints."""
import copy,json,os
from pathlib import Path
import numpy as np,pandas as pd,torch
from timemixer_dev_v6_r4.training_config import ROOT,validate,arithmetic,digest
from timemixer_dev_v6_r4.spec import CANDIDATE_ID,REVISION
from timemixer_dev_v6_r4.gate_policy import policy
from utils.ch3_onecycle import configuration

EVIDENCE=ROOT.parent/'amd-execution-evidence/m6/M6-Timemixer-v6-submodule-r4-implementation-r1'
STORED=json.loads((ROOT/'configs/timemixer_uv_round2_resolved_v6_r4.json').read_text())


def profile(dataset='UrbanEV',variant='F4',H=3):
    return copy.deepcopy(next(t['structural_source_profile'] for t in STORED['tasks'] if (t['dataset'],t['input_variant'],t['h'])==(dataset,variant,H)))


def case(name):
    path=EVIDENCE/'synthetic-fixtures'/('r4-'+os.environ.get('M6_R4_TEST_RUN','manual-'+str(os.getpid()))+'-'+name)
    path.mkdir(parents=True,exist_ok=False)
    return path


def fixture(folder,*,arm='UV',variant='F4',H=3,epochs=2,dataset='UrbanEV'):
    folder=Path(folder)
    t=copy.deepcopy(next(t for t in STORED['tasks'] if (t['dataset'],t['input_variant'],t['h'])==(dataset,variant,H)))
    t.update(arm=arm,use_u=arm in ('U','UV'),use_v=arm in ('V','UV'))
    t['id']='synthetic-'+CANDIDATE_ID+'-r4-'+dataset+'-'+variant+'-'+arm+'-h'+str(H)
    p=t['structural_source_profile'];p['structure'].update(d_model=8,d_ff=16,e_layers=1)
    p['training'].update(batch=(32-12-H+1) if dataset=='UrbanEV' else 4,eval_batch=11 if dataset=='UrbanEV' else 4,epochs=epochs,patience=None,threads=1)
    data=t['data_config'];d=data['datasets'][dataset]
    t['data_stat']={};t['baseline_prefix_metadata']={};t['baseline_record_prefix']={}
    if dataset=='UrbanEV':
        data['urban_folds']=[[32,64,96]]*6;data['synthetic_nodes']=2;n=96
        clock=pd.date_range('2022-09-01',periods=n,freq='h');path=folder/'urban';path.mkdir(exist_ok=True)
        for name,cols in [('volume.csv',['n0','n1']),('e_price.csv',['n0','n1']),('s_price.csv',['n0','n1']),('weather_central.csv',['T','P','U'])]:
            if (path/name).exists():continue
            rng=np.random.RandomState(22+len(name));frame=pd.DataFrame({c:np.arange(n)/30+np.sin(np.arange(n)/7+i)+rng.randn(n)*.1 for i,c in enumerate(cols)});frame.insert(0,'timestamp',clock.strftime('%Y-%m-%d %H:%M:%S'));frame.to_csv(path/name,index=False)
    else:
        a,b,c=(200,304,408) if p['pred_len']==96 else (129,161,193);d['endpoints']=[a,b,c];n=c
        clock=pd.date_range('2022-09-01',periods=n,freq='15min' if p['time_mark']['freq']=='t' else 'h');path=folder/'series.csv';rng=np.random.RandomState(381)
        frame=pd.DataFrame({name:np.sin(np.arange(n)/13+i)+np.arange(n)/100+rng.randn(n)*.2 for i,name in enumerate(p['features'])});frame.insert(0,'date',clock.strftime('%Y-%m-%d %H:%M:%S'))
        if not path.exists():frame.to_csv(path,index=False)
    d['path']=str(path);d.pop('source_admission',None)
    p['training']['scheduler']=configuration(epochs,2);t['profile_sha']=digest(p);t['arithmetic']=arithmetic(data,p,t['fold'])
    c=dict(schema=STORED['schema'],candidate_id=CANDIDATE_ID,structural_revision=copy.deepcopy(REVISION),training_protocol='synthetic_round2_pipeline_v1',purpose='synthetic',synthetic_root=str(folder),tasks=[t],existing_B={},F4_reuse={},gate_policy=policy(),execution_authorized=False)
    validate(c);return c,t['id'],path


def nested_equal(test,a,b):
    if isinstance(a,torch.Tensor):test.assertTrue(torch.equal(a,b));test.assertEqual(a.dtype,b.dtype)
    elif isinstance(a,np.ndarray):test.assertTrue(np.array_equal(a,b))
    elif isinstance(a,dict):
        test.assertEqual(set(a),set(b))
        for k in a:nested_equal(test,a[k],b[k])
    elif isinstance(a,(tuple,list)):
        test.assertEqual(type(a),type(b));test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):nested_equal(test,x,y)
    else:test.assertEqual(a,b)


def live(model,opt,sched,generator):
    import random
    return copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),generator=generator.get_state()))
