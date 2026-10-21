"""Synthetic-only helpers; exact native dimensions, small batches and fixed CSVs."""
import copy,json,os
from pathlib import Path
import numpy as np,pandas as pd,torch
from moderntcn_dev_v1.training_config import ROOT,validate,arithmetic,digest
from moderntcn_dev_v1.spec import CANDIDATE_ID,REVISION
from moderntcn_dev_v1.gate_policy import policy
from utils.ch3_onecycle import configuration
EVIDENCE=ROOT.parent/'amd-execution-evidence/m6/M6-ModernTCN-UV-v1-implementation-r1'
STORED=json.loads((ROOT/'configs/moderntcn_uv_round2_resolved_v1.json').read_text())
def profile(dataset='UrbanEV',variant='F4',H=3):return copy.deepcopy(next(t['structural_source_profile'] for t in STORED['tasks'] if (t['dataset'],t['input_variant'],t['h'])==(dataset,variant,H)))
def case(name):
    p=EVIDENCE/'synthetic-fixtures'/('v1-'+os.environ.get('MTCN_TEST_RUN',str(os.getpid()))+'-'+name);p.mkdir(parents=True,exist_ok=False);return p

def fixture(folder,arm='UV'):
    folder=Path(folder);t=copy.deepcopy(STORED['tasks'][0]);t.update(arm=arm,use_u=arm in ('U','UV'),use_v=arm in ('V','UV'),id='synthetic-'+CANDIDATE_ID+'-'+arm)
    p=t['structural_source_profile'];p['training'].update(batch=2,eval_batch=2,epochs=2,patience=None,threads=1)
    data=t['data_config'];data['urban_folds']=[[18,36,54]]*6;data['synthetic_nodes']=2;n=54
    path=folder/'urban';path.mkdir(exist_ok=True);clock=pd.date_range('2022-09-01',periods=n,freq='h')
    for name,cols in [('volume.csv',['n0','n1']),('e_price.csv',['n0','n1']),('s_price.csv',['n0','n1']),('weather_central.csv',['T','P','U'])]:
        if (path/name).exists():continue
        rng=np.random.RandomState(22+len(name));frame=pd.DataFrame({c:np.arange(n)/30+np.sin(np.arange(n)/7+i)+rng.randn(n)*.1 for i,c in enumerate(cols)});frame.insert(0,'timestamp',clock.strftime('%Y-%m-%d %H:%M:%S'));frame.to_csv(path/name,index=False)
    data['datasets']['UrbanEV']['path']=str(path);data['datasets']['UrbanEV'].pop('source_admission',None)
    t['data_stat']={};t['baseline_prefix_metadata']={};t['baseline_record_prefix']={}
    p['training']['scheduler']=configuration(2,4);t['profile_sha']=digest(p);t['arithmetic']=arithmetic(data,p,t['fold'])
    c=dict(schema=STORED['schema'],candidate_id=CANDIDATE_ID,structural_revision=copy.deepcopy(REVISION),training_protocol='synthetic_round2_pipeline_v1',purpose='synthetic',synthetic_root=str(folder),tasks=[t],existing_B={},F4_reuse={},gate_policy=policy(),execution_authorized=False)
    validate(c);return c,t['id'],path

def equal(test,a,b):
    if isinstance(a,torch.Tensor):test.assertTrue(torch.equal(a,b));test.assertEqual(a.dtype,b.dtype)
    elif isinstance(a,np.ndarray):test.assertTrue(np.array_equal(a,b))
    elif isinstance(a,dict):
        test.assertEqual(set(a),set(b))
        for k in a:equal(test,a[k],b[k])
    elif isinstance(a,(tuple,list)):
        test.assertEqual(type(a),type(b));test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):equal(test,x,y)
    else:test.assertEqual(a,b)
