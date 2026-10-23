"""Only synthetic CSV fixtures; never reads any real model input/checkpoint."""
import copy,json,os
from pathlib import Path
import numpy as np,pandas as pd,torch
from moderntcn_dev_v2.training_config import ROOT,validate,arithmetic,digest
from moderntcn_dev_v2.spec import CANDIDATE_ID,REVISION
from moderntcn_dev_v2.gate_policy import policy
from moderntcn_dev_v2.training_state import SCHEMA
from utils.ch3_onecycle import configuration
EVIDENCE=ROOT.parent/'amd-execution-evidence/m6/M6-ModernTCN-UV-v2-same-level-parallel-implementation-r1'
STORED=json.loads((ROOT/'configs/moderntcn_uv_fold6_epf_parallel_v2.json').read_text())
def profile(dataset='UrbanEV',variant='F4',H=3):
    p=copy.deepcopy(next(t['structural_source_profile'] for t in STORED['tasks'] if t['dataset']==dataset and t['h']==H))
    if dataset=='UrbanEV':
        from utils.feature_schema import get_feature_schema
        names=list(get_feature_schema(variant).feature_names);p.update(features=names,C=len(names),input_variant=variant,aux_idx=list(range(1,len(names))))
    return p
def case(name):
    p=EVIDENCE/'synthetic-fixtures'/('v2-'+os.environ['MTCN_TEST_RUN']+'-'+name);p.mkdir(parents=True,exist_ok=False);return p
def fixture(folder):
    t=copy.deepcopy(next(t for t in STORED['tasks'] if t['arm']=='UV'));t['id']='synthetic-'+CANDIDATE_ID+'-UV'
    p=t['structural_source_profile'];p['training'].update(batch=2,eval_batch=2,epochs=2,patience=None,threads=1)
    data=t['data_config'];data['urban_folds']=[[18,36,54]]*6;data['synthetic_nodes']=2;n=54
    path=folder/'urban';path.mkdir();clock=pd.date_range('2022-09-01',periods=n,freq='h')
    for name,cols in [('volume.csv',['n0','n1']),('e_price.csv',['n0','n1']),('s_price.csv',['n0','n1']),('weather_central.csv',['T','P','U'])]:
        rng=np.random.RandomState(22+len(name));frame=pd.DataFrame({c:np.arange(n)/30+np.sin(np.arange(n)/7+i)+rng.randn(n)*.1 for i,c in enumerate(cols)});frame.insert(0,'timestamp',clock.strftime('%Y-%m-%d %H:%M:%S'));frame.to_csv(path/name,index=False)
    data['datasets']['UrbanEV']['path']=str(path);data['datasets']['UrbanEV'].pop('source_admission',None)
    t['data_stat']={};t['baseline_prefix_metadata']={};t['baseline_record_prefix']={}
    p['training']['scheduler']=configuration(2,4);t['profile_sha']=digest(p);t['arithmetic']=arithmetic(data,p,t['fold'])
    c=dict(schema=STORED['schema'],candidate_id=CANDIDATE_ID,structural_revision=copy.deepcopy(REVISION),training_protocol='synthetic_round2_pipeline_v1',purpose='synthetic',synthetic_root=str(folder),tasks=[t],existing_B={},F4_reuse={},gate_policy=policy(),execution_authorized=False,checkpoint_schema=SCHEMA)
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
def output(name,obj):
    p=EVIDENCE/'tests'/(os.environ['MTCN_TEST_RUN']+'-'+name+'.json');p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
