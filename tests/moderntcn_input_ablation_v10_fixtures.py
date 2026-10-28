"""Deterministic synthetic CSVs and profiles; no real observations or states."""
import copy,json,os
from pathlib import Path
import numpy as np
import pandas as pd
from moderntcn_input_ablation_v10.training_config import ROOT,derive,validate,arithmetic,digest
from utils.ch3_onecycle import configuration
E=ROOT.parent/'amd-execution-evidence/m6/M6-ModernTCN-UV-v10-input-ablation-preparation-r1'

def case(name):
    p=E/'synthetic-fixtures'/('v10-input-'+os.environ['MTCN_TEST_RUN']+'-'+name)
    p.mkdir(parents=True,exist_ok=False)
    return p

def fixture(folder,variant='F1',h=3,ends=(16,32,48),epochs=2):
    c=derive();t=copy.deepcopy(next(t for t in c['tasks'] if t['input_variant']==variant and t['h']==h));p=t['structural_source_profile'];tr=p['training']
    tr.update(batch=2,eval_batch=2,epochs=epochs,patience=None,threads=1)
    data=t['data_config'];data['urban_folds']=[list(ends)]*6;data['synthetic_nodes']=2
    path=folder/'urban';path.mkdir(exist_ok=True);n=ends[-1];clock=pd.date_range('2022-09-01',periods=n,freq='h')
    for name,cols in [('volume.csv',['n0','n1']),('e_price.csv',['n0','n1']),('s_price.csv',['n0','n1']),('weather_central.csv',['T','P','U'])]:
        if (path/name).exists():continue
        rng=np.random.RandomState(22+len(name));frame=pd.DataFrame({key:np.arange(n)/30+np.sin(np.arange(n)/7+i)+rng.randn(n)*.1 for i,key in enumerate(cols)});frame.insert(0,'timestamp',clock.strftime('%Y-%m-%d %H:%M:%S'));frame.to_csv(path/name,index=False)
    data['datasets']['UrbanEV']['path']=str(path);data['datasets']['UrbanEV'].pop('source_admission',None)
    tr['scheduler']=configuration(epochs,(ends[0]-12-h+1)*2//2)
    t.update(data_stat={},baseline_prefix_metadata={},baseline_record_prefix={},profile_sha=digest(p),arithmetic=arithmetic(data,p,6))
    c.update(purpose='synthetic',training_protocol='synthetic_round2_pipeline_v1',synthetic_root=str(folder),tasks=[t])
    validate(c);return c,t,path
