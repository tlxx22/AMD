"""Small synthetic tensors only; no real data or checkpoint access."""
import copy, json, os
from pathlib import Path
import torch
from moderntcn_dev_v3.training_config import ROOT
EVIDENCE=ROOT.parent/'amd-execution-evidence/m6/M6-ModernTCN-UV-v3-target-state-routing-implementation-r1'
STORED=json.loads((ROOT/'configs/moderntcn_uv_fold6_epf_target_state_v3.json').read_text())
def profile(dataset='UrbanEV',variant='F4',H=3):
    p=copy.deepcopy(next(t['structural_source_profile'] for t in STORED['tasks'] if t['dataset']==dataset and t['h']==H))
    if dataset=='UrbanEV':
        from utils.feature_schema import get_feature_schema
        names=list(get_feature_schema(variant).feature_names)
        p.update(features=names,C=len(names),input_variant=variant,aux_idx=list(range(1,len(names))))
    return p
def equal(test,a,b):
    if isinstance(a,torch.Tensor):
        test.assertEqual(a.dtype,b.dtype);test.assertTrue(torch.equal(a,b))
    elif isinstance(a,dict):
        test.assertEqual(set(a),set(b))
        for k in a:equal(test,a[k],b[k])
    elif isinstance(a,(tuple,list)):
        test.assertEqual(type(a),type(b));test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):equal(test,x,y)
    else:test.assertEqual(a,b)
def output(name,value):
    p=EVIDENCE/'tests'/(os.environ['MTCN_TEST_RUN']+'-'+name+'.json')
    with p.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)
