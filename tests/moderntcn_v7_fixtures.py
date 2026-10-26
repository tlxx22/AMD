"""Synthetic tensors and frozen metadata only; never load real data/checkpoints."""
import copy, json, os
from pathlib import Path
import torch
from moderntcn_dev_v7.training_config import ROOT
EVIDENCE = ROOT.parent/'amd-execution-evidence/m6/M6-ModernTCN-UV-v7-tied-component-implementation-r1'
STORED = json.loads((ROOT/'configs/moderntcn_uv_fold6_epf_tied_component_v7.json').read_text())

def profile(dataset='UrbanEV', variant='F4', H=3):
    p = copy.deepcopy(next(t['structural_source_profile'] for t in STORED['tasks'] if t['dataset']==dataset and t['h']==H))
    if dataset == 'UrbanEV':
        from utils.feature_schema import get_feature_schema
        names = list(get_feature_schema(variant).feature_names)
        p.update(features=names, C=len(names), input_variant=variant, aux_idx=list(range(1,len(names))))
    return p

def equal(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertEqual(a.dtype,b.dtype); test.assertEqual(a.shape,b.shape)
        test.assertTrue(torch.equal(a,b), 'exact tensor comparison')
    elif isinstance(a, dict):
        test.assertEqual(set(a),set(b))
        for k in a: equal(test,a[k],b[k])
    elif isinstance(a, (tuple,list)):
        test.assertEqual(type(a),type(b)); test.assertEqual(len(a),len(b))
        for x,y in zip(a,b): equal(test,x,y)
    else: test.assertEqual(a,b)

def output(name, value):
    p = EVIDENCE/'tests'/(os.environ['MTCN_TEST_RUN']+'-'+name+'.json')
    with p.open('x') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def open_branches(m):
    with torch.no_grad():
        if m.use_u:
            w=m.u.W_coef
            w.copy_((torch.arange(w.numel(),dtype=w.dtype).reshape_as(w)%11-5)*.02)
        if m.use_v:
            m.v.A.fill_(.05); m.v.P_s.fill_(.03)
