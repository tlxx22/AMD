"""CPU-only parameter counts and static tensor byte arithmetic, no forward."""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy
import pypdf
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from timemixer_dev.adapter import build
from timemixer_dev.source import implementation_binding

ROOT = Path.cwd()
EVIDENCE = ROOT / 'docs/evidence/M6/timemixer-v1/implementation-r1'


def tensor(shape):
    n = math.prod(shape)
    return dict(shape=shape, float32_bytes=4*n, MiB=4*n/2**20)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('PYTHONDONTWRITEBYTECODE') != '1' or not sys.dont_write_bytecode:
        raise RuntimeError('CPU/no-bytecode environment required')
    bank = {}
    for f in ('ch3_type1_urban6_four_H_v1.json','ch3_type1_m_all_patchtst_enc2_dm4_selected_v1.json','ch3_type1_epf_all_v3.json'):
        c = json.loads((ROOT/'configs'/f).read_text())
        for p in c['resolved_profiles'].values():
            if p['model']=='TimeMixer':
                bank.setdefault((p['dataset'],p['pred_len']),p)
    rows=[]
    for (dataset,pred_len),p in sorted(bank.items()):
        counts={}
        for name,(u,v) in zip(('B','B+U','B+V','B+U+V'),((False,False),(True,False),(False,True),(True,True))):
            m=build(p,use_u=u,use_v=v)
            counts[name]=sum(t.numel() for t in m.parameters())
        s=p['structure']['down_sampling_layers']+1
        assert counts['B+U']-counts['B']==97*p['T']+4385
        assert counts['B+V']-counts['B']==48*p['structure']['d_model']+16*s+16*pred_len+576
        assert counts['B+U+V']==counts['B+U']+counts['B+V']-counts['B']
        rows.append(dict(dataset=dataset,L=p['T'],C=p['C'],D=p['structure']['d_model'],S=s,P=pred_len,parameters=counts))
    shapes={}
    # b=128 is a static scenario for later review, not a chosen training batch.
    for name,b,c,r,s,p in [('Urban-F4-b2',2,11,1,3,1),('Urban-F4-b128',128,11,1,3,1),('M-Weather-P720-b2',2,21,21,4,720),('M-Weather-P720-b128',128,21,21,4,720)]:
        shapes[name]={
            'U_pair':tensor([b,r,c,32]), 'U_logits':tensor([b,r,c,16]),
            'U_alpha':tensor([b,r,c+1,16]), 'U_core':tensor([b,r,16]),
            'V_pi_and_weights_each':tensor([b,r,s,p]),
            'Y_stack_existing_shape':tensor([b,p,c,s]),
            'V_hidden':tensor([b,r,s,16])}
    value=dict(python=sys.executable,python_version=platform.python_version(),
               torch=torch.__version__,numpy=numpy.__version__,pypdf=pypdf.__version__,
               cpu_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
               environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
               implementation=implementation_binding(),parameters=rows,static_shapes=shapes,
               interpretation='Static float32 tensor arithmetic only: no GPU allocation, measured peak, throughput or concurrency admission. Autograd, copies, activations, optimizer and backbone add memory; do not sum these as peak.')
    (EVIDENCE/'resources.json').write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    selected = [r for r in rows if (r['dataset'],r['P']) in [('UrbanEV',1),('ETTh1',96),('Weather',720),('PJM',24)]]
    print(json.dumps(dict(python=value['python'],torch=value['torch'],parameter_rows=len(rows),selected_rows=selected),indent=2))


if __name__=='__main__':
    main()
