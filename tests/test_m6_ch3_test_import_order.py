"""Fresh-process CPU regression for author utils namespace switching."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

W=Path(__file__).resolve().parents[1]
E=W.parent/'amd-execution-evidence/m6/M6-ModernTCN-UV-v10-test-reporting-import-order-repair-r1'
CHILD=r'''
import json,sys,importlib.util
from pathlib import Path
import numpy as np
import torch
W=Path.cwd();mode,model=sys.argv[1:];torch.set_num_threads(1)
if mode=='old':
    path=W.parent/'amd-execution-evidence/m6/M6-ModernTCN-UV-v10-test-reporting-import-order-repair-r1/before/evaluate.py'
    spec=importlib.util.spec_from_file_location('m6_ch3_test_reporting_v10._before_repair',path)
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
else:
    from m6_ch3_test_reporting_v10 import evaluate as ev
c=json.loads((W/'configs/m6_ch3_test_reporting_v10.json').read_text())
row=json.loads(Path(c['sources_ref']['path']).read_text())['rows']['PT-NP-'+model]
protocol=json.loads(Path(row['protocol_ref']['path']).read_text())
root=Path(protocol['sources'][model]['root'])
import utils
class Fixture(torch.nn.Module):
    def __init__(self):
        super().__init__();self.bn=torch.nn.BatchNorm1d(3);self.factor=torch.nn.Parameter(torch.tensor(.5))
    def forward(self,x,*marks):return self.bn(x.transpose(1,2)).transpose(1,2)[:,-24:,:]*self.factor
m=Fixture().eval()
for p in m.parameters():p.requires_grad_(False)
x=torch.arange(5*96*3,dtype=torch.float32).reshape(5,96,3)/1000;y=torch.zeros(5,24,1)
marks=torch.zeros(5,96,4) if model=='TimeMixer' else None
parts=(x,y,marks) if marks is not None else (x,y)
def loader():return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*parts),batch_size=2,shuffle=False,generator=torch.Generator().manual_seed(2024))
if mode=='old':
    assert 'utils.ch3_time_marks' not in sys.modules,'old reproduction must start without cached helper'
    utils.__path__=[str(root/'utils')]
    try:ev.infer(m,lambda x,marks:m(x)[:,:,2:3],loader(),24,'cpu',lambda:None,lambda *a:None)
    except ModuleNotFoundError as error:
        assert error.name=='utils.ch3_time_marks';print(json.dumps(dict(expected_failure=str(error),forward_calls=0,fixture_models=1,CUDA_initialized=torch.cuda.is_initialized())));sys.exit(0)
    raise AssertionError('old version unexpectedly survived')
from models.ch3_adapter import target_prediction
predict=lambda x,marks:target_prediction(m,x,row['profile'],marks)[0]
a=ev.infer(m,predict,loader(),24,'cpu',lambda:None,lambda *a:None)
helper=ev.batch_parts
utils.__path__=[str(root/'utils')]
b=ev.infer(m,predict,loader(),24,'cpu',lambda:None,lambda *a:None)
assert ev.batch_parts is helper
assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1]) and a[2:]==b[2:]
assert not torch.cuda.is_initialized()
print(json.dumps(dict(model=model,helper_file=sys.modules['utils.ch3_time_marks'].__file__,author_utils=utils.__path__,exact=True,forward_calls=a[3]+b[3],presentations=len(a[0])+len(b[0]),fixture_models=1,backward=0,Adam=0,real_checkpoint=0,real_test=0,CUDA_initialized=False)))
'''


class ImportOrder(unittest.TestCase):
    def run_child(self,mode,model):
        r=subprocess.run([sys.executable,'-B','-c',CHILD,mode,model],cwd=W,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
        folder=E/'tests';folder.mkdir(exist_ok=True)
        stem=mode+'-'+model;number=len(list(folder.glob(stem+'*.json')))+1
        with (folder/(stem+f'-r{number}.json')).open('x') as f:json.dump(dict(command=[sys.executable,'-B','-c','CHILD in tests/test_m6_ch3_test_import_order.py',mode,model],exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),f,ensure_ascii=False,indent=2)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        result=json.loads(r.stdout.strip().splitlines()[-1]);self.assertFalse(result['CUDA_initialized'])

    def test_old_fresh_process_reproduces(self):self.run_child('old','ModernTCN')
    def test_fixed_modern_and_marks(self):
        for model in ('ModernTCN','TimeMixer'):
            with self.subTest(model=model):self.run_child('fixed',model)


if __name__=='__main__':unittest.main(verbosity=2)
