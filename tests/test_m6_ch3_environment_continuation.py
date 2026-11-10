"""Import-only and CPU control fixtures; no model/weight/data/GPU execution."""
import ast
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

from m6_ch3_supplement_v10 import environment as env, continuation as co
from m6_ch3_supplement_v10.contract import ROOT, jobs, read, sha, write_new, validate, validate_permission, SCOPE
from m6_ch3_supplement_v10 import queue

E = Path(os.environ['M6_ENV_TEST_EVIDENCE'])
CONFIG = ROOT/'configs/m6_ch3_supplement_v10_retry2.json'


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def e_result(job):
    return dict(job=job, warmup_calls=50, timed_calls=500, batch=job['batch'],
                groups=[dict(group=i, calls=100, wall_seconds=1) for i in range(5)],
                median_batch_ms=10., median_sample_ms=10./job['batch'], stdev_batch_ms=.1,
                parameters=5, peak=dict(max_allocated=10, max_reserved=12), resident=dict(allocated=4))


def fixture(root):
    """Two explicit synthetic producers, never a real runtime or model state."""
    import numpy as np
    old, new = root/'synthetic-old', root/'synthetic-new'
    for p in (old, new):
        p.mkdir();(p/'receipts').mkdir()
    ledger = root/'synthetic-ledger.json'
    write_new(ledger, dict(entries=[dict(kind='baseline' if m != 'UV' else 'uv',
               task=dict(dataset='PJM', h=24, model=m, arm='UV'),
               data=dict(window_counts=dict(validation=4))) for m in ('ModernTCN','TimeMixer','UV')]))
    old_env,new_env=root/'old-environment.json',root/'new-environment.json'
    write_new(old_env,dict(purpose='synthetic_control',producer='old'))
    write_new(new_env,dict(purpose='synthetic_control',producer='new'))
    index_path=root/'inherited.json'
    c=read(CONFIG)
    c.update(purpose='synthetic_control',result_root=str(root),attempt=new.name,ledger_ref=ref(ledger),environment_ref=ref(new_env))
    for v in c['sensitivity_prototypes'].values():
        v['arithmetic']=dict(train_batches=1,eval_batches=dict(validation=1))
    old_config=root/'old-config.json';write_new(old_config,dict(c,attempt=old.name,environment_ref=ref(old_env)))
    old_auth=root/'old-auth.json';write_new(old_auth,dict(purpose='synthetic_control',commit='fixture-old',config_sha256=sha(old_config)))
    write_new(old/'runtime.json',dict(purpose='synthetic_control',scope=SCOPE,commit='fixture-old',config_ref=ref(old_config),auth_ref=ref(old_auth)))
    entries=[]
    for j in jobs()[:12]:
        d=old/j['id'];d.mkdir();r=e_result(j);write_new(d/'result.json',r)
        write_new(old/'receipts'/(j['id']+'.json'),dict(job=j,result=r,files={j['id']+'/result.json':sha(d/'result.json')},runtime_sha256=sha(old/'runtime.json')))
        entries.append(dict(job=j,receipt_ref=ref(old/'receipts'/(j['id']+'.json')),environment_ref=ref(old_env)))
    write_new(index_path,dict(producer_runtime_ref=ref(old/'runtime.json'),entries=entries))
    c['inherited_receipts_ref']=ref(index_path)
    config=root/'new-config.json';write_new(config,c)
    auth=root/'new-auth.json';write_new(auth,dict(purpose='synthetic_control',commit='fixture-new',config_sha256=sha(config)))
    write_new(new/'runtime.json',dict(purpose='synthetic_control',scope=SCOPE,commit='fixture-new',config_ref=ref(config),auth_ref=ref(auth),environment_ref=ref(new_env)))
    for j in jobs()[12:-1]:
        d=new/j['id'];d.mkdir()
        if j['phase']=='E':r=e_result(j)
        elif j['phase']=='S':
            r=dict(epochs=1,updates=1,test_calls=0,status='validation_only_completed',routes=dict(train=1,validation=1),best_epoch=1,validation=dict(mse=.1,mae=.2))
        else:
            count=336 if j['phase']=='C' else 4
            r=dict(presentations=count,forward_calls=(count+127)//128,test=0,metrics=dict(count_per_position=count,mse=[.1]*24,mae=[.2]*24))
            if j['phase']=='C':
                a=np.arange(336,dtype=np.float32).reshape(336,1,1)/100
                np.savez(d/'predictions.npz',truth=a,physical_truth=a,physical_prediction=a+.1)
        write_new(d/'result.json',r)
        write_new(new/'receipts'/(j['id']+'.json'),dict(job=j,result=r,files={str(f.relative_to(new)):sha(f) for f in d.iterdir()},runtime_sha256=sha(new/'runtime.json')))
    write_new(new/'case-selection.json',dict(nodes=['105','408'],target_times=[str(i) for i in range(168)]))
    for i,wave in enumerate(c['execution_waves'],1):
        write_new(new/f'wave-{i}-complete.json',dict(jobs=wave,original_wave_index=i+12))
    return c,old,new


class Repair(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix=self._testMethodName+'-',dir=E))

    def test_01_environment_exact_and_rejections(self):
        c=read(CONFIG);env.verify(c)
        with patch.object(sys,'executable','/public/home/yueweiting/miniconda/envs/amd/bin/python'):
            with self.assertRaises(PermissionError):env.verify(c)
        with patch.object(sys,'prefix','/foreign'):
            with self.assertRaises(PermissionError):env.verify(c)
        real_import=env.importlib.import_module
        def missing(name):
            if name=='reformer_pytorch':raise ModuleNotFoundError('controlled missing dependency')
            return real_import(name)
        with patch.object(env.importlib,'import_module',side_effect=missing):
            with self.assertRaisesRegex(RuntimeError,'reformer_pytorch'):env.verify(c)
        original_sha=env.sha
        with patch.object(env,'sha',side_effect=lambda p:'changed' if str(p).endswith('pyvenv.cfg') else original_sha(p)):
            with self.assertRaises(PermissionError):env.verify(c)
        with patch.object(env,'settings',return_value=dict(cpu_threads=99)):
            with self.assertRaises(PermissionError):env.verify(c)

    def test_02_all_source_imports(self):
        reports=env.check_imports(read(CONFIG))
        write_new(self.root/'imports.json',reports)
        self.assertEqual(len(reports),15)
        for r in reports:
            self.assertEqual(r['exit_code'],0)
            self.assertFalse(json.loads(r['stdout'])['cuda_initialized'])

    def test_03_cpu_child_interpreter_propagation(self):
        code='from m6_ch3_supplement_v10.environment import verify;from m6_ch3_supplement_v10.contract import read;import sys,json;verify(read(sys.argv[1]));print(json.dumps(dict(executable=sys.executable,prefix=sys.prefix)))'
        r=subprocess.run([sys.executable,'-B','-c',code,str(CONFIG)],cwd=ROOT,capture_output=True,text=True,timeout=60)
        write_new(self.root/'child.json',dict(command=[sys.executable,'-B','-c',code,str(CONFIG)],exit=r.returncode,stdout=r.stdout,stderr=r.stderr))
        self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout)['executable'],env.PYTHON)
        text=(ROOT/'scripts/ch3/start_m6_ch3_supplement_v10.sh').read_text()
        self.assertIn('PY='+env.PYTHON,text)
        self.assertIn('subprocess.Popen([sys.executable', (ROOT/'m6_ch3_supplement_v10/queue.py').read_text())

    def test_04_partition_and_budget(self):
        c=read(CONFIG);co.validate_partition(c)
        self.assertEqual(co.execution_jobs(c)[0]['id'],'E-UrbanEV-TimeXer-b1')
        self.assertEqual(len(co.execution_waves(c)),29)
        for key in ['inherited_completed_jobs','execution_jobs','execution_waves','original_wave_indices']:
            bad=copy.deepcopy(c);bad[key]=bad[key][1:]
            with self.assertRaises(ValueError):co.validate_partition(bad)
        bad=copy.deepcopy(c);bad['execution_jobs'][0]=bad['inherited_completed_jobs'][0]
        with self.assertRaises(ValueError):co.validate_partition(bad)
        bad=copy.deepcopy(c);bad['continuation_budget']['max_updates']+=1
        with self.assertRaises(ValueError):co.validate_partition(bad)
        bad=copy.deepcopy(c);del bad['technical_continuation']
        with self.assertRaises(ValueError):validate(bad)
        self.assertEqual(20*550+12*550,17600)
        self.assertEqual(10*(1+16)*550+6*(1+16)*550,149600)

    def test_05_real_sealed_index_and_rejections(self):
        c=read(CONFIG);index=co.validate_retry2(c)
        self.assertEqual(len(index['entries']),12)
        self.assertEqual(sum(len(r['payload']) for r in index['entries']),60)
        for key in ['attempt','gpu_uuid']:
            bad=copy.deepcopy(c);bad[key]='wrong'
            with self.assertRaises(ValueError):co.validate_retry2(bad)
        for key in ['previous_runtime_ref','previous_config_ref']:
            bad=copy.deepcopy(c);bad['technical_continuation'][key]['sha256']='0'*64
            with self.assertRaises(ValueError):co.validate_retry2(bad)
        for kind in ['duplicate','environment','job','payload','result']:
            ix=copy.deepcopy(index)
            if kind=='duplicate':ix['entries'][1]=ix['entries'][0]
            elif kind=='environment':ix['entries'][0]['environment_ref']=c['environment_ref']
            elif kind=='job':ix['entries'][0]['job']['batch']=99
            elif kind=='payload':ix['entries'][0]['payload']['escape']='0'*64
            else:ix['entries'][0]['result_ref']['path']='/foreign/result.json'
            p=self.root/(kind+'.json');write_new(p,ix);bad=dict(c,inherited_receipts_ref=ref(p))
            with self.assertRaises(ValueError):co.validate_retry2(bad)

    def test_06_real_two_source_plots_and_complete(self):
        from m6_ch3_supplement_v10.plots import completed
        c,old,new=fixture(self.root)
        result=completed(c,new)
        self.assertEqual(result['status'],'completed')
        with (new/'plots/efficiency.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),32)
        self.assertEqual(sum(json.loads(r['source_lineage'])['inherited'] for r in rows),12)
        self.assertEqual(len(read(new/'plots/source-ledger.json')),46)
        for r in rows:
            lineage=json.loads(r['source_lineage'])
            self.assertEqual(lineage['producer_commit'],'fixture-old' if lineage['inherited'] else 'fixture-new')
        job=jobs()[-1];write_new(new/'receipts/O.json',dict(job=job,result=result,runtime_sha256=sha(new/'runtime.json'),files={str(f.relative_to(new)):sha(f) for f in (new/'plots').rglob('*') if f.is_file()}))
        summary=co.complete_summary(c,new);write_new(self.root/'synthetic-complete.json',summary)
        self.assertEqual((summary['logical_jobs'],summary['inherited_completed'],summary['execution_completed'],summary['execution_waves_completed']),(47,12,35,29))
        self.assertEqual(summary['fresh_sensitivity'],8)
        with patch.object(co,'validate_retry2',return_value={}):
            status=queue.status(new)
        self.assertEqual((status['inherited_completed'],status['execution_completed'],status['logical_completed']),(12,35,47))
        self.assertEqual(len(list((new/'receipts').glob('*.json'))),35)
        self.assertFalse((new/'receipts/E-UrbanEV-UV-b1.json').exists())
        path=new/'wave-1-complete.json';original=path.read_bytes()
        path.write_text(json.dumps(dict(jobs=c['execution_waves'][0],original_wave_index=1)))
        with self.assertRaises(ValueError):co.complete_summary(c,new)
        path.write_bytes(original)
        write_new(new/'STOP',dict(purpose='synthetic_control'))
        with self.assertRaises(ValueError):co.complete_summary(c,new)

    def test_07_payload_and_runtime_rejection(self):
        c,old,new=fixture(self.root)
        j=jobs()[0]
        co.resolve(c,new,j)
        item=read(c['inherited_receipts_ref']['path'])['entries'][0]
        receipt_path=Path(item['receipt_ref']['path']);original=receipt_path.read_bytes()
        rr=read(receipt_path);rr['runtime_sha256']='wrong'
        write_new(self.root/'receipt-before.json',read(receipt_path))
        receipt_path.write_text(json.dumps(rr))
        with self.assertRaisesRegex(ValueError,'job/runtime'):co.checked_receipt(old,j,ref(old/'runtime.json'),ref(receipt_path))
        receipt_path.write_bytes(original)
        runtime_path=new/'runtime.json';runtime_before=runtime_path.read_bytes();runtime=read(runtime_path)
        runtime['environment_ref']={'path':'/wrong','sha256':'wrong'}
        # Bind the synthetic receipt to the altered synthetic runtime so the
        # environment check, rather than an earlier runtime hash, is exercised.
        runtime_path.write_text(json.dumps(runtime))
        next_job=jobs()[12];rp=new/'receipts'/(next_job['id']+'.json');receipt_before=rp.read_bytes();r=read(rp)
        r['runtime_sha256']=sha(runtime_path);rp.write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError,'environment/attempt'):co.resolve(c,new,next_job)
        runtime_path.write_bytes(runtime_before);rp.write_bytes(receipt_before)
        with self.assertRaises(ValueError):co.resolve(c,new,dict(j,id='outside'))
        (old/j['id']/'result.json').write_text('{}\n')
        with self.assertRaises(ValueError):co.resolve(c,new,j)

    def test_08_no_inherited_dispatch(self):
        c=read(CONFIG)
        with patch.object(queue,'local_context',return_value=c):
            with self.assertRaises(PermissionError):queue.run_job('fixture','fixture','E-UrbanEV-UV-b1')

    def test_09_auth_and_template_refusal(self):
        c=read(CONFIG);auth=read(c['technical_continuation']['previous_auth_ref']['path'])
        with self.assertRaises(PermissionError):validate_permission(CONFIG,auth)
        with self.assertRaises(PermissionError):validate_permission('/not-read',dict(review_approved=False,commit=None))
        # Rejection before source verification/model calls, synthetic permission envelope.
        a=dict(auth,config_sha256=sha(CONFIG),source_binding=c['source_binding'])
        a['resource_plan']=dict(a['resource_plan'],waves=c['execution_waves'])
        with self.assertRaisesRegex(PermissionError,'continuation authorization'):validate_permission(CONFIG,a)

    def test_10_owned_stop(self):
        folder=self.root/'owned-control';folder.mkdir()
        cp=self.root/'control-config.json';write_new(cp,dict(purpose='synthetic_control',result_root=str(self.root),attempt=folder.name))
        write_new(folder/'runtime.json',dict(purpose='synthetic_control',scope=SCOPE,config_ref=ref(cp)))
        write_new(folder/'owner.json',dict(pid=os.getpid(),start=queue.proc_start(os.getpid()),runtime_sha256=sha(folder/'runtime.json')))
        queue.owned_stop(folder);self.assertTrue((folder/'STOP').exists())
        with patch.object(queue,'proc_start',return_value='foreign'):
            with self.assertRaises(ValueError):queue.owned_stop(folder)
        with self.assertRaises(FileExistsError):folder.mkdir(exist_ok=False)

    def test_11_failure_stops_and_admission_placement(self):
        tree=ast.parse((ROOT/'m6_ch3_supplement_v10/queue.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='supervise')
        loop=next(n for n in ast.walk(fn) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple))
        i=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.While))
        class Child:
            def __init__(self,r):self.returncode=r
            def poll(self):return self.returncode
        failed,active=Child(1),Child(None)
        def sleep(_):
            self.assertTrue((self.root/'STOP').exists());active.returncode=1
        scope=dict(children=[('failed',failed),('active',active)],folder=self.root,write_new=write_new,time=Mock(sleep=sleep))
        with self.assertRaises(RuntimeError):exec(compile(ast.fix_missing_locations(ast.Module(body=loop.body[i:i+2],type_ignores=[])),'<CPU-control>','exec'),scope)
        pre=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='preflight')
        text=ast.unparse(pre);self.assertLess(text.index('verify(c)'),text.index("git('rev-parse'"))
        local=ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='local_context'))
        self.assertIn('verify(c)',local);self.assertNotIn('ls-remote',local)
        run=ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_job'))
        self.assertLess(run.index('local_context'),run.index('result = run('))
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('kill','killpg','terminate') for n in ast.walk(tree)))

    def test_12_fixed_math_case_and_readonly_protection(self):
        protected=read(E.parent/'before/project-sha.json')
        for f in ['m6_ch3_supplement_v10/gpu_identity.py','m6_ch3_supplement_v10/model.py','m6_ch3_supplement_v10/benchmark.py','m6_ch3_supplement_v10/sensitivity.py','m6_ch3_supplement_v10/readonly.py','configs/m6_ch3_supplement_v10_retry1.json']:
            self.assertEqual(sha(ROOT/f),protected[f])
        c=read(CONFIG);sel=read(c['technical_continuation']['case_selection_ref']['path'])
        self.assertEqual(sel['nodes'],['105','408']);self.assertEqual(sel['indices'],[2,54])
        self.assertEqual(c['technical_continuation']['case_selection_ref']['sha256'],co.CASE_SHA)


if __name__=='__main__':unittest.main(verbosity=2)
