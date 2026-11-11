"""CPU-only controls and synthetic caches for the explicit retained-eight retry."""
import ast
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from m6_ch3_test_reporting_v10 import contract as k, continuation as cont, queue

W=k.ROOT;R=W.parent/'amd-execution-evidence/m6'
E=R/'M6-ModernTCN-UV-v10-test-reporting-import-continuation-r1'
CONFIG=W/'configs/m6_ch3_test_reporting_v10_import_retry1.json'


def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
    return dict(path=str(p),sha256=k.sha(p))


class Continued(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=k.read(CONFIG);cls.tmp=tempfile.TemporaryDirectory(prefix='CPU-controls-',dir=E/'tests');cls.root=Path(cls.tmp.name)
    @classmethod
    def tearDownClass(cls):cls.tmp._finalizer.detach()

    def test_01_exact_scope_inheritance_and_budget(self):
        c=k.validate(self.c);self.assertEqual(len(cont.execution_jobs(c)),12);self.assertEqual(len(cont.execution_waves(c)),13)
        self.assertEqual(cont.execution_jobs(c)[0]['id'],'PT-NP-ModernTCN')
        self.assertEqual(sum(j['expected']['batches'] for j in cont.execution_jobs(c)),780)
        for key,value in [('execution_jobs',c['jobs'][7:]),('execution_waves',c['waves']),('original_wave_indices',list(range(1,14))),('numeric_policy',{'atol':1.,'rtol':1.,'identity':'exact'})]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):k.validate(bad)
        bad=copy.deepcopy(c);bad['attempt']='test-reporting-v10-import-retry2'
        with self.assertRaises(ValueError):k.validate(bad)
        bad=copy.deepcopy(c);bad['jobs'][8]['dataset']='PJM'
        with self.assertRaises(ValueError):k.validate(bad)

    def test_02_no_rerun_and_old_auth_rejected(self):
        c=self.c;folder=self.root/'no-rerun';folder.mkdir();runtime=folder/'runtime.json';dump(runtime,{'synthetic':True})
        job=c['jobs'][0]
        with self.assertRaises(PermissionError):queue.claim(folder,c,job,runtime)
        with self.assertRaises(PermissionError):k.require_evaluation(c,job,dict(scope=k.SCOPE,purpose=k.PURPOSE,job=job['id'],evaluation_key=k.evaluation_key(c,job),mode='authorized_frozen_test'))
        self.assertFalse((folder/'claims').exists())
        old=k.load_ref(c['technical_continuation']['auth_ref'])
        with self.assertRaises(PermissionError):k.permission(CONFIG,old)
        template=k.read(E/'non-execution-preparation/launch-binding.template.json')
        with self.assertRaises(PermissionError):k.permission(CONFIG,template)

    def test_03_source_identity_and_tamper(self):
        c=self.c;old=Path(c['technical_continuation']['runtime_ref']['path']).parent
        for j in c['jobs'][:8]:
            folder,runtime,inherited=cont.location(c,self.root,j);self.assertEqual(folder,old);self.assertTrue(inherited)
            self.assertEqual(cont.receipt(c,self.root,j)['job'],j)
        folder,runtime,inherited=cont.location(c,self.root,c['jobs'][8]);self.assertEqual(folder,self.root);self.assertFalse(inherited)
        for key in ['runtime_ref','config_ref','auth_ref','inherited_ref','failure_ref','case_plan_ref']:
            bad=copy.deepcopy(c);bad['technical_continuation'][key]['sha256']='0'*64
            with self.assertRaises((ValueError,PermissionError,RuntimeError)):k.validate(bad)
        index=k.load_ref(c['technical_continuation']['inherited_ref']);index['rows'][1]=index['rows'][0]
        bad=copy.deepcopy(c);bad['technical_continuation']['inherited_ref']=dump(self.root/'duplicate-eight.json',index)
        with self.assertRaises(ValueError):k.validate(bad)

    def test_04_actual_dispatch_only_remaining_and_failure_stop(self):
        c=copy.deepcopy(self.c);c.update(result_root=str(self.root/'dispatch'),global_lock=str(self.root/'lock'))
        cp=self.root/'dispatch-config.json';dump(cp,c);ap=self.root/'dispatch-auth.json';dump(ap,{'commit':'CPU-fixture'})
        class Child:
            pid=os.getpid();returncode=7
            def poll(self):return 7
        with patch.object(queue,'preflight',return_value=c),patch.object(queue,'gpu_startup',return_value={'CPU_fixture':True}),patch('m6_ch3_supplement_v10.gpu_identity.check_gpu_identity',return_value=None),patch.object(queue.subprocess,'Popen',return_value=Child()) as spawn,patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0'):
            with self.assertRaisesRegex(RuntimeError,'technical failure'):queue.supervise(cp,ap)
            self.assertEqual(spawn.call_count,1);self.assertEqual(spawn.call_args.args[0][-1],'PT-NP-ModernTCN')
        folder=Path(c['result_root'])/c['attempt'];self.assertTrue((folder/'failure.json').exists());self.assertFalse((folder/'complete.json').exists())
        self.assertEqual(k.sha(folder/'case-plan.json'),c['technical_continuation']['case_plan_ref']['sha256'])
        self.assertEqual(len(list((folder/'claims').glob('*.json'))),0)
        self.assertEqual(k.read(folder/'runtime.json')['original_wave_indices'],list(range(9,22)))

    def test_05_cache_only_O_merges_real_producer_locations(self):
        # Reuse earlier generated synthetic arrays; no real test/checkpoint data.
        fixture_root=R/'M6-ModernTCN-UV-v10-test-reporting-preparation-r1/tests'
        fixture=next(p for p in sorted(fixture_root.glob('test-reporting-fixture-*/SYNTHETIC_PLOTS_NOT_SCIENTIFIC')) if (p/'synthetic-source-ledger.json').exists())
        root=self.root/'SYNTHETIC_ONLY';old=root/cont.FIRST;new=root/cont.RETRY;old.mkdir(parents=True);new.mkdir()
        oldrt=dump(old/'runtime.json',dict(commit='synthetic-old',environment_ref={'path':'old-env','sha256':'old'},technical_continuation=None))
        rows=[];c=copy.deepcopy(self.c)
        for i,j in enumerate(c['jobs']):
            dest=old if i<8 else new;shutil.copytree(fixture/j['id'],dest/j['id'])
            receipt=k.read(fixture/'receipts'/(j['id']+'.json'))
            receipt['files']={str(p.relative_to(dest)):k.sha(p) for p in (dest/j['id']).iterdir() if p.is_file()}
            if i<8:
                receipt['runtime_sha256']=oldrt['sha256'];ref=dump(dest/'receipts'/(j['id']+'.json'),receipt);rows.append(dict(job=j,receipt_ref=ref))
            else:dump(dest/'receipts'/(j['id']+'.json'),receipt)
        c['technical_continuation']=dict(runtime_ref=oldrt,inherited_ref=dump(root/'eight.json',dict(rows=rows)))
        newrt=dump(new/'runtime.json',dict(commit='synthetic-new',environment_ref={'path':'new-env','sha256':'new'},technical_continuation=c['technical_continuation']))
        for j in c['jobs'][8:]:
            p=new/'receipts'/(j['id']+'.json');r=k.read(p);r['runtime_sha256']=newrt['sha256'];dump(p,r)
        c['sources_ref']=dict(path=str(fixture/'synthetic-source-ledger.json'),sha256=k.sha(fixture/'synthetic-source-ledger.json'))
        shutil.copy2(fixture/'case-plan.json',new/'case-plan.json')
        from m6_ch3_test_reporting_v10.plots import completed
        result=completed(c,new);self.assertEqual(result['inherited_model_evaluations'],8);self.assertEqual(result['new_test_accesses'],12)
        ledger=k.read(new/'plots/source-ledger.json')
        for i,j in enumerate(c['jobs']):
            source=ledger[j['id']];self.assertEqual(source['inherited_in_this_attempt'],i<8)
            self.assertEqual(source['evaluation_commit'],'synthetic-old' if i<8 else 'synthetic-new')
            self.assertEqual(Path(source['result_ref']['path']).parent.parent,old if i<8 else new)
        self.assertFalse((new/c['jobs'][0]['id']).exists());self.assertEqual(len(list((new/'plots').glob('case-*-test.svg'))),4)
        s=queue.status(new);self.assertEqual(s['inherited_model_evaluations'],8);self.assertEqual(s['new_test_receipts'],12);self.assertEqual(s['logical_model_evaluations'],20)
        # A changed old receipt is refused, never silently copied/rebound.
        (old/'receipts'/(c['jobs'][0]['id']+'.json')).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'bound file changed'):cont.location(c,new,c['jobs'][0])

    def test_06_only_import_order_changed_in_inference_math(self):
        old=ast.parse((R/'M6-ModernTCN-UV-v10-test-reporting-import-order-repair-r1/before/evaluate.py').read_text());new=ast.parse((W/'m6_ch3_test_reporting_v10/evaluate.py').read_text())
        before={f.name:f for f in old.body if isinstance(f,ast.FunctionDef)};after={f.name:f for f in new.body if isinstance(f,ast.FunctionDef)}
        before['infer'].body=[n for n in before['infer'].body if not (isinstance(n,ast.ImportFrom) and n.module=='utils.ch3_time_marks')]
        self.assertEqual(set(before),set(after))
        for name in before:self.assertEqual(ast.dump(before[name],include_attributes=False),ast.dump(after[name],include_attributes=False),name)
        imports=[n for n in new.body if isinstance(n,ast.ImportFrom) and n.module=='utils.ch3_time_marks'];self.assertEqual(len(imports),1)


if __name__=='__main__':unittest.main(verbosity=2)
