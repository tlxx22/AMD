"""C1 metadata/owned-chain fixtures; zero real models, optimizer, CUDA or datasets."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from moderntcn_c1 import contract as C, upstream as U, launch as L
from utils.ch3_contract import BestState

class C1Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.c=C.read(C.CONFIG)
    def test_01_exact_matrix_and_waves(self):
        C.validate(self.c,check_sources=False)
        self.assertEqual([len(w)for w in C.waves(self.c)],[4]*7+[1]+[4]*6)
        self.assertEqual({t['model']for t in self.c['tasks']},{'ModernTCN'})
        self.assertNotIn('ECL',self.c['datasets'])
    def test_02_missing_duplicate_rejected(self):
        for tasks in [self.c['tasks'][:-1],self.c['tasks']+[self.c['tasks'][0]]]:
            c=copy.deepcopy(self.c);c['tasks']=tasks
            with self.assertRaises(ValueError):C.validate(c,False)
    def test_03_parent_science_and_weather_fresh(self):
        for t in self.c['tasks']:
            p=C.profile(self.c,t);old=self.c['parents'][t['id']]['old_profile'];self.assertEqual(p['training'],old['training']);self.assertEqual({k:v for k,v in p.items()if k!='structure'},{k:v for k,v in old.items()if k!='structure'})
            self.assertEqual(p['structure']['ffn_ratio'],8)
            if t['dataset']=='Weather':self.assertEqual(p,old)
    def test_04_budget_and_steps(self):
        totals={k:0 for k in ['UrbanEV','EPF','M']}
        for t in self.c['tasks']:
            ar=C.step_arithmetic(self.c,t);p=C.profile(self.c,t);self.assertEqual(ar['train_batches'],p['training']['scheduler']['steps_per_epoch']);totals['UrbanEV'if t['dataset']=='UrbanEV'else 'M'if t['task']=='M'else 'EPF']+=ar['max_optimizer_steps']
        self.assertEqual(totals,dict(UrbanEV=2085780,EPF=28500,M=50640))
    def test_05_structure_or_protocol_drift_rejected(self):
        for section,k,v in [('structure','ffn_ratio',1),('training','batch',64)]:
            c=copy.deepcopy(self.c);c['resolved_profiles'][c['tasks'][0]['id']][section][k]=v
            with self.assertRaises(ValueError):C.validate(c,False)
        c=copy.deepcopy(self.c);c['comparison_policy']['goal2']['minimum_strict_wins']=6
        with self.assertRaises(ValueError):C.validate(c,False)
    def test_06_MS_M_native_output_routing(self):
        from models.ch3_adapter import target_prediction
        # Array-returning fixture, not nn.Module; exercises existing slice/shape contract only.
        for domain in ['UrbanEV','PJM','Weather']:
            t=next(t for t in self.c['tasks']if t['dataset']==domain);p=C.profile(self.c,t);a=np.arange(2*p['pred_len']*p['C'],dtype=np.float32).reshape(2,p['pred_len'],p['C']);x=np.zeros((2,p['T'],p['C']),np.float32)
            actual,_=target_prediction(lambda x:a,x,p)
            expected=a if t['task']=='M'else a[:,:,p['target_idx']:p['target_idx']+1]
            np.testing.assert_array_equal(actual,expected)
    def test_07_strict_best_tie_and_patience(self):
        b=BestState(2);self.assertTrue(b.update(1.,1));self.assertFalse(b.update(1.,2));self.assertEqual(b.epoch,1);b.update(2.,3);self.assertTrue(b.stopped)
    def test_08_test_once_claim_and_revision(self):
        from moderntcn_c1.worker import claim_test,identity
        c=self.c;t=c['tasks'][0];ident=identity(c,t,{},dict(commit='test',authorization_ref={},runtime_ref={}))
        self.assertEqual(ident['revision'],C.ID);self.assertNotEqual(ident['revision'],'cfg_adapted_v1')
        with tempfile.TemporaryDirectory()as d:
            claim_test(d,ident,1)
            with self.assertRaises(FileExistsError):claim_test(d,ident,1)
    def test_09_resume_refused_before_access(self):
        from moderntcn_c1.worker import guard
        with self.assertRaises(PermissionError):guard({}, {}, '/not-accessed', {},True)
    def test_10_mutex_nonblocking_existing_inode(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'mutex';p.write_text('preserve');ino=p.stat().st_ino
            f=L.acquire_mutex(p)
            try:self.assertIsNone(L.acquire_mutex(p));self.assertEqual(p.read_text(),'preserve');self.assertEqual(p.stat().st_ino,ino)
            finally:f.close()
            g=L.acquire_mutex(p);self.assertIsNotNone(g);g.close()
    def fixture(self,root):
        a=root/'attempt';a.mkdir();(a/'staging').mkdir();(a/'runs').mkdir();(a/'receipts').mkdir();ids=[f't{i}'for i in range(27)];waves=[ids[i:i+3]for i in range(0,27,3)]
        cfg=dict(tasks=[dict(id=i)for i in ids]);cp=root/'config.json';C.write_new(cp,cfg)
        auth=root/'AUTH.json';C.write_new(auth,dict(task_ids=ids));rt=a/'runtime.json';runtime=dict(task_ids=ids,waves=waves,authorization_ref=C.ref(auth),config_ref=C.ref(cp),owner_pid=1234,owner_start='99',source_binding={},commit='fixture')
        C.write_new(rt,runtime);binding=dict(folder=str(a),auth=C.ref(auth),config=C.ref(cp),runtime=C.ref(rt));C.write_new(a/'owner.json',dict(pid=1234,start='99',runtime_ref=binding['runtime']));C.write_new(a/'children.json',[]);C.write_new(a/'progress.json',dict(state='RUNNING',wave=8))
        return binding,a,ids
    def test_11_wait_live_no_completion(self):
        with tempfile.TemporaryDirectory()as d:
            b,a,ids=self.fixture(Path(d))
            with patch.object(U,'live',return_value=True):self.assertFalse(U.inspect(b)['ready'])
    def test_12_exit_without_complete_blocks(self):
        with tempfile.TemporaryDirectory()as d:
            b,a,ids=self.fixture(Path(d))
            with patch.object(U,'live',return_value=False),self.assertRaises(U.Blocked):U.inspect(b)
    def test_13_stop_and_failure_block(self):
        for name in ['STOP','failure.json']:
            with tempfile.TemporaryDirectory()as d:
                b,a,ids=self.fixture(Path(d));(a/name).write_text('{}')
                with self.assertRaises(U.Blocked):U.inspect(b)
    def test_14_complete_waits_exit_and_rejects_staging(self):
        with tempfile.TemporaryDirectory()as d:
            b,a,ids=self.fixture(Path(d));C.write_new(a/'complete.json',dict(technical_complete=True,task_ids=ids,waves=9))
            with patch.object(U,'live',return_value=True):self.assertFalse(U.inspect(b)['ready'])
            with patch.object(U,'live',return_value=False):self.assertTrue(U.inspect(b)['ready'])
            (a/'staging'/'unfinished').mkdir()
            with patch.object(U,'live',return_value=False),self.assertRaises(U.Blocked):U.inspect(b)
    def test_15_identity_drift_and_missing_completion_reject(self):
        with tempfile.TemporaryDirectory()as d:
            b,a,ids=self.fixture(Path(d));C.write_new(a/'complete.json',dict(technical_complete=True,task_ids=ids[:-1],waves=9))
            with patch.object(U,'live',return_value=False),self.assertRaises(U.Blocked):U.inspect(b)
            Path(b['auth']['path']).write_text('{}')
            with self.assertRaises(U.Blocked):U.inspect(b)
    def test_16_science_gate_does_not_block_completion(self):
        with tempfile.TemporaryDirectory()as d:
            b,a,ids=self.fixture(Path(d));C.write_new(a/'complete.json',dict(technical_complete=True,task_ids=ids,waves=9,scientific_gate='Not met'))
            with patch.object(U,'live',return_value=False):self.assertTrue(U.inspect(b)['ready'])
    def test_17_gpu_query_timeout_only_retry(self):
        import subprocess
        auth=dict(gpu_uuid='test')
        with patch.object(L.subprocess,'check_output',side_effect=[subprocess.TimeoutExpired('query',60),'0, test, 81920, 80000']),patch.object(L.time,'sleep'):
            self.assertEqual(L.gpu_startup(auth)['uuid'],'test')
        with patch.object(L.subprocess,'check_output',side_effect=subprocess.CalledProcessError(1,'query'))as m:
            with self.assertRaises(subprocess.CalledProcessError):L.gpu_startup(auth)
            self.assertEqual(m.call_count,1)
    def test_18_stop_only_owned_chain(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);cp=root/'chain';cp.mkdir();ap=root/'auth';ap.write_text('{}');C.write_new(cp/'owner.json',dict(pid=os.getpid(),start=U.proc_identity(os.getpid())['start'],worktree=str(C.ROOT),auth_ref=C.ref(ap)))
            with patch.object(L,'CONTROL',cp),patch.object(L,'AUTH_PATH',ap),self.assertRaises(PermissionError):L.stop()
            self.assertFalse((cp/'STOP').exists())
    def test_19_chain_upstream_failure_never_takes_lock(self):
        with tempfile.TemporaryDirectory()as d:
            cp=Path(d)/'chain';ap=Path(d)/'auth';ap.write_text('{}')
            with patch.object(L,'CONTROL',cp),patch.object(L,'AUTH_PATH',ap),patch.object(L,'preflight'),patch.object(L,'inspect',side_effect=U.Blocked('upstream failure')),patch.object(L,'acquire_mutex')as lock,patch.object(L,'gpu_startup')as gpu:
                with self.assertRaises(U.Blocked):L.chain()
                lock.assert_not_called();gpu.assert_not_called()
                self.assertEqual(C.read(cp/'status.json')['state'],'BLOCKED')
    def test_20_existing_legacy_profile_unchanged(self):
        from utils.ch3_contract import profile
        for rr in self.c['parent_protocols']:
            pc=C.bound(rr);t=next(t for t in pc['tasks']if t['model']=='ModernTCN')
            self.assertEqual(profile(pc,t),pc['resolved_profiles'][t['id']])

class C1SealedFixtures(unittest.TestCase):
    fixture=C1Contract.fixture
    @classmethod
    def setUpClass(cls):cls.c=C.read(C.CONFIG)
    def test_21_full_upstream_seal_then_corruption(self):
        with tempfile.TemporaryDirectory()as d:
            b,a,ids=self.fixture(Path(d));C.write_new(a/'complete.json',dict(technical_complete=True,task_ids=ids,waves=9));cfg=C.bound(b['config']);rt=C.bound(b['runtime'])
            for t in cfg['tasks']:
                p=a/'runs'/t['id'];p.mkdir();ident=dict(task=t,config_sha=C.digest(cfg),source=rt['source_binding'],producer_commit=rt['commit']);m=dict(task_id=t['id'],identity=ident,identity_sha=C.digest(ident),launch_provenance=dict(runtime_ref=b['runtime'],authorization_ref=b['auth']),status='completed');met=dict(mse=1.,mae=.5,test_calls=1)
                C.write_new(p/'manifest.json',m);C.write_new(p/'metrics.json',met)
                for n in ['history.jsonl','updates.jsonl','best.pt','last.pt','test-claim.json','test-result.json','config.json','identity.json']:(p/n).write_bytes(b'fixture only; no tensor serialization')
                (p/'checksums.sha256').write_text(''.join(C.sha(f)+'  '+f.name+'\n'for f in sorted(p.iterdir())))
                C.write_new(p/'seal.json',dict(complete=True,identity_sha=m['identity_sha'],checksums_sha256=C.sha(p/'checksums.sha256')))
                C.write_new(a/'receipts'/(t['id']+'.json'),dict(path=str(p),manifest=m,metrics=met,seal_sha256=C.sha(p/'seal.json')))
            with patch.object(U,'live',return_value=False):self.assertTrue(U.verify_completion(b)['technical_complete'])
            (a/'runs'/ids[0]/'best.pt').write_bytes(b'corrupted fixture')
            with patch.object(U,'live',return_value=False),self.assertRaises(U.Blocked):U.verify_completion(b)
    def test_22_fixed_53_dispatch_no_effect_gate(self):
        from moderntcn_c1 import summary  # bind real audit before dispatch-only mock scope
        class Child:
            returncode=0
            def __init__(self,pid):self.pid=pid
            def poll(self):return 0
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);control=root/'control';res=root/'result';authfile=root/'AUTH';C.write_new(authfile,dict(commit='fixture',task_ids=[t['id']for t in self.c['tasks']]))
            assigned=[];pid=[10000000]
            def spawn(cmd,**kwargs):
                tid=cmd[cmd.index('--task-id')+1];assigned.append(tid);pid[0]+=1
                p=res/C.ATTEMPT/'runs'/tid;p.mkdir();C.write_new(p/'seal.json',dict(fixture=True))
                C.write_new(res/C.ATTEMPT/'receipts'/(tid+'.json'),dict(task_id=tid,seal_ref=C.ref(p/'seal.json')))
                return Child(pid[0])
            lock=object()
            class Lock:
                def close(self):pass
            with patch.object(L,'CONTROL',control),patch.object(L,'RESULT_ROOT',res),patch.object(L,'AUTH_PATH',authfile),patch.object(L,'preflight'),patch.object(L,'inspect',return_value=dict(ready=True,state='VERIFY_UPSTREAM_TECHNICAL_COMPLETION')),patch.object(L,'verify_completion',return_value=dict(technical_complete=True)),patch.object(L,'acquire_mutex',return_value=Lock()),patch.object(L,'gpu_startup',return_value=dict(uuid='fixture')),patch.object(L,'proc_identity',return_value=dict(start='fixture')),patch.object(L.subprocess,'Popen',side_effect=spawn),patch('moderntcn_c1.artifacts.audit',side_effect=lambda p,c,t,commit:dict(task_id=t['id'])),patch('moderntcn_c1.summary.summarize',return_value=dict(fixture=True)):
                L.chain()
            self.assertEqual(assigned,[t['id']for t in self.c['tasks']]);self.assertEqual(C.read(res/C.ATTEMPT/'complete.json')['waves'],14);self.assertEqual(C.read(control/'status.json')['state'],'COMPLETE')
    def test_23_native_numerical_loop_preserved(self):
        import ast,subprocess
        original=subprocess.check_output(['git','show','900fb19c792251ca549ef8c4cebcc3b725d5fe22:ch3_runner.py'],cwd=C.ROOT,text=True)
        old=ast.parse(original);new=ast.parse((C.ROOT/'ch3_runner.py').read_text())
        def funcs(tree):return {x.name:x for x in tree.body if isinstance(x,ast.FunctionDef)}
        for n in ['update','evaluate','init_training','save_state','restore_state']:
            self.assertEqual(ast.dump(funcs(old)[n],include_attributes=False),ast.dump(funcs(new)[n],include_attributes=False))
        f=copy.deepcopy(funcs(new)['formal_worker']);f.body=[x for x in f.body if not (isinstance(x,ast.If)and "c.get('c1_baseline')"==ast.unparse(x.test))]
        self.assertEqual(ast.dump(funcs(old)['formal_worker'],include_attributes=False),ast.dump(f,include_attributes=False))
    def test_24_no_torch_loaded_by_metadata(self):
        import sys
        self.assertNotIn('torch',sys.modules)

    def test_25_saved_53_revision_and_goal2_summary(self):
        from moderntcn_c1 import summary as S
        from moderntcn_c1.worker import identity,claim_test
        with tempfile.TemporaryDirectory()as d:
            folder=Path(d)/'C1';folder.mkdir();(folder/'runs').mkdir();(folder/'receipts').mkdir();up=Path(d)/'UV';(up/'receipts').mkdir(parents=True)
            for t in self.c['tasks']:
                p=folder/'runs'/t['id'];p.mkdir();pr=C.profile(self.c,t);metadata=C.bound(self.c['parents'][t['id']]['manifest'])['metadata'];ident=identity(self.c,t,metadata,dict(commit='fixture',authorization_ref={},runtime_ref={}))
                C.write_new(p/'manifest.json',dict(task=t,profile=pr,identity=ident,metadata=metadata))
                n=pr['training']['epochs'];steps=pr['training']['scheduler']['steps_per_epoch'];hist=[dict(epoch=i,steps=i*steps,best_epoch=i,validation=dict(mse=1/i))for i in range(1,n+1)]
                (p/'history.jsonl').write_text(''.join(json.dumps(x)+'\n'for x in hist));(p/'best.pt').write_bytes(b'metadata fixture, not a checkpoint')
                claim_test(p,ident,n)
                r=dict(id=t['id'],commit='fixture',protocol_sha=C.digest(self.c),profile_sha=C.digest(pr),data_sha=C.digest(metadata),best_epoch=n,metric_space='train-standardized',metric_scope=t['metric_scope'],mse=1.,mae=1.,sse=2.,sae=2.,elements=2,final_test=dict(calls=1,selected='best.pt',sha256=C.sha(p/'best.pt'),epoch=n))
                C.write_new(p/'result.json',r);C.write_new(folder/'receipts'/(t['id']+'.json'),{})
                (p/'checksums.sha256').write_text(''.join(C.sha(f)+'  '+f.name+'\n'for f in sorted(p.iterdir())))
                C.write_new(p/'seal.json',dict(schema='moderntcn-C1-seal-v1',task_id=t['id'],checksums_sha256=C.sha(p/'checksums.sha256')))
            for i,name in enumerate(C.EPF):
                p=up/name;p.mkdir();metrics=dict(mse=.5,mae=.5 if i<2 else 1.);C.write_new(p/'metrics.json',metrics);C.write_new(up/'receipts'/(name+'.json'),dict(path=str(p),manifest=dict(identity=dict(task=dict(dataset=name,arm='UV'))),metrics=metrics))
            with patch.object(S,'UPSTREAM',dict(folder=str(up))):s=S.summarize(self.c,dict(commit='fixture'),folder)
            self.assertEqual(s['unchanged_cells'],318);self.assertEqual(s['replaced_cells'],53);self.assertEqual(s['goal2']['wins'],7);self.assertEqual(s['goal2']['denominator'],10);self.assertEqual(s['goal2']['status'],'Met')
            # Ties were not counted; no separate 70% requirement for MAE (2/5 here).
            self.assertEqual(s['goal2']['mae_wins'],2)
            t=self.c['tasks'][0];p=folder/'runs'/t['id'];(p/'seal.json').unlink()
            with self.assertRaises(ValueError):S.audit(p,self.c,t,'fixture')

if __name__=='__main__':unittest.main(verbosity=2)
