"""Bounded actual v6 worker/namespace/recovery acceptance on generated CSVs."""
import copy
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from tests import timemixer_v6_fixtures as f
from timemixer_dev_v6 import adapter, worker, launcher
from timemixer_dev_v6.worker import run_worker
from timemixer_dev_v6.training_state import read, validate as validate_state, restore, save, file_sha
from timemixer_dev_v6.artifacts import audit
from timemixer_dev_v6.summary import summarize
from timemixer_dev_v6.training_config import validate, digest
from utils.ch3_onecycle import Schedule
from utils.ch3_contract import BestState
from utils.ch3_data import load, batches
from utils.ch3_time_marks import batch_parts


def tree(folder):
    return {str(p.relative_to(folder)):file_sha(p) for p in Path(folder).rglob('*') if p.is_file()}


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='.uv-v6-runner-',dir=f.ROOT);self.folder=Path(self.tmp.name)
        self.results={}
    def tearDown(self):
        if self._testMethodName in ('test_unknown_test_claim_never_retests_or_changes_checkpoint','test_STOP_and_worker_failure_preserve_staging_no_overwrite','test_supervise_refusals_are_side_effect_free_and_owned_failure_stops_queue'):
            dest=f.EVIDENCE/'synthetic-negative-fixtures'/(self._testMethodName+'-attempt'+os.environ.get('UV_V6_ACCEPTANCE_ATTEMPT','1'))
            dest.parent.mkdir(exist_ok=True);shutil.copytree(self.folder,dest)
        self.tmp.cleanup()
    def make(self,**kwargs):return f.fixture(self.folder,**kwargs)
    def run_case(self,c,id,path,attempt='trial',**kwargs):
        return run_worker(c,id,self.folder/'results',attempt,path_override=path,**kwargs)

    def test_four_arms_actual_routes_and_original_off_updates_evaluation_exact(self):
        for arm in ('B','U','V','UV'):
            c,id,path=self.make(arm=arm)
            if arm=='B':r=self.run_case(c,id,path,attempt='arms')
            else:
                with patch('models.ch3_adapter.target_prediction',side_effect=AssertionError('old evaluator route')):
                    r=self.run_case(c,id,path,attempt='arms')
            self.results[arm]=r;m=r['manifest'];self.assertEqual(m['steps'],4)
            self.assertTrue(all(m['routes'][phase]>0 for phase in ('train','validation','test')))
            self.assertEqual(m['identity']['candidate_id'],'timemixer-uv-dev-v6');self.assertEqual(m['test_calls'],1)
            self.assertEqual(r['metrics']['elements'],36);self.assertTrue((Path(r['path'])/'checksums.sha256').is_file())
        c,id,path=self.make(arm='B');reference=self.run_case(c,id,path,attempt='reference',reference_original=True)
        a=read(Path(self.results['B']['path'])/'last.pt');b=read(Path(reference['path'])/'last.pt')
        for key in ('model','optimizer','scheduler','rng','best','epoch','steps'):f.nested_equal(self,a[key],b[key])
        for metric in ('mse','mae','sse','sae','elements'):self.assertEqual(self.results['B']['metrics'][metric],reference['metrics'][metric])
        uv=self.results['UV'];dest=f.EVIDENCE/('synthetic-artifacts/four-arm-UV-attempt'+os.environ.get('UV_V6_ACCEPTANCE_ATTEMPT','1'));dest.parent.mkdir(exist_ok=True)
        shutil.copytree(uv['path'],dest);self.assertEqual(audit(dest)['metrics']['test_calls'],1)

    def test_original_loader_F1_F2_F3_F4_H12_and_EPF_M_interfaces(self):
        for variant,C in [('F1',6),('F2',8),('F3',9),('F4',11)]:
            c,id,path=self.make(variant=variant,H=12);r=self.run_case(c,id,path,attempt=variant)
            self.assertEqual(r['manifest']['identity']['profile']['C'],C);self.assertEqual(r['metrics']['elements'],18)
            self.assertEqual(r['manifest']['identity']['structural_spec']['pred_len'],1)
            self.assertEqual(r['manifest']['identity']['structural_spec']['label_horizon'],12)
        for dataset,variant,H,elements in [('PJM','MS',24,216),('ETTh1','M',96,6048)]:
            c,id,path=self.make(dataset=dataset,variant=variant,H=H);r=self.run_case(c,id,path,attempt=dataset)
            self.assertEqual(r['metrics']['elements'],elements);self.assertEqual(r['manifest']['identity']['profile']['T'],96)
            self.assertEqual(r['metrics']['metric_scope'],'all_channels' if variant=='M' else 'target_only')
            self.assertEqual(r['manifest']['identity']['structural_spec']['insertion_point'],dict(u='finest_embedding_before_first_PDM',v='finest_after_last_PDM_before_future_multi_mixing'))

    def test_epoch_resume_exact_all_state_batch_order_and_test_once(self):
        c,id,path=self.make();full=run_worker(c,id,self.folder/'full','one',path_override=path)
        paused=self.run_case(c,id,path,attempt='resume',pause_after_epoch=1);self.assertEqual(paused['steps'],2)
        resumed=self.run_case(c,id,path,attempt='resume',resume=True)
        f.nested_equal(self,read(Path(full['path'])/'last.pt'),read(Path(resumed['path'])/'last.pt'))
        self.assertEqual((Path(full['path'])/'updates.jsonl').read_text(),(Path(resumed['path'])/'updates.jsonl').read_text())
        self.assertEqual(full['metrics'],resumed['metrics']);self.assertEqual(resumed['metrics']['test_calls'],1)

    def test_unknown_test_claim_never_retests_or_changes_checkpoint(self):
        c,id,path=self.make()
        with self.assertRaisesRegex(RuntimeError,'test uncertainty'):self.run_case(c,id,path,fault='after_test_claim')
        staging=next((self.folder/'results/trial/staging').iterdir());before=tree(staging)
        with patch.object(worker,'evaluate',side_effect=AssertionError('must not reevaluate')):
            with self.assertRaisesRegex(ValueError,'cannot repeat'):self.run_case(c,id,path,resume=True)
        self.assertEqual(tree(staging),before)

    def test_saved_test_reused_after_publish_failure_success_immutable(self):
        c,id,path=self.make()
        with self.assertRaisesRegex(RuntimeError,'sealing failure'):self.run_case(c,id,path,fault='before_publish')
        with patch.object(worker,'evaluate',side_effect=AssertionError('saved test must be reused')):
            r=self.run_case(c,id,path,resume=True)
        self.assertEqual(r['metrics']['test_calls'],1);self.assertEqual(r['manifest']['routes']['test'],4)
        before=tree(Path(r['path']))
        with self.assertRaises(FileExistsError):self.run_case(c,id,path,attempt='duplicate')
        self.assertEqual(tree(Path(r['path'])),before)
        report=summarize(c,self.folder/'results',None)
        self.assertEqual(report['completed'],1);self.assertEqual(report['scientific_gate'],'Not evaluated')
        changed=copy.deepcopy(r['manifest']['identity']);changed['gate_policy']={'report_only':'changed'};changed['config_sha']='report-wrapper'
        self.assertEqual(worker.successful_identity(changed),worker.successful_identity(r['manifest']['identity']))

    def test_STOP_and_worker_failure_preserve_staging_no_overwrite(self):
        c,id,path=self.make();(self.folder/'results/stopped').mkdir(parents=True);(self.folder/'results/stopped/STOP').touch()
        with self.assertRaises(InterruptedError):self.run_case(c,id,path,attempt='stopped')
        with self.assertRaisesRegex(RuntimeError,'worker failure'):self.run_case(c,id,path,attempt='failed',fault='after_update')
        staging=next((self.folder/'results/failed/staging').iterdir());self.assertEqual(len((staging/'updates.jsonl').read_text().splitlines()),1)
        with self.assertRaisesRegex(ValueError,'before first epoch'):self.run_case(c,id,path,attempt='failed',resume=True)
        self.assertFalse((self.folder/'results/failed/complete.json').exists())

    def test_strict_best_tie_and_patience_preserved_without_extra_training(self):
        best=BestState(2);self.assertTrue(best.update(1.,1));self.assertFalse(best.update(1.,2));self.assertFalse(best.stopped)
        self.assertFalse(best.update(2.,3));self.assertTrue(best.stopped);self.assertEqual(best.epoch,1)
        c,id,path=self.make(epochs=3,patience=1);evaluate=worker.evaluate
        def tied(*args,**kwargs):
            value=evaluate(*args,**kwargs)
            if args[-1]=='validation':value.update(mse=1.,sse=float(value['elements']),synthetic_controlled_validation=True)
            return value
        with patch.object(worker,'evaluate',side_effect=tied):r=self.run_case(c,id,path)
        self.assertEqual((r['manifest']['epochs'],r['manifest']['steps'],r['metrics']['best_epoch']),(2,4,1))

    def test_two_waves_forked_v6_CLI_children_and_completion(self):
        c,id,path=self.make(arm='U');v,vid,_=self.make(arm='V');c['tasks'].append(v['tasks'][0]);validate(c)
        cp=self.folder/'two-wave.json';cp.write_text(json.dumps(c));ids=self.folder/'ids.json';ids.write_text(json.dumps([id,vid]))
        root=self.folder/'queue';attempt=os.environ.get('UV_V6_ACCEPTANCE_ATTEMPT','1');log=f.EVIDENCE/('two-wave-fork-attempt'+attempt+'.log')
        command=[sys.executable,'-B',str(f.ROOT/'timemixer_train_v6_entry.py'),'supervise','--config',str(cp),'--task-list',str(ids),'--root',str(root),'--attempt','two','--synthetic-fixture-path',str(path)]
        try:
            result=f.run_cpu_supervisor(command,cwd=f.ROOT,timeout=45);code=result.returncode
            log.write_text(result.stdout+'\nSTDERR\n'+result.stderr)
        except BaseException:
            # Stop only this fixture's recorded children before snapshot/cleanup.
            import signal,time
            control=root/'two/children.json'
            if control.exists():
                owned=json.loads(control.read_text());(root/'two/STOP').touch(exist_ok=True)
                def active(child):
                    try:
                        stat=Path('/proc/'+str(child['pid'])+'/stat').read_text()
                        return launcher.proc_start(child['pid'])==child['start'] and stat[stat.rfind(')')+2:].split()[0]!='Z'
                    except (FileNotFoundError,ProcessLookupError):return False
                for child in owned:
                    if child['expected_task_id'] in (id,vid) and active(child) and os.getpgid(child['pid'])==child['pid']:os.killpg(child['pid'],signal.SIGTERM)
                end=time.monotonic()+15
                while any(active(child) for child in owned) and time.monotonic()<end:time.sleep(.1)
                for child in owned:
                    if child['expected_task_id'] in (id,vid) and active(child) and os.getpgid(child['pid'])==child['pid']:os.killpg(child['pid'],signal.SIGKILL)
            shutil.copytree(root,f.EVIDENCE/('failed-two-wave-fork-attempt'+attempt))
            consumed=sum(len(p.read_text().splitlines()) for p in root.glob('two/*/*/updates.jsonl'))
            f.CHILD_UPDATES.append(dict(test=self.id(),updates=consumed,failed=True))
            raise
        folder=root/'two'
        journals=list((folder/'runs').glob('*/updates.jsonl'))+list((folder/'staging').glob('*/updates.jsonl'))
        consumed=sum(len(p.read_text().splitlines()) for p in journals);f.CHILD_UPDATES.append(dict(test=self.id(),updates=consumed))
        self.assertEqual(code,0,log.read_text()[-3000:]);self.assertEqual(consumed,8)
        complete=json.loads((folder/'complete.json').read_text());self.assertTrue(complete['technical_complete']);self.assertEqual(complete['task_ids'],[id,vid]);self.assertEqual(complete['waves'],2)
        runtime=json.loads(next(folder.glob('runtime-*.json')).read_text());self.assertEqual(runtime['candidate_id'],'timemixer-uv-dev-v6')
        for p in (folder/'runs').iterdir():
            receipt=audit(p);command=json.loads((p/'command.json').read_text())
            self.assertIn(str(f.ROOT/'timemixer_train_v6_entry.py'),command['argv'])
            self.assertEqual(receipt['manifest']['identity']['candidate_id'],'timemixer-uv-dev-v6')
        dest=f.EVIDENCE/('synthetic-artifacts/two-wave-attempt'+attempt);shutil.copytree(folder,dest)

    def test_cold_v6_waves_CLI_metadata_only(self):
        cp=f.ROOT/'configs/timemixer_uv_round2_resolved_v6.json';ids=self.folder/'first24.json'
        ids.write_text(json.dumps([t['id'] for t in f.STORED['tasks'] if t['goal']==1]))
        result=subprocess.run([sys.executable,'-B',str(f.ROOT/'timemixer_train_v6_entry.py'),'waves','--config',str(cp),'--task-list',str(ids)],cwd=f.ROOT,capture_output=True,text=True,timeout=45)
        self.assertEqual(result.returncode,0,result.stderr);value=json.loads(result.stdout)
        self.assertEqual(len(value['waves']),6);self.assertFalse(value['training_started']);self.assertFalse(value['GPU_query'])

    def test_supervise_refusals_are_side_effect_free_and_owned_failure_stops_queue(self):
        c,id,path=self.make();cp=self.folder/'config.json';cp.write_text(json.dumps(c));root=self.folder/'queue';folder=root/'existing';folder.mkdir(parents=True)
        owner=dict(pid=os.getpid(),start=launcher.proc_start(os.getpid()),synthetic=True)
        (folder/'owner.json').write_text(json.dumps(owner));(folder/'STOP').write_text('existing STOP');before=tree(folder)
        with patch.object(launcher.subprocess,'Popen',side_effect=AssertionError('no child before ownership')):
            with self.assertRaises(PermissionError):launcher.supervise(cp,[id],root,'existing',synthetic_fixture_path=path,resume=True)
        self.assertEqual(tree(folder),before);self.assertEqual(launcher.proc_start(os.getpid()),owner['start'])
        (folder/'complete.json').write_text('{}');before=tree(folder)
        with self.assertRaises(PermissionError):launcher.supervise(cp,[id],root,'existing',synthetic_fixture_path=path,resume=True)
        self.assertEqual(tree(folder),before)
        # Resource-lock refusal through supervise, with no real GPU call/grant.
        real=copy.deepcopy(f.STORED);real['global_lock']=str(self.folder/'lock');rp=self.folder/'real.json';rp.write_text(json.dumps(real));ap=self.folder/'not-a-grant.json';ap.write_text('{}')
        with patch.object(worker,'RUN_NAMESPACE',root),patch.object(launcher,'validate_authorization'),patch.object(launcher.fcntl,'flock',side_effect=BlockingIOError('busy')),patch.object(launcher,'gpu_startup',side_effect=AssertionError('must not query')):
            with self.assertRaises(BlockingIOError):launcher.supervise(rp,[real['tasks'][0]['id']],root,'existing',ap,resume=True)
        self.assertEqual(tree(folder),before)
        class FailedChild:
            pid=os.getpid()
            returncode=1
            def poll(self):return 1
            def wait(self,timeout=None):return 1
        second=copy.deepcopy(c['tasks'][0]);second.update(id='synthetic-second-wave',arm='U',use_v=False)
        c['tasks'].append(second);cp.write_text(json.dumps(c))
        real_popen=launcher.subprocess.Popen
        def failed_child(argv,**kwargs):
            return FailedChild() if '--receipt' in argv else real_popen(argv,**kwargs)
        with patch.object(launcher.subprocess,'Popen',side_effect=failed_child) as child:
            with self.assertRaises(RuntimeError):launcher.supervise(cp,[id,second['id']],root,'owned-failure',synthetic_fixture_path=path)
        dispatched=[c for c in child.call_args_list if '--receipt' in c.args[0]]
        self.assertEqual(len(dispatched),1);self.assertTrue((root/'owned-failure/STOP').exists())
        self.assertFalse((root/'owned-failure/complete.json').exists());self.assertFalse(json.loads((root/'owned-failure/failure.json').read_text())['next_wave_dispatched'])
        class SuccessfulChild(FailedChild):
            returncode=0
            def poll(self):return 0
            def wait(self,timeout=None):return 0
        def successful_child(argv,**kwargs):
            return SuccessfulChild() if '--receipt' in argv else real_popen(argv,**kwargs)
        with patch.object(launcher.subprocess,'Popen',side_effect=successful_child),patch.object(launcher,'verify_child_receipt',return_value=id):
            with self.assertRaisesRegex(ValueError,'wave completion'):launcher.supervise(cp,[id,second['id']],root,'duplicate-receipt',synthetic_fixture_path=path)
        self.assertFalse((root/'duplicate-receipt/complete.json').exists())


class Recovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='.uv-v6-runner-',dir=f.ROOT);cls.folder=Path(cls.tmp.name)
        cls.config,cls.task_id,cls.path=f.fixture(cls.folder)
        cls.paused=run_worker(cls.config,cls.task_id,cls.folder/'results','checkpoint',path_override=cls.path,pause_after_epoch=1)
        cls.checkpoint=Path(cls.paused['staging'])/'last.pt';cls.state=read(cls.checkpoint)
        dest=f.EVIDENCE/('synthetic-artifacts/recovery-fixture-attempt'+os.environ.get('UV_V6_ACCEPTANCE_ATTEMPT','1'));dest.mkdir(parents=True)
        shutil.copy2(cls.checkpoint,dest/'last.pt');shutil.copy2(str(cls.checkpoint)+'.sha256',dest/'last.pt.sha256')
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def live(self):
        p=self.config['tasks'][0]['structural_source_profile'];m=adapter.build(p,use_u=True,use_v=True,task='MS',fold=1);tr=p['training']
        opt=torch.optim.Adam(m.parameters(),lr=tr['lr'],betas=tuple(tr['betas']),eps=tr['eps'],weight_decay=tr['weight_decay'])
        return m,opt,Schedule(opt,tr['scheduler']),torch.Generator().manual_seed(29)

    def test_cross_version_arm_trend_mode_and_corrupt_Adam_rejected_before_live_writes(self):
        m,opt,s,g=self.live();identity=self.state['identity'];before=f.live_state(m,opt,s,g)
        for kind in ('candidate','v2_candidate','v3_candidate','v4_candidate','provided_candidate','old_same_candidate_revision','provided_old_revision','arm','F','H','protocol','trend_mode','provided_trend_mode','source','data','all_Adam','one_Adam','moment','step'):
            bad=copy.deepcopy(self.state);expected=copy.deepcopy(identity)
            if kind=='candidate':bad['identity']['candidate_id']='timemixer-uv-dev-v1'
            elif kind=='v2_candidate':bad['identity']['candidate_id']='timemixer-uv-dev-v2'
            elif kind=='v3_candidate':bad['identity']['candidate_id']='timemixer-uv-dev-v3'
            elif kind=='v4_candidate':bad['identity']['candidate_id']='timemixer-uv-dev-v4'
            elif kind=='provided_candidate':expected['candidate_id']='timemixer-uv-dev-v1'
            elif kind=='old_same_candidate_revision':bad['identity']['structural_spec']['structural_revision']=dict(u='crossformer_full_TSA_receiver_conditioned_router_v6',v='moderntcn_Block_serial_nonlinear_local_dilated_v6',fusion='two_internal_finest_representation_residuals')
            elif kind=='provided_old_revision':expected['structural_spec']['structural_revision']=dict(u='crossformer_full_TSA_receiver_conditioned_router_v6',v='moderntcn_Block_serial_nonlinear_local_dilated_v6',fusion='two_internal_finest_representation_residuals')
            elif kind=='arm':bad['identity']['task']['arm']='U'
            elif kind=='F':bad['identity']['task']['input_variant']='F1'
            elif kind=='H':bad['identity']['task']['h']=12
            elif kind=='protocol':bad['identity']['training_protocol']='third-round'
            elif kind=='trend_mode':bad['identity']['structural_spec']['module_constants']['patch_len']=4
            elif kind=='provided_trend_mode':expected['structural_spec']['module_constants']['temporal_kernels']=[5,3]
            elif kind=='source':bad['identity']['source']['wrong']='sha'
            elif kind=='data':bad['identity']['data_sha']='wrong'
            elif kind=='all_Adam':bad['optimizer']['state']={}
            elif kind=='one_Adam':bad['optimizer']['state'].pop(next(iter(bad['optimizer']['state'])))
            elif kind=='moment':bad['optimizer']['state'][next(iter(bad['optimizer']['state']))].pop('exp_avg')
            else:bad['optimizer']['state'][next(iter(bad['optimizer']['state']))]['step'].fill_(-1)
            with self.subTest(kind=kind),self.assertRaises(ValueError):validate_state(bad,m,opt,s,expected)
            f.nested_equal(self,before,f.live_state(m,opt,s,g))
        bad=copy.deepcopy(identity);bad['candidate_id']='timemixer-uv-dev-v1'
        with self.assertRaises(ValueError):restore(self.checkpoint,m,opt,s,g,bad)
        f.nested_equal(self,before,f.live_state(m,opt,s,g))

    def test_artifact_checksum_and_receipt_identity_negatives(self):
        # Finish the shared two-step checkpoint once (two additional updates).
        r=run_worker(self.config,self.task_id,self.folder/'results','checkpoint',path_override=self.path,resume=True)
        folder=self.folder/'results/checkpoint';rp=folder/'receipt.json';rp.write_text(json.dumps(r))
        runtime=dict(schema='timemixer-uv-owned-runtime-v2',candidate_id=self.config['candidate_id'],structural_revision=self.config['structural_revision'],
                     source_binding=worker.source_binding(),commit=r['manifest']['identity']['producer_commit'],synthetic=True,gate_policy=self.config['gate_policy'])
        child=dict(receipt=rp,expected_task_id=self.task_id)
        self.assertEqual(launcher.verify_child_receipt(child,self.config,folder,runtime,None),self.task_id)
        other=copy.deepcopy(self.config);t=copy.deepcopy(other['tasks'][0]);t['id']='synthetic-wrong-assignment';t.update(arm='U',use_v=False);other['tasks'].append(t)
        with self.assertRaises(ValueError):launcher.verify_child_receipt(dict(receipt=rp,expected_task_id=t['id']),other,folder,runtime,None)
        with self.assertRaises(ValueError):launcher.verify_child_receipt(child,self.config,self.folder/'different-attempt',runtime,None)
        with self.assertRaises(FileNotFoundError):launcher.verify_child_receipt(dict(receipt=self.folder/'missing.json',expected_task_id=self.task_id),self.config,folder,runtime,None)
        wrong=copy.deepcopy(runtime);wrong['source_binding']['wrong']='sha'
        with self.assertRaises(ValueError):launcher.verify_child_receipt(child,self.config,folder,wrong,None)
        half=self.folder/'half';half.mkdir()
        with self.assertRaises(FileNotFoundError):audit(half)
        copied=self.folder/'corrupt-copy';shutil.copytree(r['path'],copied);p=copied/'best.pt';raw=p.read_bytes();p.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
        with self.assertRaisesRegex(ValueError,'membership/SHA'):audit(copied)
        best=read(Path(r['path'])/'last.pt');self.assertEqual(best['scheduler']['updates'],4)
        # Real OneCycle LR/beta sequence is compared to the reviewed implementation.
        tr=self.config['tasks'][0]['structural_source_profile']['training'];p=torch.nn.Parameter(torch.ones(1));opt=torch.optim.Adam([p],lr=tr['lr'],betas=tuple(tr['betas']))
        schedule=Schedule(opt,tr['scheduler']);curve=[]
        for _ in range(4):p.grad=torch.ones_like(p);opt.step();schedule.after_successful_update();curve.append((opt.param_groups[0]['lr'],opt.param_groups[0]['betas'][0]))
        journal=[json.loads(l) for l in (Path(r['path'])/'updates.jsonl').read_text().splitlines()]
        self.assertEqual([(v['lr'],v['beta1']) for v in journal],curve)


if __name__=='__main__':unittest.main(verbosity=2)
