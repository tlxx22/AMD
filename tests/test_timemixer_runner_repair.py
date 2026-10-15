"""Targeted lifecycle, checkpoint and assignment regressions; CPU fixtures only."""
import copy
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from unittest.mock import patch
import numpy as np
import torch
from tests.test_timemixer_training import ROOT,fixture,adapter,_assert_nested_equal
from timemixer_dev import launcher,worker
from timemixer_dev.training_state import read,save,restore,validate_cuda_rng
from timemixer_dev.training_config import validate,digest
from utils.ch3_onecycle import Schedule

E=ROOT.parent/'amd-execution-evidence/m6/M6-Timemixer-runner-integration-r1/repair-r1'

def tree(folder):
    return {str(p.relative_to(folder)):('dir' if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mode&0o777) for p in folder.rglob('*')}

def live_state(model,opt,schedule,generator):
    return copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=schedule.state_dict(),
        python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),generator=generator.get_state()))

def run_cpu_supervisor(command,*,cwd,timeout):
    """Real CPU processes through the actual CLI, sharing a warmed CPU engine.

    This fixture changes process creation only. Production still uses exec;
    cold CLI import/update/test/publication is exercised by separate fixtures.
    """
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized() or '--synthetic-fixture-path' not in command or timeout!=45:raise AssertionError('bounded CPU fixture only')
    context=multiprocessing.get_context('fork');real_popen=subprocess.Popen
    def entry(argv,stdout,stderr,directory,new_session):
        try:
            if new_session:os.setsid()
            os.chdir(directory);os.dup2(stdout.fileno(),1);os.dup2(stderr.fileno(),2);sys.argv=argv[2:]
            import runpy
            runpy.run_path(argv[2],run_name='__main__')
        except SystemExit as error:
            if error.code not in (None,0):raise
        except BaseException:
            traceback.print_exc();raise
        finally:sys.stdout.flush();sys.stderr.flush()
    class WorkerProcess:
        def __init__(self,argv,**kwargs):
            stderr=kwargs['stdout'] if kwargs['stderr']==subprocess.STDOUT else kwargs['stderr']
            self.argv=argv;self.process=context.Process(target=entry,args=(argv,kwargs['stdout'],stderr,kwargs['cwd'],kwargs.get('start_new_session',False)))
            self.process.start();self.pid=self.process.pid
        @property
        def returncode(self):return self.process.exitcode
        def poll(self):self.process.join(0);return self.returncode
        def wait(self,timeout=None):
            self.process.join(timeout)
            if self.process.is_alive():raise subprocess.TimeoutExpired(self.argv,timeout)
            return self.returncode
    def factory(argv,**kwargs):
        if '--receipt' in argv and '--synthetic-fixture-path' in argv:return WorkerProcess(argv,**kwargs)
        return real_popen(argv,**kwargs)
    with tempfile.TemporaryFile() as out,tempfile.TemporaryFile() as err:
        def supervisor():
            with patch.object(subprocess,'Popen',side_effect=factory):entry(command,out,err,cwd,False)
        process=context.Process(target=supervisor);start=time.monotonic();process.start();process.join(timeout)
        if process.is_alive():
            process.kill();process.join();raise subprocess.TimeoutExpired(command,timeout)
        elapsed=time.monotonic()-start
        if elapsed>timeout:raise subprocess.TimeoutExpired(command,timeout)
        out.seek(0);err.seek(0)
        return subprocess.CompletedProcess(command,process.exitcode,out.read().decode(),err.read().decode())

class Repair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='.uv-runner-',dir=ROOT);cls.folder=Path(cls.tmp.name)
        cls.addClassCleanup(cls.preserve)
        cls.c,cls.id,cls.data=fixture(cls.folder)
        second=copy.deepcopy(cls.c['tasks'][0]);second['id']+='-second'
        third=copy.deepcopy(second);third.update(id=cls.id+'-U',arm='U',use_v=False)
        cls.c['tasks'] += [second,third];cls.second=second['id'];cls.third=third['id']
        cls.cp=cls.folder/'config.json';cls.cp.write_text(json.dumps(cls.c))
        cls.root=cls.folder/'results'
        cls.receipts={id:worker.run_worker(cls.c,id,cls.root,'origin-'+str(n),path_override=cls.data) for n,id in enumerate((cls.id,cls.second))}
        paused=worker.run_worker(cls.c,cls.id,cls.folder/'paused-results','paused',path_override=cls.data,pause_after_epoch=1)
        cls.checkpoint=read(Path(paused['staging'])/'last.pt')

    @classmethod
    def preserve(cls):
        shutil.copytree(cls.folder,E/('synthetic-fixtures-'+os.environ.get('UV_REPAIR_ID','manual')))
        cls.tmp.cleanup()

    def live(self):
        p=self.c['tasks'][0]['structural_source_profile'];tr=p['training']
        model=adapter.build(p,use_u=True,use_v=True,task='MS',fold=1)
        opt=torch.optim.Adam(model.parameters(),lr=tr['lr'],betas=tuple(tr['betas']),eps=tr['eps'],weight_decay=tr['weight_decay'])
        return model,opt,Schedule(opt,tr['scheduler']),torch.Generator().manual_seed(975)

    def test_Adam_activity_corruption_rejected_before_all_live_writes(self):
        state=self.checkpoint;key=next(iter(state['optimizer']['state']));name=next(iter(state['adam_activity']))
        self.assertTrue(state['adam_activity']);self.assertLess(len(state['adam_activity']),sum(len(g) for g in state['parameter_names']))
        model,opt,schedule,generator=self.live()
        cases=['all_state','empty_state_and_ledger','one_state','moment','zero_step','negative_step','fractional_step','excess_step','nan_step','step_dtype','step_rank','ledger_step','ledger_missing','CUDA_empty_GPU','CUDA_extra_CPU','CUDA_dtype','CUDA_rank','CUDA_missing_key','CUDA_metadata_missing']
        for kind in cases:
            s=copy.deepcopy(state)
            if kind=='all_state':s['optimizer']['state']={}
            elif kind=='empty_state_and_ledger':s['optimizer']['state']={};s['adam_activity']={}
            elif kind=='one_state':del s['optimizer']['state'][key]
            elif kind=='moment':del s['optimizer']['state'][key]['exp_avg']
            elif kind in ('zero_step','negative_step','fractional_step','excess_step'):s['optimizer']['state'][key]['step']=torch.tensor({'zero_step':0.,'negative_step':-1.,'fractional_step':1.5,'excess_step':99.}[kind])
            elif kind=='nan_step':s['optimizer']['state'][key]['step']=torch.tensor(float('nan'))
            elif kind=='step_dtype':s['optimizer']['state'][key]['step']=s['optimizer']['state'][key]['step'].double()
            elif kind=='step_rank':s['optimizer']['state'][key]['step']=s['optimizer']['state'][key]['step'].reshape(1)
            elif kind=='ledger_step':s['adam_activity'][name]+=1
            elif kind=='ledger_missing':del s['adam_activity'][name]
            elif kind=='CUDA_empty_GPU':s['rng_devices']['training_device']='cuda:0'
            elif kind=='CUDA_extra_CPU':s['rng']['cuda']=[torch.zeros(16,dtype=torch.uint8)]
            elif kind=='CUDA_dtype':s['rng']['cuda']=[torch.zeros(16,dtype=torch.float32)]
            elif kind=='CUDA_missing_key':del s['rng']['cuda']
            elif kind=='CUDA_metadata_missing':del s['rng_devices']
            else:s['rng']['cuda']=[torch.zeros((1,16),dtype=torch.uint8)]
            target=self.folder/('damaged-'+kind+'.pt');save(target,s);before=live_state(model,opt,schedule,generator)
            with self.subTest(kind=kind),self.assertRaises((ValueError,RuntimeError)):restore(target,model,opt,schedule,generator,state['identity'])
            _assert_nested_equal(self,live_state(model,opt,schedule,generator),before)

    def test_CUDA_metadata_validation_without_initializing_GPU(self):
        self.assertFalse(torch.cuda.is_initialized())
        expected=dict(training_device='cuda:0',gpu_uuid='GPU-fixture',cuda_visible_devices='synthetic-device',cuda_devices=[dict(index=0,name='synthetic',uuid='GPU-fixture',total_memory=1,capability=[1,0],state_bytes=16)])
        states=[torch.zeros(16,dtype=torch.uint8)];validate_cuda_rng(expected,states,expected)
        model,opt,schedule,generator=self.live();before=live_state(model,opt,schedule,generator)
        for kind in ('missing','dtype','rank','length','metadata','device','index','noncontiguous'):
            m=copy.deepcopy(expected);s=copy.deepcopy(states)
            if kind=='missing':s=[]
            elif kind=='dtype':s=[s[0].float()]
            elif kind=='rank':s=[s[0].reshape(2,8)]
            elif kind=='length':s=[torch.zeros(15,dtype=torch.uint8)]
            elif kind=='metadata':m['cuda_visible_devices']='foreign'
            elif kind=='device':m['training_device']='cuda:1'
            elif kind=='index':m['cuda_devices'][0]['index']=1
            else:s=[torch.zeros(32,dtype=torch.uint8)[::2]]
            with self.subTest(kind=kind),self.assertRaises(ValueError):validate_cuda_rng(m,s,expected)
            _assert_nested_equal(self,live_state(model,opt,schedule,generator),before)
        self.assertTrue(torch.equal(torch.get_rng_state(),before['torch']));self.assertFalse(torch.cuda.is_initialized())

    def test_supervise_refusal_does_not_mutate_attempt_or_signal_owner(self):
        cfg=copy.deepcopy(self.c);cfg.update(purpose='development',global_lock=str(self.folder/'temporary-global.lock'))
        cp=self.folder/'lifecycle.json';cp.write_text(json.dumps(cfg))
        owner=subprocess.Popen([sys.executable,'-B','-c','import sys; sys.stdin.readline()'],stdin=subprocess.PIPE)
        try:
            for kind in ('lock','live','completed','authorization','previous_source'):
                folder=self.root/('refuse-'+kind);folder.mkdir();grant={}
                previous=dict(commit='fixture',task_ids=[self.id],owner_pid=999999999,owner_start='0',config_sha='wrong',source_binding={})
                refpath=self.folder/('previous-'+kind+'.json');refpath.write_text(json.dumps(previous));ref=dict(path=str(refpath),sha256=hashlib.sha256(refpath.read_bytes()).hexdigest())
                own=dict(pid=owner.pid if kind=='live' else 999999999,start=launcher.proc_start(owner.pid) if kind=='live' else '0',runtime_ref=ref)
                for name,value in [('owner.json',own),('progress.json',dict(state='original')),('failure.json',dict(original=True))]:(folder/name).write_text(json.dumps(value)+'\n')
                (folder/'STOP').write_text('original STOP\n')
                if kind=='completed':(folder/'complete.json').write_text('{"technical_complete":true}\n')
                if kind=='previous_source':grant=dict(resume_review_approved=True,resume_task_ids=[self.id],previous_runtime_ref=ref,commit='fixture')
                auth=self.folder/('auth-'+kind+'.json');auth.write_text(json.dumps(grant));before=tree(folder)
                with open(cfg['global_lock'],'a') as held:
                    if kind=='lock':fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    with patch.object(worker,'RUN_NAMESPACE',self.root),patch.object(launcher,'validate',side_effect=lambda c:c),patch.object(launcher,'validate_authorization',return_value=grant),patch.object(launcher,'gpu_startup',side_effect=AssertionError('GPU preflight must not run for rejected ownership')) as gpu:
                        with self.subTest(kind=kind),self.assertRaises((PermissionError,BlockingIOError)):launcher.supervise(cp,[self.id],self.root,folder.name,auth,resume=True)
                        gpu.assert_not_called()
                self.assertEqual(tree(folder),before);self.assertIsNone(owner.poll())
        finally:owner.stdin.write(b'\n');owner.stdin.flush();owner.wait(timeout=10);owner.stdin.close()

    def transport(self,mode,config,root,attempt,ids,*,resume=False):
        calls=[];self.last_transport_calls=calls;real_popen=subprocess.Popen
        class Finished:
            pid=os.getpid();returncode=0
            def __init__(inner,command,**kwargs):
                id=command[command.index('--task-id')+1];calls.append(id)
                if mode=='missing':return
                source_id=self.second if resume else self.id;receipt=copy.deepcopy(self.receipts[source_id])
                if mode!='other_attempt':
                    destination=root/attempt/'runs'/Path(receipt['path']).name
                    if not destination.exists():destination.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(receipt['path'],destination)
                    receipt['path']=str(destination)
                Path(command[command.index('--receipt')+1]).write_text(json.dumps(receipt))
            def poll(inner):return 0
            def wait(inner,**kwargs):return 0
        def spawn(command,**kwargs):return Finished(command,**kwargs) if '--receipt' in command else real_popen(command,**kwargs)
        cp=self.folder/('transport-'+attempt+'.json');cp.write_text(json.dumps(config))
        with patch.object(launcher.subprocess,'Popen',side_effect=spawn):
            launcher.supervise(cp,ids,root,attempt,synthetic_fixture_path=self.data,resume=resume)
        return calls

    def test_receipt_negatives_block_next_wave_and_completion(self):
        for kind in ('wrong_task','duplicate','other_attempt','missing','config','fold','producer','source'):
            c=copy.deepcopy(self.c);ids=[self.id,self.third]
            if kind=='wrong_task':ids=[self.second,self.third]
            if kind=='duplicate':ids=[self.id,self.second,self.third]
            if kind=='config':c['fixture_binding']='different'
            if kind=='fold':c['tasks'][0]['fold']=4
            name='bad-receipt-'+kind
            context=patch.object(launcher,'git',return_value='foreign-producer') if kind=='producer' else patch.object(worker,'source_binding',return_value={'foreign-source':'wrong'}) if kind=='source' else patch.object(launcher,'git',wraps=launcher.git)
            with context,self.subTest(kind=kind),self.assertRaises((ValueError,FileNotFoundError)):self.transport(kind,c,self.root,name,ids)
            folder=self.root/name
            self.assertTrue((folder/'failure.json').exists());self.assertTrue((folder/'STOP').exists());self.assertFalse((folder/'complete.json').exists())
            self.assertEqual(json.loads((folder/'progress.json').read_text())['wave'],0)
            self.assertNotIn(self.third,self.last_transport_calls)

    def test_resume_subset_accounts_only_new_verified_ids(self):
        folder=self.root/'subset';folder.mkdir();old=copy.deepcopy(self.receipts[self.id]);dest=folder/'runs'/Path(old['path']).name;dest.parent.mkdir();shutil.copytree(old['path'],dest);before=tree(dest)
        (folder/'owner.json').write_text(json.dumps(dict(pid=999999999,start='0')));(folder/'STOP').write_text('old STOP\n')
        self.assertEqual(self.transport('valid',self.c,self.root,'subset',[self.second],resume=True),[self.second])
        complete=json.loads((folder/'complete.json').read_text());self.assertEqual(complete['task_ids'],[self.second]);self.assertTrue(complete['technical_complete'])
        self.assertEqual(tree(dest),before);self.assertEqual(json.loads((folder/'summary.json').read_text())['completed'],2)

    def test_resume_rejects_already_completed_subset_without_changes(self):
        folder=self.root/'subset-completed';folder.mkdir();dest=folder/'runs'/Path(self.receipts[self.id]['path']).name;dest.parent.mkdir();shutil.copytree(self.receipts[self.id]['path'],dest)
        (folder/'owner.json').write_text(json.dumps(dict(pid=999999999,start='0')));(folder/'STOP').touch();before=tree(folder)
        with self.assertRaisesRegex(PermissionError,'already completed'):launcher.supervise(self.cp,[self.id],self.root,folder.name,synthetic_fixture_path=self.data,resume=True)
        self.assertEqual(tree(folder),before)

    def test_stored_resolved_config_science_and_current_source_binding(self):
        stored=json.loads((ROOT/'configs/timemixer_uv_round2_resolved_v1.json').read_text());validate(stored)
        old=json.loads((E/'before/configs/timemixer_uv_round2_resolved_v1.json').read_text());allowed={'source_binding','checkpoint_schema'}
        self.assertEqual({k:v for k,v in stored.items() if k not in allowed},{k:v for k,v in old.items() if k not in allowed})
        self.assertEqual(stored['source_binding'],worker.source_binding());self.assertEqual(len(stored['tasks']),77);self.assertEqual(stored['max_run_epochs'],1340)
