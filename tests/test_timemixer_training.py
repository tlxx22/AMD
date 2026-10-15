"""Explicit bounded CPU acceptance for the actual candidate training pipeline."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import uuid
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from timemixer_dev.training_config import ROOT,derive,validate,arithmetic,digest
from timemixer_dev.worker import run_worker
from timemixer_dev import worker,adapter
from timemixer_dev.training_state import read,validate as validate_state,file_sha
from timemixer_dev.artifacts import audit
from timemixer_dev.launcher import batch,validate_authorization,stop,status
from timemixer_dev.summary import summarize
from timemixer_dev.gate_policy import policy
from utils.ch3_onecycle import configuration,Schedule
from utils.ch3_contract import BestState
from tests.test_runner import _assert_nested_equal

EVIDENCE=ROOT.parent/'amd-execution-evidence/m6/M6-Timemixer-runner-integration-r1'
REAL=None
UPDATES=0
CHILD_UPDATES=0

def actual_config():
    global REAL
    if REAL is None:REAL=derive()
    return REAL

def fixture(folder,dataset='UrbanEV',variant='F4',arm='UV',H=3,epochs=3,patience=None):
    real=actual_config();source=next(t for t in real['tasks'] if t['dataset']==dataset and t['input_variant']==variant and t['h']==H)
    t=copy.deepcopy(source);t.update(arm=arm,use_u=arm in ('U','UV'),use_v=arm in ('V','UV'))
    t['id']='synthetic-'+dataset+'-'+variant+'-'+arm+'-h'+str(H)
    p=t['structural_source_profile'];p['structure'].update(d_model=8,d_ff=16,e_layers=1)
    tr=p['training'];tr.update(batch=16 if dataset=='UrbanEV' and H==3 else (8 if dataset=='UrbanEV' else 4),eval_batch=11 if dataset=='UrbanEV' else 4,epochs=epochs,patience=patience,threads=1)
    data=t['data_config'];d=data['datasets'][dataset];t['data_stat']={};t['baseline_prefix_metadata']={}
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if dataset=='UrbanEV':
        data['urban_folds']=[[32,64,96]]*6;data['synthetic_nodes']=2;n=96;clock=pd.date_range('2022-09-01',periods=n,freq='h');path=folder/'urban';path.mkdir(exist_ok=True)
        for name,cols in [('volume.csv',['n0','n1']),('e_price.csv',['n0','n1']),('s_price.csv',['n0','n1']),('weather_central.csv',['T','P','U'])]:
            rng=np.random.RandomState(22+len(name));frame=pd.DataFrame({c:np.arange(n)/30+np.sin(np.arange(n)/7+i)+rng.randn(n)*.1 for i,c in enumerate(cols)});frame.insert(0,'timestamp',clock.strftime('%Y-%m-%d %H:%M:%S'));frame.to_csv(path/name,index=False)
    else:
        a,b,c=(200,304,408) if p['pred_len']==96 else (129,161,193);d['endpoints']=[a,b,c];n=c
        clock=pd.date_range('2022-09-01',periods=n,freq='15min' if p['time_mark']['freq']=='t' else 'h');path=folder/'series.csv';rng=np.random.RandomState(381)
        frame=pd.DataFrame({name:np.sin(np.arange(n)/13+i)+np.arange(n)/100+rng.randn(n)*.2 for i,name in enumerate(p['features'])});frame.insert(0,'date',clock.strftime('%Y-%m-%d %H:%M:%S'));frame.to_csv(path,index=False)
    d['path']=str(path);d.pop('source_admission',None)
    tr['scheduler']=configuration(epochs,2);t['profile_sha']=digest(p);t['arithmetic']=arithmetic(data,p,t['fold'])
    config=dict(schema=real['schema'],candidate_id=real['candidate_id'],training_protocol='synthetic_round2_pipeline_v1',purpose='synthetic',synthetic_root=str(folder),tasks=[t],existing_B={},F4_reuse={},gate_policy=policy(),execution_authorized=False)
    validate(config)
    return config,t['id'],path

class Pipeline(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='.uv-runner-',dir=ROOT);self.folder=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def make(self,**kwargs):return fixture(self.folder,**kwargs)
    def run_case(self,c,id,path,attempt='trial',**kw):return run_worker(c,id,self.folder/'results',attempt,path_override=path,**kw)

    def test_77_resolved_second_round_and_original_review(self):
        c=actual_config();self.assertEqual(len(c['tasks']),77);self.assertEqual(sum(t['arithmetic']['max_epochs'] for t in c['tasks']),1340);self.assertEqual(len(c['existing_B']),37)
        for t in c['tasks']:
            p=t['structural_source_profile'];self.assertEqual(p['training'],t['baseline_training']);self.assertEqual(p['T'],12 if t['dataset']=='UrbanEV' else 96);self.assertEqual(t['arithmetic']['max_optimizer_steps'],p['training']['epochs']*p['training']['scheduler']['steps_per_epoch'])
        self.assertEqual(c['gate_policy'],policy());self.assertNotIn('gate_main_metric',c);self.assertFalse(c['execution_authorized'])
        self.assertEqual(len(c['F4_reuse']),8)

    def test_four_arms_actual_update_validation_test_and_off_reference_exact(self):
        results={}
        for arm in ('B','U','V','UV'):
            c,id,path=self.make(arm=arm);results[arm]=self.run_case(c,id,path,attempt=arm)
            m=results[arm]['manifest'];self.assertEqual(m['steps'],6);self.assertTrue(all(m['routes'][phase]>0 for phase in ('train','validation','test')));self.assertEqual(m['identity']['task']['model'],'timemixer-uv-dev-v1');self.assertEqual(m['test_calls'],1)
            self.assertEqual(results[arm]['metrics']['elements'],36)
        c,id,path=self.make(arm='B');ref=self.run_case(c,id,path,attempt='reference',reference_original=True)
        a=read(Path(results['B']['path'])/'last.pt');b=read(Path(ref['path'])/'last.pt')
        for name in ('model','optimizer','scheduler','rng','best','steps','epoch'):_assert_nested_equal(self,a[name],b[name])
        self.assertEqual(results['B']['metrics']['mse'],ref['metrics']['mse']);self.assertEqual(results['B']['metrics']['mae'],ref['metrics']['mae'])
        # Preserve one genuinely published synthetic artifact outside temporary fixtures.
        dest=EVIDENCE/'synthetic-artifacts'/('worker-'+os.environ.get('UV_SUITE_ID','manual'));dest.parent.mkdir(exist_ok=True)
        shutil.copytree(results['UV']['path'],dest);self.assertEqual(audit(dest)['metrics']['test_calls'],1)

    def test_real_loader_F1_F2_F3_and_H12_P1(self):
        for variant in ('F1','F2','F3'):
            c,id,path=self.make(variant=variant);r=self.run_case(c,id,path,attempt=variant)
            self.assertEqual(r['manifest']['identity']['profile']['C'],{'F1':6,'F2':8,'F3':9}[variant]);self.assertEqual(r['metrics']['elements'],36)
        c,id,path=self.make(H=12);r=self.run_case(c,id,path,attempt='H12');self.assertEqual(r['metrics']['elements'],18);self.assertEqual(r['manifest']['identity']['profile']['pred_len'],1)

    def test_real_loader_EPF_L96_P24_and_M_all_channels_tails(self):
        for dataset,variant,H,expected in [('PJM','MS',24,216),('ETTh1','M',96,6048),('ETTm1','M',96,6048),('Weather','M',96,18144)]:
            c,id,path=self.make(dataset=dataset,variant=variant,H=H);r=self.run_case(c,id,path,attempt=dataset)
            self.assertEqual(r['metrics']['elements'],expected);self.assertEqual(r['manifest']['identity']['profile']['T'],96)
            self.assertEqual(r['metrics']['metric_scope'],'all_channels' if variant=='M' else 'target_only');self.assertEqual(r['metrics']['test_calls'],1)

    def test_epoch_resume_matches_uninterrupted_all_state_and_order(self):
        c,id,path=self.make();full=run_worker(c,id,self.folder/'uninterrupted-results','full',path_override=path);paused=self.run_case(c,id,path,attempt='resume',pause_after_epoch=1)
        self.assertEqual(paused['steps'],2);resumed=self.run_case(c,id,path,attempt='resume',resume=True)
        a=read(Path(full['path'])/'last.pt');b=read(Path(resumed['path'])/'last.pt');_assert_nested_equal(self,a,b)
        self.assertEqual((Path(full['path'])/'updates.jsonl').read_text(),(Path(resumed['path'])/'updates.jsonl').read_text());self.assertEqual(full['metrics'],resumed['metrics'])

    def test_checkpoint_prevalidation_rejects_all_before_live_writes(self):
        c,id,path=self.make();paused=self.run_case(c,id,path,pause_after_epoch=1);state=read(Path(paused['staging'])/'last.pt');identity=state['identity'];p=c['tasks'][0]['structural_source_profile']
        model=adapter.build(p,use_u=True,use_v=True,task='MS',fold=1);tr=p['training'];opt=torch.optim.Adam(model.parameters(),lr=tr['lr'],betas=tuple(tr['betas']),eps=tr['eps'],weight_decay=tr['weight_decay']);schedule=Schedule(opt,tr['scheduler'])
        before=copy.deepcopy(model.state_dict());before_opt=copy.deepcopy(opt.state_dict())
        for kind in ('arm','F','H','protocol','source','data','schema','dtype','optimizer','scheduler','rng','progress'):
            s=copy.deepcopy(state)
            if kind=='arm':s['identity']['task']['arm']='V'
            elif kind=='F':s['identity']['task']['input_variant']='F1'
            elif kind=='H':s['identity']['task']['h']=12
            elif kind=='protocol':s['identity']['training_protocol']='third'
            elif kind=='source':s['identity']['source']['wrong']='SHA'
            elif kind=='data':s['identity']['data_sha']='wrong'
            elif kind=='schema':s['schema']='old'
            elif kind=='dtype':k=next(iter(s['model']));s['model'][k]=s['model'][k].double()
            elif kind=='optimizer':s['optimizer']['param_groups'][0]['weight_decay']=.5
            elif kind=='scheduler':s['scheduler']['updates']+=1
            elif kind=='rng':s['rng']['torch']=torch.zeros(2,dtype=torch.float32)
            else:s['steps']+=1
            with self.subTest(kind=kind),self.assertRaises((ValueError,RuntimeError)):validate_state(s,model,opt,schedule,identity)
            _assert_nested_equal(self,model.state_dict(),before);_assert_nested_equal(self,opt.state_dict(),before_opt)

    def test_uncertain_test_refuses_repeat(self):
        c,id,path=self.make()
        with self.assertRaisesRegex(RuntimeError,'test uncertainty'):self.run_case(c,id,path,fault='after_test_claim')
        with self.assertRaisesRegex(ValueError,'cannot repeat'):self.run_case(c,id,path,resume=True)

    def test_saved_test_reused_after_sealing_failure(self):
        c,id,path=self.make()
        with self.assertRaisesRegex(RuntimeError,'sealing failure'):self.run_case(c,id,path,fault='before_publish')
        result=self.run_case(c,id,path,resume=True);self.assertEqual(result['metrics']['test_calls'],1);self.assertEqual(result['manifest']['routes']['test'],4)

    def test_STOP_failure_mid_epoch_and_duplicate_completed(self):
        c,id,path=self.make();root=self.folder/'results';(root/'stopped').mkdir(parents=True);(root/'stopped/STOP').touch()
        with self.assertRaises(InterruptedError):self.run_case(c,id,path,attempt='stopped')
        with self.assertRaisesRegex(RuntimeError,'worker failure'):self.run_case(c,id,path,attempt='failed',fault='after_update')
        with self.assertRaisesRegex(ValueError,'before first epoch'):self.run_case(c,id,path,attempt='failed',resume=True)
        r=self.run_case(c,id,path,attempt='completed');before={p.name:file_sha(p) for p in Path(r['path']).iterdir()}
        with self.assertRaises(FileExistsError):self.run_case(c,id,path,attempt='duplicate')
        self.assertEqual(before,{p.name:file_sha(p) for p in Path(r['path']).iterdir()})

    def test_artifact_corruption_half_directory_and_no_unknown_gate(self):
        c,id,path=self.make();r=self.run_case(c,id,path);report=summarize(c,self.folder/'results','trial');self.assertEqual(report['scientific_gate'],'Not evaluated');self.assertFalse(report['coverage_complete'])
        half=self.folder/'half';half.mkdir()
        with self.assertRaises(FileNotFoundError):audit(half)
        f=Path(r['path'])/'best.pt';v=f.read_bytes();f.write_bytes(v[:-1]+bytes([v[-1]^1]))
        with self.assertRaisesRegex(ValueError,'membership/SHA'):audit(r['path'])

    def test_strict_best_tie_patience_and_actual_early_stop(self):
        b=BestState(2);self.assertTrue(b.update(1.,1));self.assertFalse(b.update(1.,2));self.assertFalse(b.stopped);self.assertFalse(b.update(2.,3));self.assertTrue(b.stopped);self.assertEqual(b.epoch,1)
        c,id,path=self.make(epochs=4,patience=1);original=worker.evaluate
        def controlled(*args,**kwargs):
            result=original(*args,**kwargs)
            if args[-1]=='validation':result.update(mse=1.,sse=float(result['elements']),synthetic_controlled_validation=True)
            return result
        with patch.object(worker,'evaluate',side_effect=controlled):r=self.run_case(c,id,path)
        self.assertEqual((r['manifest']['epochs'],r['manifest']['steps'],r['metrics']['best_epoch']),(2,4,1));self.assertEqual(read(Path(r['path'])/'best.pt')['epoch'],1)

    def test_fixed_explicit_waves_and_no_unreviewed_real_grant(self):
        c=actual_config();ids=[t['id'] for t in c['tasks'][:17]];waves=batch(c,ids);self.assertEqual(sum(waves,[]).__len__(),17);self.assertTrue(all(len(x)<=4 for x in waves));self.assertEqual(set(sum(waves,[])),set(ids))
        with self.assertRaises(ValueError):batch(c,ids+[ids[0]])
        with self.assertRaises(PermissionError):validate_authorization(c,'unused',ids,{'review_approved':False})

    def test_OneCycle_reference_actual_lr_beta_and_eval_no_steps(self):
        c,id,path=self.make();r=self.run_case(c,id,path);tr=c['tasks'][0]['structural_source_profile']['training'];p=torch.nn.Parameter(torch.ones(1));opt=torch.optim.Adam([p],lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=1e-7)
        schedule=torch.optim.lr_scheduler.OneCycleLR(opt,**{k:v for k,v in tr['scheduler'].items() if k!='name'})
        self.assertEqual(tr['scheduler']['max_lr']/tr['scheduler']['div_factor'],.0004)
        self.assertEqual(opt.param_groups[0]['lr'],.0003999999999999993)
        start=json.loads((Path(r['path'])/'worker.log').read_text().splitlines()[0]);self.assertEqual(start['initial_lr'],opt.param_groups[0]['lr']);self.assertEqual(start['initial_beta1'],opt.param_groups[0]['betas'][0])
        curve=[]
        for i in range(6):p.grad=torch.ones_like(p);opt.step();schedule.step();curve.append((opt.param_groups[0]['lr'],opt.param_groups[0]['betas'][0]))
        journal=[json.loads(x) for x in (Path(r['path'])/'updates.jsonl').read_text().splitlines()];self.assertEqual([(x['lr'],x['beta1']) for x in journal],curve);self.assertEqual(read(Path(r['path'])/'last.pt')['scheduler']['updates'],6)

    def test_empty_owned_tmux_guardian_stop_and_status(self):
        self.assertIsNotNone(shutil.which('tmux'))
        root=EVIDENCE/'synthetic-daemon';attempt='synthetic-'+uuid.uuid4().hex[:10];session='uv-empty-'+uuid.uuid4().hex[:10]
        command=['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',sys.executable,'-B',str(ROOT/'timemixer_train_entry.py'),'supervise','--empty-synthetic','--root',str(root),'--attempt',attempt]
        import shlex
        shell='exec '+shlex.join(command)+' > '+shlex.quote(str(EVIDENCE/(attempt+'.log')))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),shell],check=True)
        try:
            end=time.monotonic()+30
            while not (root/attempt/'owner.json').exists() and time.monotonic()<end:time.sleep(.1)
            self.assertTrue((root/attempt/'owner.json').exists());self.assertEqual(status(root,attempt)['progress.json']['state'],'EMPTY_SYNTHETIC_WAIT')
            stop(root,attempt)
            while not (root/attempt/'stopped.json').exists() and time.monotonic()<end:time.sleep(.1)
            self.assertTrue((root/attempt/'stopped.json').exists());self.assertFalse((root/attempt/'complete.json').exists())
        finally:
            if (root/attempt/'owner.json').exists() and not (root/attempt/'stopped.json').exists():(root/attempt/'STOP').touch()

    def test_actual_supervisor_child_failure_prevents_next_wave(self):
        global CHILD_UPDATES
        c,id,path=self.make();first=c['tasks'][0];first['synthetic_fault']='after_update';second=copy.deepcopy(first);second.update(id=id+'-second',arm='U',use_v=False);second.pop('synthetic_fault');c['tasks'].append(second)
        config=self.folder/'fixture.json';config.write_text(json.dumps(c));ids=self.folder/'ids.json';ids.write_text(json.dumps([id,second['id']]))
        root=self.folder/'supervised';attempt='synthetic-failure'
        result=subprocess.run([sys.executable,'-B',str(ROOT/'timemixer_train_entry.py'),'supervise','--config',str(config),'--task-list',str(ids),'--root',str(root),'--attempt',attempt,'--synthetic-fixture-path',str(path)],cwd=ROOT,capture_output=True,text=True,timeout=30)
        self.assertNotEqual(result.returncode,0,result.stdout);self.assertTrue((root/attempt/'failure.json').exists());self.assertEqual(json.loads((root/attempt/'progress.json').read_text())['wave'],0)
        stages=list((root/attempt/'staging').iterdir());self.assertEqual(len(stages),1);self.assertEqual(len((stages[0]/'updates.jsonl').read_text().splitlines()),1)
        (EVIDENCE/('child-consumption-'+os.environ.get('UV_SUITE_ID','manual')+'.json')).write_text(json.dumps(dict(actual_Adam_updates=1,purpose='one failed owned synthetic child; next wave not dispatched'))+'\n')
        CHILD_UPDATES+=1
        shutil.copytree(root/attempt,EVIDENCE/('supervisor-failure-'+os.environ.get('UV_SUITE_ID','manual')))

    def test_successful_owned_supervisor_advances_two_audited_waves(self):
        global CHILD_UPDATES
        c,id,path=self.make();self.assertEqual(c['tasks'][0]['arithmetic']['train_batches'],2)
        second=copy.deepcopy(c['tasks'][0]);second.update(id=id+'-second',arm='U',use_v=False);c['tasks'].append(second)
        config=self.folder/'fixture.json';config.write_text(json.dumps(c));ids=self.folder/'ids.json';ids.write_text(json.dumps([id,second['id']]))
        root=self.folder/'supervised';attempt='synthetic-success'
        timed_out=False
        try:
            from tests.test_timemixer_runner_repair import run_cpu_supervisor
            result=run_cpu_supervisor([sys.executable,'-B',str(ROOT/'timemixer_train_entry.py'),'supervise','--config',str(config),'--task-list',str(ids),'--root',str(root),'--attempt',attempt,'--synthetic-fixture-path',str(path)],cwd=ROOT,timeout=45)
        except subprocess.TimeoutExpired:
            timed_out=True
            raise
        finally:
            if timed_out and (root/attempt/'children.json').exists():
                from timemixer_dev.launcher import proc_start
                import signal
                (root/attempt/'STOP').touch(exist_ok=True)
                owned=json.loads((root/attempt/'children.json').read_text())
                def active(child):
                    try:
                        status=Path('/proc/'+str(child['pid'])+'/stat').read_text()
                        return proc_start(child['pid'])==child['start'] and status[status.rfind(')')+2:].split()[0]!='Z'
                    except (FileNotFoundError,ProcessLookupError):return False
                end=time.monotonic()+15
                while any(active(c) for c in owned) and time.monotonic()<end:time.sleep(.1)
                for child in owned:
                    if active(child) and os.getpgid(child['pid'])==child['pid']:os.killpg(child['pid'],signal.SIGTERM)
                end=time.monotonic()+2
                while any(active(c) for c in owned) and time.monotonic()<end:time.sleep(.1)
                for child in owned:
                    if active(child) and os.getpgid(child['pid'])==child['pid']:os.killpg(child['pid'],signal.SIGKILL)
            saved=EVIDENCE/('supervisor-success-'+os.environ.get('UV_SUITE_ID','manual'))
            if (root/attempt).exists():
                shutil.copytree(root/attempt,saved)
                CHILD_UPDATES+=sum(len(p.read_text().splitlines()) for section in ('staging','runs') for p in (saved/section).glob('*/updates.jsonl'))
        self.assertEqual(result.returncode,0,result.stderr);complete=json.loads((root/attempt/'complete.json').read_text());self.assertTrue(complete['technical_complete']);self.assertEqual(complete['waves'],2);self.assertEqual(complete['scientific_gate'],'Not evaluated')
        receipts=[audit(p) for p in (root/attempt/'runs').iterdir()];self.assertEqual(len(receipts),2);self.assertEqual(sum(r['manifest']['steps'] for r in receipts),12)

    def test_M_off_original_worker_and_legacy_aggregate_exact(self):
        from ch3_runner import evaluate as legacy_evaluate
        from utils.ch3_data import load,batches
        from models.ch3_adapter import target_prediction
        for dataset,variant,H in [('ETTh1','M',96),('PJM','MS',24)]:
            c,id,path=self.make(dataset=dataset,variant=variant,H=H,arm='B');a=self.run_case(c,id,path,attempt=dataset+'-candidate');b=self.run_case(c,id,path,attempt=dataset+'-reference',reference_original=True)
            x=read(Path(a['path'])/'last.pt');y=read(Path(b['path'])/'last.pt')
            for name in ('model','optimizer','scheduler','rng','best','steps','epoch'):_assert_nested_equal(self,x[name],y[name])
            self.assertEqual(a['metrics']['mse'],b['metrics']['mse']);self.assertEqual(a['metrics']['mae'],b['metrics']['mae'])
            t=c['tasks'][0];p=t['structural_source_profile'];ds,meta=load(t['data_config'],t,resolved_profile=p,path_override=path);m=adapter.build_original(p)
            old=legacy_evaluate(m,batches(ds['validation'],p,'validation'),p,'cpu')
            new=worker.evaluate(m,ds['validation'],p,'cpu',lambda model,x,mark:target_prediction(model,x,p,mark)[0],dict(validation=0),'validation')
            for name in ('mse','mae','sse','sae','elements'):self.assertEqual(old[name],new[name])

    def test_resume_owner_guard_and_control_history(self):
        from timemixer_dev.launcher import prepare_resume,proc_start
        folder=self.folder/'owned';folder.mkdir();owner=dict(pid=os.getpid(),start=proc_start(os.getpid()));(folder/'owner.json').write_text(json.dumps(owner));(folder/'STOP').touch()
        with self.assertRaisesRegex(PermissionError,'still running'):prepare_resume(folder,['one'],{},synthetic=True)
        self.assertTrue((folder/'STOP').exists());(folder/'owner.json').write_text(json.dumps(dict(pid=999999999,start='0')))
        with self.assertRaisesRegex(PermissionError,'reviewed resume'):prepare_resume(folder,['one'],{},synthetic=False)
        prepare_resume(folder,['one'],{},synthetic=True);self.assertFalse((folder/'STOP').exists());self.assertEqual(len(list((folder/'control-history').glob('*/STOP'))),1)

    def test_resume_journal_corruption_is_rejected(self):
        c,id,path=self.make();paused=self.run_case(c,id,path,pause_after_epoch=1);journal=Path(paused['staging'])/'updates.jsonl';lines=journal.read_text().splitlines();row=json.loads(lines[0]);row['loss']+=1;lines[0]=json.dumps(row);journal.write_text('\n'.join(lines)+'\n')
        with self.assertRaisesRegex(ValueError,'journal checksum'):self.run_case(c,id,path,resume=True)

    def test_prepublication_corrupt_checkpoint_cannot_be_resealed(self):
        c,id,path=self.make()
        with self.assertRaisesRegex(RuntimeError,'sealing failure'):self.run_case(c,id,path,fault='before_publish')
        stage=next((self.folder/'results/trial/staging').iterdir());f=stage/'best.pt';data=f.read_bytes();f.write_bytes(data[:-1]+bytes([data[-1]^1]))
        with self.assertRaisesRegex(ValueError,'sidecar checksum'):self.run_case(c,id,path,resume=True)
        self.assertFalse((self.folder/'results/trial/runs').exists())

    def test_cross_attempt_summary_and_success_identity_guard(self):
        c,id,path=self.make();second=copy.deepcopy(c['tasks'][0]);second.update(id=id+'-second',arm='U',use_v=False);c['tasks'].append(second)
        a=self.run_case(c,id,path,attempt='batch1');b=self.run_case(c,second['id'],path,attempt='batch2');result=summarize(c,self.folder/'results',None)
        self.assertEqual(result['completed'],2);self.assertEqual(result['scientific_gate'],'Not evaluated')
        with self.assertRaisesRegex(ValueError,'startup contracts'):summarize(c,self.folder/'results',None,'mse')
        identity=a['manifest']['identity'];changed=copy.deepcopy(identity);changed.update(producer_commit='doc-only',gate_main_metric='mae')
        self.assertEqual(worker.successful_identity(identity),worker.successful_identity(changed))

if __name__=='__main__':
    original_step=torch.optim.Adam.step
    def counted(self,*args,**kwargs):
        global UPDATES
        result=original_step(self,*args,**kwargs);UPDATES+=1
        if UPDATES>220:raise AssertionError('predeclared per-suite Adam update cap exceeded')
        return result
    torch.optim.Adam.step=counted
    try:result=unittest.main(verbosity=2,exit=False).result
    finally:
        torch.optim.Adam.step=original_step
        (EVIDENCE/('consumption-'+os.environ.get('UV_SUITE_ID','manual')+'.json')).write_text(json.dumps(dict(actual_Adam_updates=UPDATES,owned_child_Adam_updates=CHILD_UPDATES,total_Adam_updates=UPDATES+CHILD_UPDATES,purpose='CPU synthetic including direct scheduler reference',cap=220))+'\n')
        assert UPDATES+CHILD_UPDATES<=220
    sys.exit(0 if result.wasSuccessful() else 1)
