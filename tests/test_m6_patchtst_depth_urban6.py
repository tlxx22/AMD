"""Bounded CPU-only acceptance for exit observation and the exact continuation."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from utils.ch3_contract import ROOT,digest,profile,step_arithmetic
from utils.ch3_native_recovery_records import bound,exclusive,ref
from utils import ch3_patchtst_depth_urban6_recovery as r
r.activate()
from utils import ch3_type1_tasks as s,ch3_type1_chain as q
from tools.restricted_regression.m5_formal_entry import ExitObservation,owned_pid_metadata,resource_assessment,gpu_sample


def sample(now,active=True,nvml=True,**extra):
    value=dict(time=now,uuid='fixture-uuid',total=32*1024**3,used=1024**3,free=31*1024**3,driver_reserved=0,nvml_processes={'110':100}if nvml else{},process_gpu={'10':100}if active else{},cpu_rss={'10':1000},owned_host_pids=['110']if active else[],process_table_reliable=True,owned_pid_metadata={'10':dict(pid=10,host_pid=110,start_ticks='100',namespace='pid:[1]')if active else dict(pid=10,host_pid=None,state='exited_during_sample')})
    value.update(extra);return value


class ExitCases(unittest.TestCase):
    def test_original_slow_sampling_sequence_settles_only_after_fresh_clean_tail(self):
        source=bound(r.SOURCE_REF);v=bound(r.REUSE_REF);p=Path(v['retained_failed']['process_ref']['path']);rows=[json.loads(x)for x in p.with_name('memory.jsonl').read_text().splitlines()]
        observation=ExitObservation(bound(v['retained_failed']['process_ref'])['baseline'])
        pid='14330'
        for row in rows[:-1]:observation.classify(row,[pid],[pid])
        with self.assertRaises(ValueError):observation.classify(rows[-1],[pid],[])
        # The historical ambiguous read remains rejected; only this explicitly
        # synthetic, proven disappearance exercises residual-to-clean settling.
        synthetic=copy.deepcopy(rows[-1]);synthetic['owned_pid_metadata']={pid:dict(pid=int(pid),host_pid=None,state='exited_during_sample')}
        self.assertEqual(observation.classify(synthetic,[pid],[]),['14956']);self.assertFalse(observation.final_clean)
        tail=copy.deepcopy(rows[-1]);tail.update(time=tail['time']+10,nvml_processes={},process_gpu={},owned_host_pids=[],owned_pid_metadata={pid:dict(pid=14330,host_pid=None,state='exited_during_sample')})
        self.assertEqual(observation.classify(tail,[pid],[]),[]);self.assertTrue(observation.final_clean)
        self.assertIn('failure',source['refs'])

    def test_temporary_residual_10_seconds_does_not_grant_early_admission(self):
        v=ExitObservation(sample(0));v.classify(sample(1),[10],[10]);v.classify(sample(2,False),[10],[])
        self.assertEqual(v.classify(sample(12,False),[10],[]),['110']);self.assertFalse(v.final_clean)
        v.classify(sample(13,False,False),[10],[]);self.assertTrue(v.final_clean)

    def test_long_residual_and_query_deadline_reject(self):
        v=ExitObservation(sample(0));v.classify(sample(1),[10],[10]);v.classify(sample(2,False),[10],[])
        self.assertLessEqual(v.query_timeout(55),3.5)
        with self.assertRaises(ValueError):v.classify(sample(63,False),[10],[])
        with self.assertRaises(ValueError):v.query_timeout(63)

    def test_pid_reuse_and_namespace_conflict_reject(self):
        for field,value in (('start_ticks','101'),('host_pid',111),('namespace','pid:[2]')):
            v=ExitObservation(sample(0));v.classify(sample(1),[10],[10]);changed=sample(2);changed['owned_pid_metadata']['10'][field]=value
            with self.assertRaises(ValueError):v.classify(changed,[10],[])

    def test_active_metadata_unknown_uuid_and_foreign_pid_reject(self):
        v=ExitObservation(sample(0))
        for value in (sample(1,False),sample(1,uuid='other'),sample(1,owned_pid_metadata={'11':dict(host_pid=111,start_ticks='1',namespace='pid:[1]')})):
            with self.assertRaises(ValueError):v.classify(value,[10],[10])

    def test_inactive_permission_or_unstable_read_cannot_admit(self):
        v=ExitObservation(sample(0));v.classify(sample(1),[10],[10])
        with self.assertRaises(ValueError):v.classify(sample(2,False,False,owned_pid_metadata={'10':dict(host_pid=None,state='metadata_unavailable')}),[10],[])
        self.assertFalse(v.final_clean)
        with self.assertRaises(ValueError):v.classify(sample(63,False,False),[10],[])

    def test_unknown_external_occupancy_and_uuid_are_not_ownership(self):
        a=sample(1);a['nvml_processes']['999']=100
        self.assertFalse(resource_assessment(a,[10],sample(0,False,False))['admission'])
        v=ExitObservation(sample(0))
        with self.assertRaises(ValueError):v.classify(sample(1,uuid='new'),[10],[])

    def test_gpu_queries_have_finite_timeout_and_failure_propagates(self):
        with patch('tools.restricted_regression.m5_formal_entry.subprocess.check_output',side_effect=subprocess.TimeoutExpired('nvidia-smi',1))as call:
            with self.assertRaises(subprocess.TimeoutExpired):gpu_sample([],query_timeout=1)
            self.assertEqual(call.call_args.kwargs['timeout'],1)

    def test_monitor_query_nonzero_and_timeout_are_not_resource_fallback(self):
        from tools.restricted_regression import m5_formal_entry as tool
        for error in (subprocess.CalledProcessError(1,['nvidia-smi']),subprocess.TimeoutExpired('nvidia-smi',1)):
            with tempfile.TemporaryDirectory(prefix='m6-depth-query-')as d:
                root=Path(d);cfg=dict(purpose='CPU_observation_fixture',output=str(root/'worker'),audit_log=str(root/'audit.jsonl'),limits=dict(seconds=5))
                with patch.object(tool,'read_profiles',return_value=dict(execution=dict(evidence=d))),patch.object(tool,'gpu_sample',side_effect=error),patch.object(tool,'spawn',side_effect=AssertionError('query failure must prevent dispatch')):
                    result=tool.run_configs([cfg],root/'observation',monitor=True)
                self.assertEqual(result['failure_kind'],'observation');self.assertFalse(result['resource_admission']);self.assertEqual(result['returncodes'],[])

    def test_real_CPU_child_instance_then_exit_no_signal(self):
        child=subprocess.Popen([q.PYTHON,'-B','-c','import time;time.sleep(.2)'])
        m=owned_pid_metadata(child.pid);self.assertIsNotNone(m.get('host_pid'))
        value=sample(0,False,False);v=ExitObservation(value)
        value.update(time=1,owned_pid_metadata={str(child.pid):m},nvml_processes={str(m['host_pid']):100})
        v.classify(value,[child.pid],[child.pid]);self.assertEqual(child.wait(timeout=3),0)
        value.update(time=2,owned_pid_metadata={str(child.pid):owned_pid_metadata(child.pid)})
        self.assertEqual(v.classify(value,[child.pid],[]),[str(m['host_pid'])]);self.assertFalse(v.final_clean)
        value.update(time=3,nvml_processes={});v.classify(value,[child.pid],[]);self.assertTrue(v.final_clean)


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cs=q.configs()

    def test_exact48_unique_variants_only_one_structure_field_changes(self):
        ids=[]
        for depth in (1,2):
            c=self.cs['PATCH_ENC'+str(depth)];self.assertEqual(len(c['tasks']),24)
            for t in c['tasks']:
                parent=c['baseline_unified']['parent_refs'][t['id']];old=bound(parent['config_ref']);prior=next(x for x in old['tasks']if x['id']==parent['task_id']);expected=profile(old,prior);expected['structure']['e_layers']=depth
                self.assertEqual(profile(c,t),expected);self.assertEqual(t['variant'],'encoder-'+str(depth));ids.append(t['id'])
                self.assertEqual(expected['training']['scheduler']['name'],'OneCycleLR')
                if t['dataset']=='Weather':self.assertEqual((expected['training']['epochs'],expected['training']['patience']),(20,10))
            self.assertEqual(s.formal_budget(c)['total']['run_epochs'],280)
        self.assertEqual(len(set(ids)),48)

    def test_urban84_six_folds_two_H_original28_unchanged(self):
        c=self.cs['URBAN_SUBSET'];old=bound(r.contract()['old_config_refs']['URBAN_SUBSET'])
        self.assertEqual({(t['model'],t['fold'],t['h'])for t in c['tasks']},{(m,f,h)for m in s.MODELS for f in range(1,7)for h in (3,12)})
        for t in old['tasks']:self.assertEqual(profile(c,t),profile(old,t))
        data=bound(c['baseline_unified']['data_ref'])
        for t in c['tasks']:
            a=step_arithmetic(c,t);m=data['metadata']['UrbanEV'][t['id']]
            self.assertEqual(m['fold'],t['fold']);self.assertEqual(m['window_counts'],dict(train=a['train_windows'],validation=a['validation_windows']))
            self.assertEqual(profile(c,t)['training']['scheduler']['coefficient'],4/9)

    def test_five_frozen_policy_tables_unchanged_and_two_depth_tables_uniform(self):
        old=r.contract()['old_config_refs'];self.assertEqual(sum(len(bound(v)['baseline_unified']['numeric_policies'])for v in old.values()),133)
        for stage in old:self.assertEqual(self.cs[stage]['baseline_unified']['numeric_policies'],bound(old[stage])['baseline_unified']['numeric_policies'])
        self.assertEqual(sum(len(c['baseline_unified']['numeric_policies'])for c in self.cs.values()),145)
        self.assertEqual(profile(self.cs['M_ALL'],next(t for t in self.cs['M_ALL']['tasks']if t['model']=='PatchTST'))['structure']['e_layers'],2)
        historical=bound(old['M_ALL'])
        self.assertEqual(profile(historical,next(t for t in historical['tasks']if t['model']=='PatchTST'))['structure']['e_layers'],3)

    def test_exact_new_task_config_namespace_and_profile_tamper_reject(self):
        for stage in ('PATCH_ENC1','PATCH_ENC2','URBAN_SUBSET'):
            c=copy.deepcopy(self.cs[stage]);c['tasks'][0]['h']=9
            with self.assertRaises(ValueError):r.validate_revision(c)
            c=copy.deepcopy(self.cs[stage]);c['resolved_profiles'][c['tasks'][0]['id']]['training']['batch']=64
            with self.assertRaises(ValueError):r.validate_revision(c)

    def test_third287_remaining335_total531_4550_epochs(self):
        self.assertEqual(sum(len(self.cs[k]['tasks'])for k in ('URBAN_SUBSET','EPF_ALL','M_ALL')),287)
        self.assertEqual(sum(s.formal_budget(self.cs[k])['total']['run_epochs']for k in s.STAGES[2:]),4550)
        self.assertEqual(sum(len(self.cs[k]['tasks'])for k in s.STAGES[2:]),335)
        self.assertEqual(q.completion_counts(),r.completion_counts())

    def test_metadata_binding_includes_new_contract_and_real_parents(self):
        from utils.ch3_type1_execution import metadata_files
        with patch.object(q,'closure',return_value='f'*40):
            values=r.worker_metadata(self.cs['PATCH_ENC1'])
            self.assertIn(r.CONTRACT_REF,values);self.assertIn(r.contract()['old_config_refs']['M_AMEND'],values)
            self.assertNotIn('trajectory.json',' '.join(v['path']for v in values))

    def test_completed_stages_and_old_attempt_permits_rejected(self):
        for stage,c in self.cs.items():
            with self.assertRaises(PermissionError):r.validate_permit_link(c,dict(execution_attempt=r.prefix.ATTEMPT,probe_recovery_ref=r.prefix.REUSE_REF),True)
        for stage in r.COMPLETED_STAGES:
            with self.assertRaises(PermissionError):r.validate_permit_link(self.cs[stage],dict(execution_attempt=r.ATTEMPT,probe_recovery_ref=r.REUSE_REF),True)

    def test_original371_immutable_and_encoder2_choice_is_pre_results(self):
        before=ref(bound(r.SOURCE_REF)['refs']['main']['path']);value=r.depth_results()
        self.assertEqual(before,bound(r.SOURCE_REF)['refs']['main']);self.assertEqual(len(value['rows']),72);self.assertEqual(value['main_table_selection'],'pre_results_fixed_encoder_2')
        self.assertFalse(value['metric_based_reselection']);self.assertTrue(value['enc1_retained_not_main']);self.assertEqual(sum(x['result']['status']=='complete'for x in value['rows']),24)

    def test_third_only24_encoder_changes_type1_and_MS_profiles_unchanged(self):
        old=bound(r.contract()['old_config_refs']['M_ALL']);c=self.cs['M_ALL'];changed=[]
        for prior,t in zip(old['tasks'],c['tasks']):
            expected=profile(old,prior)
            if prior['model']=='PatchTST':expected['structure']['e_layers']=2;changed.append(t['id'])
            else:self.assertEqual(t,prior)
            self.assertEqual(t,r.encoder2_type1_task(prior));self.assertEqual(profile(c,t),expected)
            self.assertEqual(profile(c,t)['training']['scheduler']['name'],'type1_horizon_scaled_v1')
        self.assertEqual(len(changed),24);self.assertEqual(len(set(t['id']for t in c['tasks'])),168)
        for stage in ('URBAN_SUBSET','EPF_ALL'):
            parent=bound(r.contract()['old_config_refs'][stage]);oldpatch=[t for t in parent['tasks']if t['model']=='PatchTST']
            for prior in oldpatch:self.assertEqual(profile(self.cs[stage],prior),profile(parent,prior))
        r.validate_revision(c)


SMOKE=r'''
import json,os,sys
from utils.ch3_native_recovery_records import bound
from utils.ch3_contract import profile,digest
from utils import ch3_patchtst_depth_urban6_recovery as r
c=bound(r.config_refs()['PATCH_ENC'+sys.argv[1]])
def audit(event,values):
 if event=='open'and isinstance(values[0],(str,bytes)):
  p=os.fsdecode(values[0])
  if '/AMD/data/'in p or p.endswith(('.pt','.pth','.safetensors')):raise PermissionError('CPU smoke forbids real data/checkpoint')
sys.addaudithook(audit)
import torch
torch.set_num_threads(4)
def denied(*a,**k):raise AssertionError('CPU smoke forbids backward/Adam/GPU/checkpoint')
torch.Tensor.backward=denied;torch.optim.Adam=denied;torch.cuda.init=denied;torch.load=denied
from models.ch3_adapter import build,target_prediction
bank={};rows=[]
for t in c['tasks']:
 p=profile(c,t);key=digest({k:v for k,v in p.items()if k not in('dataset','training')})
 if key not in bank:
  m=build(c,t);m.eval();blocks=len(m.model.backbone.encoder.layers)
  assert blocks==int(sys.argv[1])
  x=torch.linspace(-1,1,2*p['T']*p['C']).reshape(2,p['T'],p['C'])
  with torch.no_grad():y,_=target_prediction(m,x,p,None)
  assert list(y.shape)==[2,t['h'],p['C']]and torch.isfinite(y).all()
  bank[key]=dict(representative=t['id'],blocks=blocks,shape=list(y.shape))
 rows.append(dict(task=t['id'],profile_sha=digest(p),construction_signature=key,evidence=bank[key]))
assert not torch.cuda.is_initialized()
print(json.dumps(dict(rows=rows,construction=len(bank),forward=len(bank),backward=0,adam=0,GPU=0,test=0,real_validation=0,engineering_batch=2,formal_batch_unchanged=128)))
'''


class RealPatchInterface(unittest.TestCase):
    def test_third_encoder2_reuses_exact_constructed_CPU_signature_not_concurrency(self):
        evidence=bound(ref(r.PACKAGE/'cpu-smoke-reuse.json'));capture=json.loads(bound(evidence['captures']['2'])['stdout'])
        bank={row['construction_signature']:row for row in capture['rows']};c=q.configs()['M_ALL'];count=0
        for t in c['tasks']:
            if t['model']!='PatchTST':continue
            p=profile(c,t);signature=digest({k:v for k,v in p.items()if k not in('dataset','training')})
            self.assertIn(signature,bank);self.assertEqual(bank[signature]['evidence']['blocks'],2)
            self.assertEqual(bank[signature]['evidence']['shape'],[2,t['h'],p['C']]);count+=1
        self.assertEqual(count,24)

    def test_enc1_enc2_actual_adapter_blocks_and_output_shapes(self):
        output=[]
        for depth in (1,2):
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT))
            destination=os.environ.get('M6_DEPTH_SMOKE_EVIDENCE')
            reuse=os.environ.get('M6_DEPTH_REUSE_SMOKE')
            if reuse:
                import hashlib
                evidence=bound(ref(reuse));self.assertEqual(evidence['CPU_worker_sha256'],hashlib.sha256(SMOKE.encode()).hexdigest())
                record=bound(evidence['captures'][str(depth)])
            else:
                result=subprocess.run([q.PYTHON,'-B','-c',SMOKE,str(depth)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
                record=dict(depth=depth,exit=result.returncode,stdout=result.stdout,stderr=result.stderr)
            if destination and not reuse:
                with (Path(destination)/('enc'+str(depth)+'.json')).open('x')as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
            self.assertEqual(record['exit'],0,record['stderr']);value=json.loads(record['stdout']);self.assertEqual(len(value['rows']),24);output.append(value)
            c=q.configs()['PATCH_ENC'+str(depth)]
            self.assertEqual({x['task']:x['profile_sha']for x in value['rows']},{t['id']:digest(profile(c,t))for t in c['tasks']})
            self.assertTrue(all(row['evidence']['blocks']==depth for row in value['rows']))
        self.assertEqual(sum(x['construction']for x in output),32);self.assertEqual(sum(x['forward']for x in output),32)


class PrefixAndReuse(unittest.TestCase):
    def test_real_light_prefix_and_metadata_integrity_no_checkpoint_load(self):
        self.assertEqual(len(r.verify_prefix()['main']['cells']),371)
        self.assertNotIn('torch',sys.modules)

    def test_light_source_active_old_owner_and_wrong_failure_refuse(self):
        from utils import ch3_type1_upstream as u
        with patch.object(u,'assert_owned_exited',side_effect=ValueError('synthetic old live instance')):
            with self.assertRaises(ValueError):r.verify_source_light()
        source=bound(r.SOURCE_REF);source['producer_commit']='0'*40
        original=r.bound
        with patch.object(r,'bound',side_effect=lambda value:source if value==r.SOURCE_REF else original(value)):
            with self.assertRaises(ValueError):r.verify_source_light()

    def test_only_valid_H3_is_seeded_H12_failure_cost_preserved(self):
        c=q.configs()['URBAN_SUBSET'];a=dict(execution_attempt=r.ATTEMPT,probe_recovery_ref=r.REUSE_REF,round2_boundary_ref=dict(path=str(r.main_boundary_path())))
        seed=r.load_seed(c,a);self.assertEqual(len(seed['evidence']),1);self.assertEqual(len(seed['decisions']),0)
        self.assertEqual(seed['budget']['historical_actual'],dict(adam=12,backward=12,forward=16))
        self.assertEqual(seed['budget']['diagnostic_actual'],dict(adam=6,backward=6,forward=8))
        run=next(iter(seed['evidence'].values()))['task_ids'][0];self.assertIn('-h3-',run)
        report=dict(scope=s.context(c)['probe_scope'],probe_recovery_ref=r.REUSE_REF,evidence=seed['evidence'])
        key=next(iter(seed['evidence']));self.assertEqual(r.producer(report,key,{} )['commit'],r.PRODUCER)
        self.assertEqual(r.location(report,key,Path('/tmp/future')).parent.name,'serial')
        self.assertEqual(r.extra_actual(report),dict(adam=6,backward=6,forward=8))
        self.assertFalse(any('-h12-'in x for v in seed['evidence'].values()for x in v['task_ids']))

    def test_adoption_new_lifecycle_retains_old_training_commit_and_index(self):
        from utils import ch3_round2_amendment as amend
        with tempfile.TemporaryDirectory(prefix='m6-depth-prefix-')as d,contextlib.ExitStack()as stack:
            root=Path(d);stack.enter_context(patch.object(s,'RESULT',root));stack.enter_context(patch.object(q,'CONTROL',root/'queue/controller'));stack.enter_context(patch.object(amend,'RESULT',root/'round2-amendment'))
            exclusive(q.CONTROL/'controller.json',dict(owner=q.owner(),scope=s.ID))
            stack.enter_context(patch.dict(os.environ,{q.SECRET:'synthetic-only'}));stack.enter_context(patch.object(q,'closure',return_value='f'*40));stack.enter_context(patch.object(q,'stop_check'))
            stack.enter_context(patch.object(r,'verify_production_inheritance',return_value={}))
            # Actual source verification/adoption/MAC and boundary validators run.
            adopted=r.adopt_prefix();m=bound(adopted['SEAL_M128_BOUNDARY']);a=bound(adopted['SEAL_AMEND_BOUNDARY'])
            self.assertEqual(m['commit'],'760b9dd7162d200c11b8836a7c9ece41822dfcf1');self.assertEqual(a['commit'],r.PRODUCER)
            self.assertEqual(m['adoption_commit'],'f'*40);self.assertEqual(a['adoption_commit'],'f'*40)
            q.validate_boundary_light(adopted['SEAL_AMEND_BOUNDARY'],'M_AMEND');b=q.validate_round2_boundary_light(adopted['SEAL_ROUND2_REVISED_BOUNDARY'])
            self.assertEqual(b['main_index_ref'],bound(r.SOURCE_REF)['refs']['main']);self.assertFalse((root/'M_BASE').exists());self.assertFalse((root/'probe').exists())
            with patch.dict(os.environ,{q.SECRET:'wrong'}),self.assertRaises(PermissionError):q.validate_boundary_light(adopted['SEAL_AMEND_BOUNDARY'],'M_AMEND')

    def test_actual_chain_skips_completed_stages_and_has_one_audit_per_new_stage(self):
        cs=q.configs();calls=[];initial={k:dict(path='/tmp/synthetic-'+k,sha256='0'*64)for k in ('VERIFY_IMPORT_MS203_AND_SEAL','SEAL_M128_BOUNDARY','SEAL_BASE_287_BOUNDARY','SEAL_AMEND_BOUNDARY','SEAL_ROUND2_REVISED_BOUNDARY')}
        with tempfile.TemporaryDirectory(prefix='m6-depth-flow-')as d,contextlib.ExitStack()as stack:
            control=Path(d);stack.enter_context(patch.object(s,'RESULT',control/'fixture-output'));stack.enter_context(patch.object(q,'CONTROL',control));stack.enter_context(patch.object(q,'stop_check'));stack.enter_context(patch.object(q,'validate_start'));stack.enter_context(patch.object(q,'bound',side_effect=lambda v:dict(boundaries=dict(MS='retained-MS'))if v==initial['VERIFY_IMPORT_MS203_AND_SEAL']else{}))
            stack.enter_context(patch.object(r,'adopt_prefix',return_value=initial))
            def permit(c,*args,**kwargs):
                stage=c['baseline_unified']['stage'];source=kwargs.get('round2_ref',args[4]if len(args)>4 else None)
                if stage in ('URBAN_SUBSET','EPF_ALL','M_ALL'):self.assertEqual(source,dict(path='/tmp/fixed-enc2',sha256='0'*64))
                return dict(stage=stage)
            stack.enter_context(patch.object(q,'create_permit',side_effect=permit))
            stack.enter_context(patch.object(q,'seal_runtime',return_value={}))
            stack.enter_context(patch.object(q,'wait_owned',side_effect=lambda stage,permit,probe,model=None,runtime=None:calls.append(('probe'if probe else'formal',stage,model))))
            audit=stack.enter_context(patch.object(q,'audit_probe',side_effect=lambda c:dict(stage=c['baseline_unified']['stage'])))
            stack.enter_context(patch.object(q,'ref',side_effect=lambda p:dict(path=str(p),sha256='0'*64)))
            stack.enter_context(patch.object(q,'seal_boundary',side_effect=lambda c,rs:dict(stage=c['baseline_unified']['stage'])))
            stack.enter_context(patch.object(r,'seal_fixed_main',side_effect=lambda rs:(calls.append(('fixed371','PATCH_ENC2',None))or dict(path='/tmp/fixed-enc2',sha256='0'*64))))
            q.run({});complete=json.loads((control/'complete.json').read_text())
            stages=['PATCH_ENC1','PATCH_ENC2','URBAN_SUBSET','EPF_ALL','M_ALL']
            self.assertEqual([stage for kind,stage,_ in calls if kind=='probe'],stages);self.assertEqual(audit.call_count,5)
            self.assertEqual(sum(len([t for t in cs[stage]['tasks']if t['model']==model])for kind,stage,model in calls if kind=='formal'),335)
            self.assertEqual(complete['total_runs'],531);self.assertEqual(complete['adopted_new_formal_runs'],196)
            self.assertEqual(complete['chosen_PatchTST_M_encoder'],2)
            self.assertEqual(complete['round2_fixed_enc2_boundary']['path'],'/tmp/fixed-enc2')
            self.assertLess(calls.index(('fixed371','PATCH_ENC2',None)),calls.index(('probe','URBAN_SUBSET',None)))
            self.assertFalse(any(stage in r.COMPLETED_STAGES for _,stage,_ in calls))

    def test_drive_failure_stops_later_depth_and_third_round(self):
        seen=[]
        def failed(_):raise RuntimeError('synthetic numeric failure, not resource')
        with self.assertRaises(RuntimeError):q.drive({'enc1':failed,'enc2':lambda _:seen.append('enc2')},lambda x:seen.append(x),lambda:None,states=['enc1','enc2'])
        self.assertEqual(seen,['enc1'])

    def test_bundle_and_exact_producer_delta_pure_import(self):
        from tools.restricted_regression.run_restricted import verify_bundle
        self.assertEqual(verify_bundle(),r.verify_production_inheritance()['new_bundle_sha']);self.assertNotIn('torch',sys.modules)

    def test_public_preflight_light_checks_once_remote_and_creates_no_boundary(self):
        from utils import ch3_round2_amendment as amend
        with tempfile.TemporaryDirectory(prefix='m6-depth-preflight-')as d,contextlib.ExitStack()as stack:
            root=Path(d);stack.enter_context(patch.object(s,'RESULT',root/'future'));stack.enter_context(patch.object(amend,'RESULT',root/'future/round2-amendment'));stack.enter_context(patch.object(q,'CONTROL',root/'future/controller'))
            stack.enter_context(patch.object(q,'closure',return_value='f'*40));stack.enter_context(patch.object(q,'dynamic',return_value={}))
            remote=stack.enter_context(patch.object(q,'verify_live_remote',return_value='f'*40))
            value=q.start_template();value.update(reviewed=True,execution_permitted=True,structure_frozen=True,m6_authorized=True,budget_authorized=True,closure_commit='f'*40,authorization_basis='temporary CPU-only preflight fixture')
            result=q.readiness_report(value);self.assertEqual(result['blocked'],[]);self.assertTrue(result['READY_TO_ARM_HANDOFF']);self.assertFalse(result['READY_FOR_GPU_EXECUTION']);self.assertEqual(result['remaining_formal_runs'],335);self.assertEqual(remote.call_count,1)
            self.assertFalse((root/'future').exists());self.assertNotIn('torch',sys.modules)
            with patch('tools.restricted_regression.run_restricted.verify_bundle',side_effect=RuntimeError('synthetic seal mismatch')):
                result=q.readiness_report(value);self.assertTrue(any('bundle verification failed'in x for x in result['blocked']));self.assertFalse(result['READY_TO_ARM_HANDOFF'])
            self.assertFalse((root/'future').exists())

    def test_old_namespace_logs_and_authorization_never_used_as_new(self):
        self.assertNotEqual(r.RESULT,r.OLD_RESULT);self.assertNotEqual(r.SESSION,r.prefix.SESSION);self.assertNotEqual(r.LOG,r.prefix.LOG)
        template=q.start_template();self.assertEqual(template['upstream_anchors']['execution_attempt'],r.ATTEMPT)
        with patch.object(q,'closure',return_value='f'*40),self.assertRaises(PermissionError):q.validate_start(bound(bound(r.SOURCE_REF)['refs']['authorization']))


class ActualProbeAudit(unittest.TestCase):
    def test_all_new_stage_seals_keep_one_protocol_digest_and_full_binding(self):
        cs=q.configs()
        with tempfile.TemporaryDirectory(prefix='m6-depth-seals-')as d:
            for stage in ('PATCH_ENC1','PATCH_ENC2','URBAN_SUBSET','EPF_ALL','M_ALL'):
                c=cs[stage];ctx=dict(s.context(c),control=Path(d)/stage);receipts={}
                for model in ctx['models']:
                    receipts[model]=exclusive(Path(d)/(stage+'-'+model+'.json'),dict(technical_complete=True,model=model,task_ids=[t['id']for t in c['tasks']if t['model']==model]))
                binding=dict(protocol_sha=digest(c),commit='f'*40,code={},environment={},hardware={},source_states={})
                with patch.object(s,'context',return_value=ctx),patch.object(q,'stop_check'),patch.object(q,'dynamic',return_value=binding)as call:
                    value=bound(q.seal_boundary(c,receipts));self.assertEqual(call.call_count,1)
                    for key,expected in binding.items():self.assertEqual(value[key],expected)
                    with self.assertRaises(ValueError):q.seal_boundary(c,{})
                with patch.object(s,'context',return_value=ctx),patch.object(q,'stop_check'),patch.object(q,'dynamic',return_value=dict(binding,protocol_sha='0'*64)),self.assertRaises(ValueError):q.seal_boundary(c,receipts)

    def test_enc1_real_compare_completion_audit_manifest_and_compact_summary(self):
        from tools.restricted_regression import m5_formal_entry as tool,resource_budget
        from utils import ch3_native_execution as native,ch3_round2_amendment as amend,ch3_type1_execution as execution
        from tests.test_m6_probe_schema_recovery import trajectory
        from utils.ch3_contract import task_by_id,numeric_probe_policy
        c=q.configs()['PATCH_ENC1'];events=[];counter=[10000000]
        with tempfile.TemporaryDirectory(prefix='m6-depth-probe-')as d,contextlib.ExitStack()as stack:
            base=Path(d);ctx=dict(s.context(c),control=base/'queue',probe_root=base/'probe',fixture=base/'fixture',result_root=base/'formal')
            for name in ('control','probe_root','fixture'):ctx[name].mkdir()
            old_context=s.context
            stack.enter_context(patch.object(s,'RESULT',base/'adopted'));stack.enter_context(patch.object(amend,'RESULT',base/'adopted/round2-amendment'))
            stack.enter_context(patch.object(s,'context',side_effect=lambda cc:ctx if cc==c else old_context(cc)));stack.enter_context(patch.object(q,'CONTROL',base/'controller'));stack.enter_context(patch.dict(os.environ,{q.SECRET:'CPU-synthetic-only'}));stack.enter_context(patch.dict(sys.modules,{'m5_formal_entry':tool,'resource_budget':resource_budget}))
            exclusive(q.CONTROL/'controller.json',dict(scope=s.ID,owner=q.owner()))
            stack.enter_context(patch.object(q,'closure',return_value='f'*40));stack.enter_context(patch.object(r,'verify_production_inheritance',return_value={}))
            stack.enter_context(patch.object(q,'dynamic',side_effect=lambda cc,worker=False:dict(commit='f'*40,protocol_sha=digest(cc),code={},environment={},hardware={'cpu_affinity':[]},source_states={})))
            stack.enter_context(patch('utils.ch3_native_recovery.resource_check'));stack.enter_context(patch('utils.ch3_m_execution.gpu_environment_reasons',return_value=[]))
            remote=stack.enter_context(patch.object(q,'verify_live_remote',side_effect=AssertionError('runtime cannot query remote')))
            adopted=r.adopt_prefix();predecessors=dict(M_BASE=adopted['SEAL_M128_BOUNDARY'],M_AMEND=adopted['SEAL_AMEND_BOUNDARY'])
            auth=q.start_template();auth.update(reviewed=True,execution_permitted=True,structure_frozen=True,m6_authorized=True,budget_authorized=True,closure_commit='f'*40,authorization_basis='CPU synthetic fixture only, cannot authorize real HEAD')
            start=exclusive(base/'synthetic-start.json',auth);pr=q.create_permit(c,start,True,boundary_ref=predecessors,round2_ref=adopted['SEAL_ROUND2_REVISED_BOUNDARY']);a=bound(pr)
            exclusive(ctx['probe_root']/'approval.json',a)
            def make(cc,purpose,out,*,task,approval):
                t=task_by_id(c,task);out=Path(out);counts=s.worker_counts(c,t);counter[0]+=1;pid=counter[0]
                cfg=dict(task=task,output=str(out),approval=approval,protocol_sha=digest(c),successor_scope=ctx['probe_scope'],successor_phase=out.parent.name,prefix_files={},limits=dict(**counts,seconds=1800))
                exclusive(out/'trajectory.json',trajectory(c,t,out,numeric_probe_policy(c,t)));exclusive(out/'config.json',cfg);exclusive(out/'budget.json',dict(counts=counts,by_pid={str(pid):counts}));exclusive(out/'runtime.json',dict(pid=pid,task=task,error=None));(out/'audit.jsonl').write_text('{"event":"CPU synthetic payload"}\n')
                return dict(cfg,pid=pid,budget_file=str(out/'budget.json'))
            def launch(configs,out,monitor):
                out=Path(out);out.mkdir();events.extend(x['task']for x in configs);peaks={str(x['pid']):1 for x in configs}
                value=dict(failure=None,returncodes=[0]*len(configs),resource_admission=True,elapsed=2. if out.parent.name=='serial'else 1.,exit_transitions_resolved=True,process_attribution='Measured',process_peaks=peaks,cpu_peaks=peaks)
                exclusive(out/'process.json',value);(out/'memory.jsonl').write_text(json.dumps(dict(owned_pid_metadata={pid:dict(pid=int(pid),start_ticks='1')for pid in peaks}))+'\n');return value
            stack.enter_context(patch.object(tool,'make_config',side_effect=make));stack.enter_context(patch.object(tool,'run_configs',side_effect=launch))
            compare=stack.enter_context(patch.object(native,'compare',wraps=native.compare));audit=stack.enter_context(patch.object(native,'validate_probe_completion',wraps=native.validate_probe_completion))
            report=native.run_probe(c,a);self.assertEqual(len(events),48);self.assertEqual(audit.call_count,0)
            admission=q.audit_probe(c);self.assertEqual(audit.call_count,1);summary=q.validate_summary_light(c,admission)
            self.assertEqual(len(summary['decisions']),6);self.assertTrue(summary['technical_admission']);self.assertGreater(compare.call_count,48)
            from utils.ch3_native_recovery_records import scan_manifest
            manifest=bound(summary['manifest_ref']);self.assertTrue(scan_manifest(manifest,summary['complete_ref'],ctx['probe_root'])['integrity_scan_passed'])
            formal=q.create_permit(c,start,False,admission,predecessors,adopted['SEAL_ROUND2_REVISED_BOUNDARY']);runtime=q.seal_runtime(c,formal)
            q.validate_runtime(c,runtime,formal)
            t=c['tasks'][0];out=ctx['result_root']/'formal-PatchTST'/t['id']
            with patch.dict(os.environ,{'TMPDIR':str(ctx['fixture'])}):
                cfg=execution.make_config(c,'ch3_formal',out,task=t['id'],approval=bound(formal),runtime_ref=runtime);execution.validate_worker(c,cfg)
            self.assertEqual(audit.call_count,1);remote.assert_not_called()
            with self.assertRaises((PermissionError,ValueError)):q.validate_permit(q.configs()['PATCH_ENC2'],bound(formal))
            bad=copy.deepcopy(report);bad['decisions'].pop(next(iter(bad['decisions'])))
            with self.assertRaises(ValueError):native.validate_probe_completion(c,bad)
            self.assertNotIn('torch',sys.modules)


@contextlib.contextmanager
def fixed_main_fixture():
    """Real adoption/seal/index validators, exclusively under a temporary root."""
    from utils import ch3_round2_amendment as amend
    with tempfile.TemporaryDirectory(prefix='m6-fixed-enc2-')as d,contextlib.ExitStack()as stack:
        root=Path(d)
        for module,name,value in ((r,'RESULT',root),(s,'RESULT',root),(q,'CONTROL',root/'queue/controller'),(amend,'RESULT',root/'round2-amendment')):
            stack.enter_context(patch.object(module,name,value))
        stack.enter_context(patch.dict(os.environ,{q.SECRET:'CPU-synthetic-only'}))
        stack.enter_context(patch.object(q,'closure',return_value='f'*40))
        stack.enter_context(patch.object(q,'dynamic',side_effect=lambda c,worker=False:dict(commit='f'*40,protocol_sha=digest(c),code={},environment={},hardware={},source_states={})))
        exclusive(q.CONTROL/'controller.json',dict(scope=s.ID,owner=q.owner()))
        adopted=r.adopt_prefix();c=q.configs()['PATCH_ENC2'];artifacts={}
        for i,t in enumerate(c['tasks']):
            p=profile(c,t);path=root/'PATCH_ENC2'/'formal-PatchTST'/t['id'];best=dict(path=str(path/'best.pt'),sha256='a'*64)
            # Intentionally worse than enc3, and unlike any enc1 outcome: no ranking.
            result=dict(id=t['id'],task='M',input_variant='M',metric_scope='all_channels',profile_sha=digest(p),protocol_sha=digest(c),scientific_protocol=c['baseline_unified']['id'],commit='f'*40,seed=2024,from_scratch=True,scheduler_sha=digest(p['training']['scheduler']),best_epoch=1,final_test=dict(calls=1,selected='best.pt',sha256=best['sha256'],epoch=1),mse=1000.+i,mae=500.+i)
            manifest=dict(task=t,profile=p,identity=dict(run_id=t['id'],task='M',metric_scope='all_channels',profile_sha=digest(p),protocol_sha=digest(c),scientific_protocol=c['baseline_unified']['id'],commit='f'*40,from_scratch=True))
            artifacts[t['id']]={'result.json':exclusive(path/'result.json',result),'manifest.json':exclusive(path/'manifest.json',manifest),'runtime.json':exclusive(path/'runtime.json',dict(task=t['id'],error=None)),'best.pt':best}
        group=exclusive(root/'PATCH_ENC2/group.json',dict(technical_complete=True,model='PatchTST',task_ids=[t['id']for t in c['tasks']],artifacts=artifacts))
        sealed=q.seal_boundary(c,dict(PatchTST=group));adopted[q.STAGE_STATES['PATCH_ENC2'][3]]=sealed
        yield root,adopted,c,artifacts


class FixedEncoderAdoption(unittest.TestCase):
    def test_real_new371_seal_24_fixed_even_when_worse_347_unchanged_and_old_sources_retained(self):
        before=ref(bound(r.SOURCE_REF)['refs']['main']['path'])
        with fixed_main_fixture()as(root,receipts,c,artifacts):
            original=r.original_second_round();index=r.build_fixed_main_index(receipts[q.STAGE_STATES['PATCH_ENC2'][3]])
            self.assertEqual(len(index['cells']),371);self.assertEqual(len(index['adopted_cells']),24)
            self.assertEqual(index['retained_cells'],347);self.assertFalse(index['metric_based_reselection'])
            for old,new in zip(original['cells'],index['cells']):
                if old['model']=='PatchTST'and old['task']=='M':
                    self.assertEqual(new['encoder'],2);self.assertEqual(new['supersedes'],old['result_ref'])
                    self.assertEqual(index['adopted_cells'][old['cell_id']]['previous_cell'],old)
                    self.assertGreater(new['mse'],old['mse'])
                else:self.assertEqual(new,old)
            sealed=r.seal_fixed_main(receipts);r.validate_main_boundary(sealed)
            r.validate_fixed_main_index(bound(bound(sealed)['main_index_ref']))
            self.assertEqual(r.second_round(),index)
            oldrows=[x for x in r.depth_results()['rows']if x['encoder']==3]
            self.assertEqual({(x['dataset'],x['H']):x['result']['result_ref']for x in oldrows},{(x['dataset'],x['H']):x['result_ref']for x in original['cells']if x['model']=='PatchTST'and x['task']=='M'})
            receipts[r.MAIN_ADOPTION_STATE]=sealed
            complete=dict(round2_boundary=receipts['SEAL_ROUND2_REVISED_BOUNDARY'],PATCH_ENC2_boundary=receipts[q.STAGE_STATES['PATCH_ENC2'][3]],**r.completion_fields(receipts))
            self.assertEqual(r.validate_complete(complete),index)
            # Existing exclusive writer accepts identical bytes, never overwrites.
            self.assertEqual(r.seal_fixed_main(receipts),sealed)
            wrong=copy.deepcopy(index);wrong['chosen_PatchTST_M_encoder']=1
            with self.assertRaises(FileExistsError):exclusive(r.main_index_path(),wrong)
            self.assertNotIn('torch',sys.modules)
        self.assertEqual(ref(before['path']),before)

    def test_adoption_refuses_missing_or_mixed_enc1_cells_changed_347_and_outcome_selection(self):
        with fixed_main_fixture()as(_,receipts,c,artifacts):
            ref2=receipts[q.STAGE_STATES['PATCH_ENC2'][3]];correct=r.build_fixed_main_index(ref2)
            for mutate in ('remove','metric_select','unchanged347','chosen_layer','mixed_enc1'):
                bad=copy.deepcopy(correct)
                if mutate=='remove':bad['cells'].pop()
                elif mutate=='metric_select':bad['metric_based_reselection']=True
                elif mutate=='chosen_layer':bad['chosen_PatchTST_M_encoder']=1
                elif mutate=='mixed_enc1':next(x for x in bad['cells']if x.get('encoder')==2)['encoder']=1
                else:next(x for x in bad['cells']if x['model']!='PatchTST')['execution_commit']='e'*40
                with self.assertRaises(ValueError,msg=mutate):r.validate_fixed_main_index(bad)
            pending=copy.deepcopy(bound(ref2));pending['technical_complete']=False
            invalid=exclusive(Path(ref2['path']).with_name('incomplete.json'),pending)
            with self.assertRaises(ValueError):r.build_fixed_main_index(invalid)

    def test_source_tamper_duplicate_missing_wrong_producer_and_test_once_rejected(self):
        with fixed_main_fixture()as(_,receipts,c,artifacts):
            sealed=receipts[q.STAGE_STATES['PATCH_ENC2'][3]];boundary=bound(sealed);group_ref=boundary['receipts']['PatchTST'];group=bound(group_ref)
            boundary_path=Path(sealed['path']);boundary_bytes=boundary_path.read_bytes()
            # Changed bytes are rejected before interpretation.
            first=c['tasks'][0];file=artifacts[first['id']]['result.json'];path=Path(file['path']);saved=path.read_bytes();path.write_bytes(saved+b' ')
            with self.assertRaises(ValueError):r.build_fixed_main_index(sealed)
            path.write_bytes(saved)
            for badkey,value in (('commit','e'*40),('profile_sha','0'*64),('from_scratch',False),('final_test',dict(calls=2,selected='best.pt',sha256='a'*64,epoch=1))):
                result=bound(file);result[badkey]=value;path.write_text(json.dumps(result)+'\n')
                invalid_group=copy.deepcopy(group);invalid_group['artifacts'][first['id']]['result.json']=ref(path)
                altered_group=exclusive(Path(group_ref['path']).with_name('bad-'+badkey+'.json'),invalid_group)
                invalid_boundary=copy.deepcopy(boundary);invalid_boundary['receipts']['PatchTST']=altered_group
                boundary_path.write_text(json.dumps(invalid_boundary)+'\n')
                with self.assertRaises(ValueError):r.build_fixed_main_index(ref(boundary_path))
                path.write_bytes(saved);boundary_path.write_bytes(boundary_bytes)
            for change in ('duplicate','missing'):
                invalid_group=copy.deepcopy(group)
                if change=='duplicate':invalid_group['task_ids'].append(first['id'])
                else:invalid_group['artifacts'].pop(first['id'])
                altered=exclusive(Path(group_ref['path']).with_name('bad-'+change+'.json'),invalid_group)
                b=copy.deepcopy(boundary);b['receipts']['PatchTST']=altered;boundary_path.write_text(json.dumps(b)+'\n')
                with self.assertRaises(ValueError):r.build_fixed_main_index(ref(boundary_path))
                boundary_path.write_bytes(boundary_bytes)

    def test_third_permit_requires_fixed_boundary_and_current_mac_not_original371(self):
        with fixed_main_fixture()as(_,receipts,c,artifacts):
            sealed=r.seal_fixed_main(receipts);third=q.configs()['M_ALL']
            a=dict(execution_attempt=r.ATTEMPT,probe_recovery_ref=r.REUSE_REF,round2_boundary_ref=receipts['SEAL_ROUND2_REVISED_BOUNDARY'])
            with self.assertRaises(PermissionError):r.validate_permit_link(third,a,True)
            a['round2_boundary_ref']=sealed;r.validate_permit_link(third,a,True)
            with patch.dict(os.environ,{q.SECRET:'different-lifecycle'}),self.assertRaises(PermissionError):q.validate_round2_boundary_light(sealed)
            original=bound(bound(sealed)['main_index_ref']);index_path=r.main_index_path();index_path.write_text(json.dumps(dict(original,chosen_PatchTST_M_encoder=1))+'\n')
            with self.assertRaises(ValueError):q.validate_round2_boundary_light(sealed)

    def test_new_template_requires_new_third_config_plan_choice_and_exact_closure(self):
        a=q.start_template();old=bound(ref(r.PACKAGE/'start-approval.template.json'))
        self.assertNotEqual(a['config_refs']['M_ALL'],old['config_refs']['M_ALL'])
        self.assertEqual(a['config_refs']['EPF_ALL'],old['config_refs']['EPF_ALL'])
        self.assertIn('fixed_encoder_policy_ref',a['upstream_anchors'])
        with patch.object(q,'closure',return_value='f'*40):
            approved=dict(a,reviewed=True,execution_permitted=True,structure_frozen=True,m6_authorized=True,budget_authorized=True,closure_commit='f'*40,authorization_basis='temporary CPU fixture only')
            q.validate_start(approved)
            for change in ('closure','config','plan','policy'):
                bad=copy.deepcopy(approved)
                if change=='closure':bad['closure_commit']='e'*40
                elif change=='config':bad['config_refs']['M_ALL']=old['config_refs']['M_ALL']
                elif change=='plan':bad['plan_refs']['M_ALL']=old['plan_refs']['M_ALL']
                else:bad['upstream_anchors']['fixed_encoder_policy_ref']['sha256']='0'*64
                with self.assertRaises(PermissionError):q.validate_start(bad)


if __name__=='__main__':unittest.main()
