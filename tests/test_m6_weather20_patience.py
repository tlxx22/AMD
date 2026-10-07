"""Weather's explicit pre-start revision; synthetic events, no business models."""
import copy
import json
import math
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from utils import ch3_amend_ett_identity_recovery as r
from utils import ch3_moderntcn_etth1_recovery as numeric
from utils import ch3_type1_chain as q, ch3_type1_tasks as s
from utils import ch3_round2_amendment as amendment
from utils.ch3_contract import BestState, digest, profile, step_arithmetic
from utils.ch3_native_recovery_records import bound, exclusive, ref, sha
from tests.test_m6_amend_ett_identity_recovery import attempt


class Configuration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with attempt(): cls.cs = q.configs()

    def test_exact_56_fields_and_no_other_science_changes(self):
        count = 0
        for stage, c in self.cs.items():
            old = bound(numeric.CONFIG_REFS[stage])
            if stage not in r.WEATHER_CONFIG_REFS:
                self.assertEqual(c, old)
                continue
            projected = copy.deepcopy(c)
            self.assertEqual(projected['baseline_unified'].pop('weather20_patience_ref'), r.WEATHER_REF)
            changes = []
            for t in c['tasks']:
                p = profile(c,t); prior = profile(old,t)
                if t['dataset'] == 'Weather' and prior['training']['epochs'] == 20:
                    self.assertIsNone(prior['training']['patience'])
                    self.assertEqual(p['training']['patience'],10)
                    projected['resolved_profiles'][t['id']]['training']['patience'] = None
                    changes.append((t['model'],t['h'])); count += 1
                else:
                    self.assertEqual(p,prior)
            self.assertEqual(set(changes),{(m,h) for m in s.MODELS for h in (96,192,336,720)})
            self.assertEqual(projected,old)
            self.assertEqual(s.formal_budget(c),s.formal_budget(old))
            self.assertEqual(s.probe_budget(c),s.probe_budget(old))
        self.assertEqual(count,56)

    def test_generators_match_current_and_historical_configs(self):
        for stage in r.WEATHER_CONFIG_REFS:
            c=self.cs[stage]; old=bound(numeric.CONFIG_REFS[stage])
            generator=amendment.inherited if stage=='M_AMEND' else s.inherited_profile
            for prior,t in zip(s.selected(stage),c['tasks']):
                self.assertEqual(generator(prior,c),profile(c,t))
                self.assertEqual(generator(prior),profile(old,t))
            self.assertEqual(s.validate(old),old)
            self.assertEqual(r.revise_weather_config(old),c)

    def test_133_policies_tasks_metadata_and_caps_unchanged(self):
        self.assertEqual(sum(len(c['tasks']) for c in self.cs.values()),427)
        self.assertEqual(sum(len(c['baseline_unified']['numeric_policies']) for c in self.cs.values()),133)
        for stage,c in self.cs.items():
            old=bound(numeric.CONFIG_REFS[stage])
            for key in ('numeric_policies','parent_refs','data_ref'):
                self.assertEqual(c['baseline_unified'][key],old['baseline_unified'][key])
            self.assertEqual(c['tasks'],old['tasks'])
            for key in ('datasets','sources','urban_folds','urban_input_variants'):
                self.assertEqual(c[key],old[key])
        for stage in ('M_BASE','M_AMEND','M_ALL'):
            policy=self.cs[stage]['baseline_unified']['numeric_policies']['TimeMixer-Weather']
            self.assertEqual((policy['loss_atol'],policy['loss_rtol']),(1e-6,1e-5))
        self.assertTrue(all(profile(self.cs['M_BASE'],t)['training']['patience'] is None for t in self.cs['M_BASE']['tasks'] if t['dataset']=='Weather'))

    def test_mutated_revision_science_and_false_marker_rejected(self):
        for field in ('patience','epochs','batch','lr'):
            bad=copy.deepcopy(self.cs['M_AMEND']); t=next(t for t in bad['tasks'] if t['dataset']=='Weather')
            bad['resolved_profiles'][t['id']]['training'][field]=11
            with self.assertRaises(ValueError):s.validate(bad)
        for stage in ('M_AMEND','M_ALL'):
            bad=copy.deepcopy(self.cs[stage]);bad['baseline_unified']['weather20_patience_ref']['sha256']='0'*64
            with self.assertRaises(ValueError):s.validate(bad)
        bad=copy.deepcopy(self.cs['M_BASE']);bad['baseline_unified']['weather20_patience_ref']=r.WEATHER_REF
        with self.assertRaises(ValueError):numeric.validate_revision(bad)

    def test_old_start_cannot_authorize_new_configuration(self):
        old=bound(ref(r.PACKAGE/'start-review.json'))
        with attempt(),patch.object(q,'closure',return_value=old['closure_commit']):
            with self.assertRaisesRegex(PermissionError,'exact unified authorization'):q.validate_start(old)
            new=q.start_template();new.update({k:old[k] for k in ('reviewed','execution_permitted','structure_frozen','m6_authorized','budget_authorized','closure_commit','authorization_basis')})
            self.assertEqual(q.validate_start(new),new)
            new['closure_commit']=r.PRODUCER
            with self.assertRaises(PermissionError):q.validate_start(new)

    def test_old_completed_source_refs_stay_frozen(self):
        source=bound(r.SOURCE_REF); prefix=bound(r.REUSE_REF)
        self.assertEqual(source['config_refs'],numeric.CONFIG_REFS)
        self.assertEqual(source['producer_commit'],r.PRODUCER)
        self.assertEqual(prefix['producer_commit'],r.PRODUCER)
        self.assertEqual(prefix['counts'],dict(MS=203,M=84,total=287))
        self.assertEqual(r.anchors()['candidate_config_refs'],r.config_refs())
        self.assertNotEqual(r.config_refs()['M_AMEND'],numeric.CONFIG_REFS['M_AMEND'])

    def test_old_config_cannot_be_used_in_current_worker_permit(self):
        a=dict(execution_attempt=r.ATTEMPT,probe_recovery_ref=r.REUSE_REF)
        for stage in ('M_AMEND','M_ALL'):
            r.validate_permit_link(self.cs[stage],a,True)
            with self.assertRaises(PermissionError):r.validate_permit_link(bound(numeric.CONFIG_REFS[stage]),a,True)
        with self.assertRaises(PermissionError):r.validate_permit_link(self.cs['M_BASE'],a,True)

    def test_compact_worker_metadata_binds_both_new_configs(self):
        from utils.ch3_type1_execution import metadata_files
        with attempt():
            a=dict(start_authorization_ref=ref(r.WEATHER_PACKAGE/'start-approval.template.json'),probe_recovery_ref=r.REUSE_REF)
            values=metadata_files(self.cs['M_ALL'],a)
            for value in [r.WEATHER_REF,*r.WEATHER_CONFIG_REFS.values()]:
                self.assertEqual(values[value['path']],value['sha256'])
            self.assertFalse(any('/serial/' in p or '/q4/' in p or p.endswith('.bin') for p in values))


class SyntheticModel:
    """Plain Python state container, deliberately not a torch Module."""
    def state_dict(self): return dict(synthetic_value=1.)
    def load_state_dict(self,state,strict=True):
        if state != self.state_dict(): raise ValueError('synthetic model identity')


@contextmanager
def formal_fixture(stage,values,interrupt=None):
    # The real scheduler wrapper and checkpoint functions consume synthetic CPU
    # state. No model, Adam, backward, data reader or real evaluation is invoked.
    import torch
    import ch3_runner as runner
    from utils import ch3_data as data, ch3_time_marks as marks, ch3_type1_execution as execution
    class Events(torch.optim.Optimizer):
        def __init__(self):super().__init__([torch.zeros(0)],dict(lr=.01,betas=(.9,.999)))
        def step(self,closure=None):return None
    with tempfile.TemporaryDirectory(prefix='m6-weather-patience-')as tmp,attempt(),ExitStack()as stack:
        c=copy.deepcopy(q.configs()[stage]); t=next(t for t in c['tasks'] if t['model']=='AMD' and t['dataset']=='Weather' and t['h']==96)
        c['tasks']=[t];c['resolved_profiles']={t['id']:profile(c,t)}
        p=profile(c,t);a=step_arithmetic(c,t);model='AMD';root=Path(tmp)/stage;out=root/('formal-'+model)/t['id'];out.mkdir(parents=True)
        ctx=dict(s.context(c),result_root=root)
        original_context=s.context
        stack.enter_context(patch.object(s,'context',side_effect=lambda value:ctx if value is c else original_context(value)))
        for target,name in ((torch.nn.Module,'__init__'),(torch.cuda,'_lazy_init'),(torch.optim.Adam,'__init__')):
            stack.enter_context(patch.object(target,name,side_effect=AssertionError('business model/GPU/Adam forbidden')))
        metadata=dict(synthetic=True);approval=dict(commit='f'*40,data_bindings={'Weather':{t['id']:digest(metadata)}})
        updates=[];evaluations=[];reads=[];epoch=[0]
        def initialize(*args):return p,SyntheticModel(),Events(),torch.Generator().manual_seed(2024)
        def load(config,task,test_capability=None):
            reads.append(test_capability)
            return dict(train='train',validation='validation',test='test'),metadata
        def batches(dataset,*args):return [None]*a['train_batches'] if dataset=='train' else dataset
        def update(model,opt,*args):opt.step();updates.append(1)
        def evaluate(model,dataset,*args):
            if dataset=='test':
                mse=1.; elements=a['test_windows_arithmetic_only']*p['pred_len']*p['C'];evaluations.append('test')
            else:
                epoch[0]+=1
                if epoch[0]==interrupt:raise RuntimeError('synthetic interruption before committed epoch')
                mse=values[epoch[0]-1];elements=a['validation_windows']*p['pred_len']*p['C'];evaluations.append(epoch[0])
            return dict(mse=mse,mae=mse,sse=mse*elements,sae=mse*elements,elements=elements)
        stack.enter_context(patch.object(runner,'init_training',side_effect=initialize))
        stack.enter_context(patch.object(runner,'update',side_effect=update))
        stack.enter_context(patch.object(runner,'evaluate',side_effect=evaluate))
        stack.enter_context(patch.object(data,'load',side_effect=load))
        stack.enter_context(patch.object(data,'batches',side_effect=batches))
        stack.enter_context(patch.object(marks,'batch_parts',return_value=(None,None,None)))
        def closeout():
            history=[json.loads(x)for x in (out/'history.jsonl').read_text().splitlines()]
            steps=history[-1]['steps'];counts=dict(adam=steps,backward=steps,forward=len(history)*(a['train_batches']+math.ceil(a['validation_windows']/128))+math.ceil(a['test_windows_arithmetic_only']/128))
            exclusive(out/'budget.json',dict(counts=counts,by_pid={'123':counts}))
            exclusive(out/'runtime.json',dict(task=t['id'],pid=123,error=None))
            return execution.technical_group(c,model,approval),history
        yield dict(c=c,t=t,p=p,a=a,out=out,approval=approval,runner=runner,execution=execution,closeout=closeout,updates=updates,evaluations=evaluations,reads=reads,epoch=epoch,stack=stack)


class EarlyStopping(unittest.TestCase):
    def test_strict_improvement_ties_reset_and_limit(self):
        b=BestState(10)
        self.assertTrue(b.update(2.,1))
        self.assertFalse(b.update(2.,2));self.assertEqual((b.bad,b.epoch),(1,1))
        self.assertTrue(b.update(1.,3));self.assertEqual(b.bad,0)
        for e in range(4,13):self.assertFalse(b.update(1.,e));self.assertFalse(b.stopped)
        b.update(1.,13);self.assertTrue(b.stopped);self.assertEqual(b.epoch,3)

    def test_real_formal_loop_stops_after_ten_nonimprovements(self):
        for stage in ('M_AMEND','M_ALL'):
            with self.subTest(stage=stage),formal_fixture(stage,[2.]*20)as f:
                f['runner'].formal_worker(f['c'],f['t'],f['out'],f['approval'])
                receipt,h=f['closeout']();self.assertTrue(receipt['technical_complete'])
                self.assertEqual(len(h),11);self.assertEqual(h[-1]['best_epoch'],1)
                self.assertEqual(len(f['updates']),11*f['a']['train_batches'])
                self.assertEqual(f['evaluations'].count('test'),1)
                self.assertEqual(sum(x is not None for x in f['reads']),1)
                self.assertEqual(f['p']['training']['scheduler']['epochs'],20)
                saved=__import__('torch').load(f['out']/'last.pt',map_location='cpu')
                self.assertEqual((saved['best']['bad'],saved['best']['patience'],saved['epoch']),(10,10,11))
                self.assertEqual(saved['scheduler']['config'],f['p']['training']['scheduler'])

    def test_full_twenty_epochs_legally_complete_both_schedulers(self):
        for stage in ('M_AMEND','M_ALL'):
            with self.subTest(stage=stage),formal_fixture(stage,list(range(20,0,-1)))as f:
                f['runner'].formal_worker(f['c'],f['t'],f['out'],f['approval'])
                receipt,h=f['closeout']();self.assertTrue(receipt['technical_complete'])
                self.assertEqual((len(h),h[-1]['best_epoch']),(20,20))
                self.assertEqual(f['evaluations'].count('test'),1)

    def test_actual_checkpoint_resume_preserves_bad_and_fixed_scheduler(self):
        for stage in ('M_AMEND','M_ALL'):
            with self.subTest(stage=stage),formal_fixture(stage,[2.]*20,interrupt=5)as f:
                with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):
                    f['runner'].formal_worker(f['c'],f['t'],f['out'],f['approval'])
                import torch
                state=torch.load(f['out']/'last.pt',map_location='cpu')
                self.assertEqual((state['epoch'],state['best']['bad']),(4,3))
                f['approval']['resume_audits']={f['t']['id']:dict(mode='resume',test_access_status='not_accessed',sha256={name:sha(f['out']/name)for name in ('manifest.json','last.pt','best.pt','history.jsonl')})}
                f['epoch'][0]=4
                # Drop the synthetic interrupt; the actual audited restore,
                # BestState and scheduler load paths remain unmocked.
                f['stack'].enter_context(patch.object(f['runner'],'evaluate',side_effect=lambda model,dataset,*args:self.metric(f,dataset)))
                f['runner'].formal_worker(f['c'],f['t'],f['out'],f['approval'],resume=True)
                receipt,h=f['closeout']();self.assertTrue(receipt['technical_complete'])
                self.assertEqual((len(h),h[-1]['epoch'],h[-1]['best_epoch']),(11,11,1))
                state=torch.load(f['out']/'last.pt',map_location='cpu')
                self.assertEqual((state['best']['bad'],state['scheduler']['updates']),(10,11*f['a']['train_batches']))
                self.assertEqual(state['scheduler']['config']['epochs'],20)
                self.assertEqual(f['evaluations'].count('test'),1)

    @staticmethod
    def metric(f,dataset):
        if dataset=='test':
            f['evaluations'].append('test');elements=f['a']['test_windows_arithmetic_only']*f['p']['pred_len']*f['p']['C']
        else:
            f['epoch'][0]+=1;f['evaluations'].append(f['epoch'][0]);elements=f['a']['validation_windows']*f['p']['pred_len']*f['p']['C']
        return dict(mse=2.,mae=2.,sse=2.*elements,sae=2.*elements,elements=elements)

    def test_unknown_test_state_refused_before_checkpoint_read(self):
        with formal_fixture('M_ALL',[2.]*20,interrupt=5)as f:
            with self.assertRaises(RuntimeError):f['runner'].formal_worker(f['c'],f['t'],f['out'],f['approval'])
            f['approval']['resume_audits']={f['t']['id']:dict(mode='resume',test_access_status='unknown')}
            with patch('torch.load',side_effect=AssertionError('checkpoint read forbidden')):
                with self.assertRaises(PermissionError):f['runner'].formal_worker(f['c'],f['t'],f['out'],f['approval'],resume=True)

    def test_technical_audit_rejects_truncation_overrun_and_second_test(self):
        for stage in ('M_AMEND','M_ALL'):
            with self.subTest(stage=stage),formal_fixture(stage,[2.]*20)as f:
                f['runner'].formal_worker(f['c'],f['t'],f['out'],f['approval']);f['closeout']()
                history=f['out']/'history.jsonl';original=history.read_text();result=f['out']/'result.json';saved=result.read_text()
                history.write_text(''.join(original.splitlines(keepends=True)[:10]))
                with self.assertRaisesRegex(ValueError,'incomplete epoch'):f['execution'].technical_group(f['c'],'AMD',f['approval'])
                history.write_text(original);row=json.loads(original.splitlines()[-1]);row['epoch']=12;row['steps']=12*f['a']['train_batches']
                history.write_text(original+json.dumps(row)+'\n')
                with self.assertRaisesRegex(ValueError,'early-stopping'):f['execution'].technical_group(f['c'],'AMD',f['approval'])
                history.write_text(original);v=json.loads(saved);v['final_test']['calls']=2;result.write_text(json.dumps(v))
                with self.assertRaisesRegex(ValueError,'test-once'):f['execution'].technical_group(f['c'],'AMD',f['approval'])

    def test_active_scope_stays_fresh_only(self):
        from utils.ch3_type1_execution import make_config
        with attempt():
            c=q.configs()['M_ALL']
            # Mirror the actual group-child tool import context, not preflight.
            with patch.object(sys,'path',[str(s.ROOT/'tools/restricted_regression'),*sys.path]),self.assertRaisesRegex(PermissionError,'fresh only'):
                make_config(c,'ch3_formal','unused',task=c['tasks'][0]['id'],approval={},resume=True)


if __name__ == '__main__':
    unittest.main()
