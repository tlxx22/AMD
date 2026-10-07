"""Routine defaults only; no model construction, torch import or GPU compute."""
import copy,math,struct,sys,tempfile,types,unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import ch3_runner as runner
from utils import ch3_contract as k,ch3_native_execution as native,ch3_type1_chain as q,ch3_type1_tasks as s
from utils import ch3_moderntcn_etth1_recovery as r
from utils.ch3_native_recovery_records import bound,ref
from tests.test_m6_probe_schema_recovery import trajectory
from tests.test_m6_moderntcn_etth1_recovery import attempt
from tests import test_m6_ms_seal_recovery as ms_tests

class Defaults(unittest.TestCase):
    def test_all_133_rules_427_tasks_and_106_actual_probe_groups(self):
        counts={'M_BASE':(84,21),'M_AMEND':(112,28),'M_ALL':(168,42),'URBAN_SUBSET':(28,7),'EPF_ALL':(35,35)}
        with attempt():
            cs=q.configs();self.assertEqual(set(cs),set(counts));rules=[];groups=0;representatives=0
            for stage,c in cs.items():
                registry=k.validate_baseline_numeric_registry(c);self.assertEqual((len(c['tasks']),len(registry)),counts[stage])
                self.assertTrue(all(x and x['kind']=='full_float_state'for x in registry.values()))
                rules.extend((stage,x)for x in registry.values());gg=s.probe_groups(c);groups+=len(gg);representatives+=sum(len(g['representatives'])for g in gg)
                for t in c['tasks']:self.assertEqual(k.numeric_probe_policy(c,t),registry[t['model']+'-'+t['dataset']])
            self.assertEqual((len(rules),groups,representatives),(133,106,427))
            self.assertEqual(sum(x['task']=='M'for _,x in rules),91)
    def test_loss_exception_and_EPF_exception_are_exactly_scoped(self):
        with attempt():
            exceptions=[];epf=[]
            for stage,c in q.configs().items():
                for rule in c['baseline_unified']['numeric_policies'].values():
                    expected=2e-4 if rule['task']=='M'else 5e-4 if rule['model']=='ModernTCN'and rule['dataset']!='UrbanEV'else 1e-4
                    self.assertEqual(rule['state_atol'],expected)
                    self.assertEqual((rule['metric_atol'],rule['loss_atol'],rule['rtol'],rule['equal_nan']),(1e-6,1e-6,0,False))
                    special=rule['task']=='M'and(rule['model'],rule['dataset'])==('TimeMixer','Weather')
                    self.assertEqual(rule['loss_rtol'],1e-5 if special else 0)
                    if special:exceptions.append(stage)
                    if rule['state_atol']==5e-4:epf.append(rule['dataset'])
            self.assertEqual(set(exceptions),{'M_BASE','M_AMEND','M_ALL'});self.assertEqual(set(epf),{'PJM','NP','BE','FR','DE'})
    def test_none_exact_entries_materialize_without_mutating_old_versions(self):
        extension=bound(r.ALL_M_POLICY_REF);converted=0
        for stage,row in extension['stages'].items():
            old=bound(row['old_config_ref']);before=copy.deepcopy(old);new=k.materialize_baseline_numeric_admission(old)
            self.assertEqual(old,before);self.assertNotIn('numeric_admission_version',old['baseline_unified'])
            converted+=sum(x is None for x in old['baseline_unified']['numeric_policies'].values())
            for t in old['tasks']:
                if old['baseline_unified']['numeric_policies'][t['model']+'-'+t['dataset']]is None:self.assertIsNone(k.numeric_probe_policy(old,t))
                self.assertEqual(k.numeric_probe_policy(new,t)['kind'],'full_float_state')
        self.assertEqual(converted,86)
        prior_exact=0
        for stage,row in extension['stages'].items():
            relative=Path(r.CONFIG_REFS[stage]['path']).relative_to(k.ROOT)
            frozen=r.POLICY_PACKAGE/'before'/relative
            before=bound(ref(frozen))if frozen.exists()else bound(row['old_config_ref'])
            prior_exact+=sum(x is None for x in before['baseline_unified']['numeric_policies'].values())
        self.assertEqual(prior_exact,84)
    def test_science_metadata_source_task_order_and_all_budgets_unchanged(self):
        extension=bound(r.ALL_M_POLICY_REF)
        for stage,row in extension['stages'].items():
            old=bound(row['old_config_ref']);new=bound(r.CONFIG_REFS[stage]);left=copy.deepcopy(old);right=copy.deepcopy(new)
            for field in ('numeric_policies','numeric_admission_version','numeric_revision_ref'):
                left['baseline_unified'].pop(field,None);right['baseline_unified'].pop(field,None)
            self.assertEqual(left,right);self.assertEqual(s.formal_budget(old),s.formal_budget(new));self.assertEqual(s.probe_budget(old),s.probe_budget(new));r.validate_revision(new)
    def test_registry_missing_none_foreign_policy_version_and_horizon_rejected(self):
        c=bound(r.CONFIG_REF);t=c['tasks'][0];key=t['model']+'-'+t['dataset']
        for mode in ('missing','none','foreign','version','H'):
            v=copy.deepcopy(c);tt=k.task_by_id(v,t['id'])
            if mode=='missing':del v['baseline_unified']['numeric_policies'][key]
            elif mode=='none':v['baseline_unified']['numeric_policies'][key]=None
            elif mode=='foreign':v['baseline_unified']['numeric_policies'][key]['dataset']='Exchange'
            elif mode=='version':v['baseline_unified']['numeric_admission_version']='unknown'
            else:tt['h']=97
            with self.assertRaises(ValueError,msg=mode):k.numeric_probe_policy(v,tt)
    def test_unregistered_model_category_and_M_MS_conflicts_rejected(self):
        c=bound(r.CONFIG_REF)
        for field,value in [('model','J'),('dataset','ECL'),('task','MS'),('input_variant','F4'),('metric_scope','target_only')]:
            v=copy.deepcopy(c);t=v['tasks'][0];t[field]=value
            with self.assertRaises((ValueError,KeyError)):k.baseline_numeric_admission_policy(v,t)
        for field,value in [('supervised_channels',[0]),('output_order',[]),('C',1)]:
            v=copy.deepcopy(c);t=v['tasks'][0];v['resolved_profiles'][t['id']][field]=value
            with self.assertRaises(ValueError):k.baseline_numeric_admission_policy(v,t)
    def test_old_strict_evidence_inclusion_and_collection_evaluation_identity(self):
        previous=bound(r.PRIOR_REUSE_REF);proof=bound(r.INCLUSION_REF);adopted=r.evidence()
        self.assertEqual((proof['retained_groups'],proof['retained_probe_trajectories'],proof['new_numeric_replays'],proof['new_compute']),(16,128,0,0))
        self.assertEqual(len(previous['decisions']),16);self.assertFalse(proof['whole_M_admission'])
        for group,rows in proof['records'].items():
            for original,row,new in zip(previous['decisions'][group]['numerical_comparisons'],rows,adopted['decisions'][group]['numerical_comparisons']):
                self.assertEqual(original,row['collection_comparison']);self.assertEqual(new['collection_comparison'],original)
                self.assertTrue(row['identity_verified']);self.assertGreater(row['complete_floating_elements'],0)
                self.assertTrue(all(row['old_limits'][key]<=row['new_limits'][key]for key in row['old_limits']))
                self.assertEqual(new['policy_id'],row['evaluation_policy']['id']);self.assertEqual(new['policy_inclusion_ref'],r.INCLUSION_REF)
        self.assertEqual(previous['artifacts'],adopted['artifacts']);self.assertEqual(previous['producer_refs'],adopted['producer_refs'])

class Comparison(unittest.TestCase):
    def pair(self,stage='M_BASE',model='AMD',name=None):
        with attempt():c=q.configs()[stage]
        t=next(t for t in c['tasks']if t['model']==model and(name is None or t['dataset']==name));rule=k.numeric_probe_policy(c,t)
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name)
        return c,t,rule,trajectory(c,t,root/'serial',rule),trajectory(c,t,root/'parallel',rule),root
    def payload(self,tr,delta=0,index=0,value=None):
        for point in tr['full_numeric_trace']:
            values=[1.,1.,1.,1.,1.,float(point['step'])];values[index]=values[index]+delta if value is None else value
            path=Path(point['data_file']);path.write_bytes(struct.pack('<6d',*values));point['data_sha']=ref(path)['sha256']
    def metrics(self,tr,delta):
        v=tr['validation'];rows=v.get('channels',[v])
        for row in rows:row.update(sse=(1+delta)*row['elements'],mse=(1+delta)*row['elements']/row['elements'])
        v.update(sse=(1+delta)*v['elements'],mse=(1+delta)*v['elements']/v['elements'])
    def test_identical_and_nonzero_bounded_actual_compare_all_categories(self):
        for stage in ('M_BASE','M_AMEND','M_ALL','URBAN_SUBSET','EPF_ALL'):
            c,t,rule,x,y,root=self.pair(stage)
            with patch.object(native.scope,'context',return_value={'probe_root':root}):
                result=native.compare(c,t,x,y);self.assertTrue(result['passed']);self.assertTrue(result['bitwise_equal'])
                self.payload(y,.5*rule['state_atol']);result=native.compare(c,t,x,y)
                self.assertTrue(result['passed']);self.assertFalse(result['bitwise_equal']);self.assertTrue(result['exact_residual_state'])
    def test_parameter_buffer_gradient_and_both_moments_checked(self):
        for index in range(5):
            c,t,rule,x,y,_=self.pair();self.payload(y,1.1*rule['state_atol'],index)
            self.assertFalse(runner.compare_probe_trajectories(c,t,x,y)['passed'])
    def test_optimizer_step_nonfloating_and_groups_remain_exact(self):
        c,t,rule,x,y,_=self.pair();self.payload(y,1e-12,5)
        self.assertFalse(runner.compare_probe_trajectories(c,t,x,y)['passed'])
        schema=bound(dict(path=x['full_numeric_trace'][0]['schema_file'],sha256=x['full_numeric_trace'][0]['schema_sha']))
        for mode in ('step','dtype','groups','offset'):
            bad=copy.deepcopy(schema)
            if mode=='step':bad['entries'][-1]['mode']='bounded'
            elif mode=='dtype':bad['entries'][0]['dtype']='torch.complex64'
            elif mode=='groups':bad['optimizer_groups'][0]['params']=[]
            else:bad['entries'][0]['offset']=8
            with self.assertRaises(ValueError):k.validate_baseline_full_schema(bad,rule)
        # Identical integer tensors are exact even under an arbitrarily large float bound.
        schema['entries'][1].update(dtype='torch.int64',mode='exact');k.validate_baseline_full_schema(schema,rule)
    def test_NaN_Inf_in_bounded_and_exact_float_state_always_rejected(self):
        for index in (0,5):
            for value in (float('nan'),float('inf')):
                c,t,rule,x,y,_=self.pair();self.payload(x,index=index,value=value);self.payload(y,index=index,value=value)
                with self.assertRaises(ValueError):runner.compare_probe_trajectories(c,t,x,y)
    def test_loss_and_normalized_metrics_bound_and_TimeMixer_Weather_exception(self):
        for model,name in [('AMD','Weather'),('TimeMixer','Weather'),('TimeMixer','ETTh1')]:
            c,t,rule,x,y,_=self.pair(model=model,name=name);y['trajectory'][0]['loss']+=5e-6
            self.assertEqual(runner.compare_probe_trajectories(c,t,x,y)['passed'],model=='TimeMixer'and name=='Weather')
            y['trajectory'][0]['loss']+=1e-3;self.assertFalse(runner.compare_probe_trajectories(c,t,x,y)['passed'])
        c,t,rule,x,y,_=self.pair();self.metrics(y,5e-7)
        self.assertGreater(abs(y['validation']['sse']-x['validation']['sse']),1e-6)
        self.assertTrue(runner.compare_probe_trajectories(c,t,x,y)['passed'])
        self.metrics(y,2e-6);self.assertFalse(runner.compare_probe_trajectories(c,t,x,y)['passed'])
    def test_identity_schema_missing_gradients_moments_and_aggregation_rejected(self):
        c,t,rule,x,y,_=self.pair();schema=bound(dict(path=x['full_numeric_trace'][0]['schema_file'],sha256=x['full_numeric_trace'][0]['schema_sha']))
        for path in ('gradient/w','optimizer/w/exp_avg','optimizer/w/exp_avg_sq'):
            bad=copy.deepcopy(schema);bad['entries']=[e for e in bad['entries']if e['path']!=path];offset=0
            for e in bad['entries']:e['offset']=offset;offset+=e['nbytes']
            bad['total_bytes']=offset
            with self.assertRaises(ValueError):k.validate_baseline_full_schema(bad,rule)
        for key,value in [('initial','other'),('initial_rng','other'),('batch_ids',[]),('finite',False),('numeric_policy_sha','wrong')]:
            bad=copy.deepcopy(y);bad[key]=value
            with self.assertRaises(ValueError):runner.compare_probe_trajectories(c,t,x,bad)
        for mode in ('extra','missing','order','aggregation'):
            bad=copy.deepcopy(y)
            if mode=='extra':bad['validation']['unknown']=0
            elif mode=='missing':bad['validation'].pop('channels')
            elif mode=='order':bad['validation']['channels'].reverse()
            else:bad['validation']['sse']+=.1
            with self.assertRaises(ValueError):runner.compare_probe_trajectories(c,t,x,bad)
    def test_OneCycle_type1_scheduler_and_generic_M_payload_compare_once(self):
        for stage in ('M_BASE','M_ALL'):
            c,t,rule,x,y,root=self.pair(stage)
            with patch.object(native.scope,'context',return_value={'probe_root':root}),patch.object(runner,'_compare_full_numeric_files',wraps=runner._compare_full_numeric_files)as compare:
                self.assertTrue(native.compare(c,t,x,y)['passed']);self.assertEqual(compare.call_count,6)
                y['scheduler_trace'][0]['updates']+=1
                with self.assertRaises(ValueError):native.compare(c,t,x,y)
    def test_TimeMixer_Urban_endpoint_preservation_still_required(self):
        from utils import ch3_urban_confirmation as urban
        c,t,rule,x,y,_=self.pair('URBAN_SUBSET','TimeMixer');p=k.profile(c,t)
        target=dict(index=p['target_idx'],pred_len=p['pred_len'],C=p['C'],metric_space='train-standardized target-only')
        for tr in (x,y):
            state=dict(rng=tr['final_rng'],model=tr['final'],optimizer=tr['trajectory'][-1]['optimizer'])
            tr['urban_confirmation']=dict(policy_sha=k.digest(rule),evaluations={str(i):dict(metrics=copy.deepcopy(tr['validation']),batch_ids=['one','tail'],before=state,after=state,mode_restored=True,target=target)for i in (2,6)})
        detail=dict(points=[dict(step=i,metadata_equal=True,rng_equal=True,identical=i==0,tensors=[dict(name='w',structure_equal=True,finite=True,max_abs_diff=0 if i==0 else 5e-5,exact_required=False,byte_equal=i==0)])for i in range(7)])
        self.assertTrue(urban.compare_measured(c,t,x,y,detail)['passed'])
        bad=copy.deepcopy(y);bad['urban_confirmation']['evaluations']['6']['after']={'rng':'changed'}
        with self.assertRaises(ValueError):urban.compare_measured(c,t,x,bad,detail)

class Capture(unittest.TestCase):
    def fixture(self,tmp):
        class Tensor:
            dtype='torch.float64';layout='dense';shape=(1,)
            def __init__(self,value):self.value=value;self.grad=None
            def numel(self):return 1
            def element_size(self):return 8
            def is_floating_point(self):return True
        w=Tensor(1);w.grad=Tensor(1);b=Tensor(1)
        container=types.SimpleNamespace(named_parameters=lambda:[('w',w)],named_buffers=lambda:[('b',b)])
        opt=types.SimpleNamespace(param_groups=[dict(params=[w],lr=1e-4,betas=(.9,.999))],state={w:dict(step=Tensor(1),exp_avg=Tensor(1),exp_avg_sq=Tensor(1))})
        copier=types.SimpleNamespace(copy_bytes=lambda t:(struct.pack('<d',t.value),None))
        fake=types.SimpleNamespace(is_tensor=lambda t:isinstance(t,Tensor),strided='dense',isfinite=lambda t:types.SimpleNamespace(all=lambda:math.isfinite(t.value)))
        c=bound(r.CONFIG_REF);rule=k.numeric_probe_policy(c,c['tasks'][0]);writer=runner.FullNumericStateWriter(container,opt,tmp,copier,rule)
        return fake,writer,w,opt,rule
    def test_actual_writer_captures_all_state_and_structure_is_frozen(self):
        with tempfile.TemporaryDirectory()as tmp:
            fake,writer,w,opt,rule=self.fixture(tmp)
            with patch.dict(sys.modules,{'torch':fake}):
                first=writer.capture(1);k.validate_baseline_full_schema(writer.schema,rule);w.value+=1e-4;second=writer.capture(2)
                row=runner._compare_full_numeric_files(rule,first,second);self.assertTrue(row['passed']);self.assertGreater(row['state_max_abs'],0)
                self.assertEqual(len(writer.schema['entries']),6);w.grad=None
                with self.assertRaises(ValueError):writer.capture(3)
        self.assertNotIn('torch',sys.modules)
    def test_actual_writer_rejects_nonfinite_without_model_or_optimizer_compute(self):
        with tempfile.TemporaryDirectory()as tmp:
            fake,writer,w,opt,rule=self.fixture(tmp);w.value=float('nan')
            with patch.dict(sys.modules,{'torch':fake}),self.assertRaises(FloatingPointError):writer.capture(1)
        self.assertNotIn('torch',sys.modules)

class Formal(ms_tests.Runtime):
    """Execute inherited protected entrypoints against the five new policies."""
    @contextmanager
    def fixture(self):
        with attempt(),super().fixture()as values:yield values

if __name__=='__main__':unittest.main()
