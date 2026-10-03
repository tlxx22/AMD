"""Five-scope admission revision; pure fixtures and read-only saved v1 replay."""
import copy,json,struct,tempfile,unittest
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch
from utils import ch3_baseline_unified_tasks as s
from utils import ch3_baseline_unified_chain as q
from utils import ch3_native_execution as native
from utils.ch3_contract import digest,profile
from utils.ch3_native_recovery_records import bound,sha

EPF=('PJM','NP','BE','FR','DE')
V1_RESULT=s.RESULT.with_name('baseline-unified96-onecycle001-v1')


def old_config(stage):
    return json.loads((s.ROOT/'configs'/('ch3_baseline_'+stage.lower()+'_u96_oc01.json')).read_text())


@lru_cache(maxsize=1)
def replay_saved_v1():
    """In-memory policy annotation only; never a v2 admission or a changed v1 report."""
    c=old_config('MS');root=V1_RESULT/'probe/MS'
    c['baseline_unified']['numeric_policies'].update({'ModernTCN-'+d:copy.deepcopy(s.MODERNTCN_EPF_POLICY) for d in EPF})
    saved=bound(dict(path=str(s.PACKAGE/'v1-retained-before.json'),sha256=sha(s.PACKAGE/'v1-retained-before.json')))['files']
    result=[]
    for domain in EPF[:4]:
        t=next(t for t in c['tasks'] if t['model']=='ModernTCN' and t['dataset']==domain)
        values=[];refs=[];categories=set()
        for phase in ('serial','q4'):
            path=root/'ModernTCN-EPF-compatible-0'/phase/t['id']/'trajectory.json'
            assert sha(path)==saved[str(path)]['sha256']
            value=json.loads(path.read_text());refs.append(dict(path=str(path),sha256=sha(path)))
            # The exact run captured M_full_state_trace even under policy=None.
            # Project that saved trace into the generic full-state comparator in memory.
            value['numeric_policy_sha']=digest(s.MODERNTCN_EPF_POLICY)
            value['full_numeric_trace']=value['M_full_state_trace']
            for point in value['M_full_state_trace']:
                for key in ('schema_file','data_file'):
                    p=Path(point[key]);assert sha(p)==saved[str(p)]['sha256']
                schema=json.loads(Path(point['schema_file']).read_text())
                for e in schema['entries']:
                    p=e['path']
                    if p.startswith('model/parameter/'):categories.add('parameters')
                    if p.startswith('model/buffer/'):categories.add('buffers')
                    if p.startswith('gradient/'):categories.add('gradients')
                    if p.endswith('/exp_avg'):categories.add('Adam.exp_avg')
                    if p.endswith('/exp_avg_sq'):categories.add('Adam.exp_avg_sq')
                    if p.endswith('/step'):assert e['mode']=='exact';categories.add('optimizer_step_exact')
            values.append(value)
        with patch('utils.ch3_native_tasks.context',return_value={'probe_root':root}):
            row=native.compare(c,t,*values)
        assert values[0]['scheduler_trace']==values[1]['scheduler_trace']
        assert categories=={'parameters','buffers','gradients','Adam.exp_avg','Adam.exp_avg_sq','optimizer_step_exact'}
        for r in refs:assert sha(r['path'])==r['sha256']
        result.append(dict(dataset=domain,parent_task_id=t['id'],passed=row['passed'],state_max_abs=row['state_max_abs'],loss_max_abs=row['loss_max_abs'],metric_max_abs=row['validation_max_abs'],finite=True,scheduler_LR_beta1_exact=True,exact_residual_state=row['exact_residual_state'],compared_elements=row['compared_elements'],captured_categories=sorted(categories),source_trajectory_refs=refs))
    return dict(purpose='offline_saved_v1_payload_policy_replay_only',policy=s.MODERNTCN_EPF_POLICY,policy_sha=digest(s.MODERNTCN_EPF_POLICY),new_v2_admission=False,old_payloads_modified=False,rows=result)


class UnifiedV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cs=q.configs()

    def test_matrix_exact(self):
        self.assertEqual([len(self.cs[k]['tasks']) for k in ('MS','M')],[203,84])
        self.assertEqual(len({t['id'] for c in self.cs.values() for t in c['tasks']}),287)
        for stage,c in self.cs.items():
            self.assertEqual({t['model'] for t in c['tasks']},set(s.MODELS))
            self.assertEqual(set(c['datasets']),set(s.MS_DOMAINS if stage=='MS' else s.M_DOMAINS))
            self.assertTrue(all('-u96-oc01-v2-' in t['id'] for t in c['tasks']))

    def test_only_five_policies_changed(self):
        for stage,c in self.cs.items():
            before=old_config(stage)['baseline_unified']['numeric_policies'];after=c['baseline_unified']['numeric_policies']
            self.assertEqual(set(before),set(after))
            self.assertEqual({k for k in before if before[k]!=after[k]}, {'ModernTCN-'+d for d in EPF} if stage=='MS' else set())
            for d in EPF:self.assertIsNone(old_config('MS')['baseline_unified']['numeric_policies']['ModernTCN-'+d])

    def test_policy_exact_constants(self):
        for d in EPF:
            p=self.cs['MS']['baseline_unified']['numeric_policies']['ModernTCN-'+d]
            self.assertEqual(p,s.MODERNTCN_EPF_POLICY)
            for key,value in dict(kind='full_float_state',state_atol=5e-4,loss_atol=1e-6,metric_atol=1e-6,loss_rtol=0.0,rtol=0.0,equal_nan=False).items():self.assertEqual(p[key],value)

    def test_UrbanEV_and_M_unchanged(self):
        self.assertIsNone(self.cs['MS']['baseline_unified']['numeric_policies']['ModernTCN-UrbanEV'])
        self.assertEqual(self.cs['M']['baseline_unified']['numeric_policies'],old_config('M')['baseline_unified']['numeric_policies'])

    def test_287_profiles_unchanged(self):
        for stage,c in self.cs.items():
            old=old_config(stage)
            for a,b in zip(old['tasks'],c['tasks']):
                self.assertEqual(old['resolved_profiles'][a['id']],profile(c,b))
                self.assertEqual({k:v for k,v in a.items() if k not in ('id','group','profile')},{k:v for k,v in b.items() if k not in ('id','group','profile')})
            for key in ('datasets','sources','contract','execution','training_common','urban_folds','urban_input_variants'):self.assertEqual(c[key],old[key])

    def test_data_metadata_unchanged(self):
        for stage,c in self.cs.items():
            old=old_config(stage);a=bound(old['baseline_unified']['data_ref']);b=bound(c['baseline_unified']['data_ref'])
            self.assertEqual(a['source_states'],b['source_states'])
            for ot,nt in zip(old['tasks'],c['tasks']):self.assertEqual(a['metadata'][ot['dataset']][ot['id']],b['metadata'][nt['dataset']][nt['id']])

    def test_validate_rejects_any_other_policy_revision(self):
        for stage,key in (('MS','ModernTCN-UrbanEV'),('MS','TimeMixer-PJM'),('M','ModernTCN-Weather')):
            c=copy.deepcopy(self.cs[stage]);c['baseline_unified']['numeric_policies'][key]=copy.deepcopy(s.MODERNTCN_EPF_POLICY)
            with self.assertRaisesRegex(ValueError,'numeric registry'):s.validate(c)
        c=copy.deepcopy(self.cs['MS']);c['baseline_unified']['numeric_policies']['ModernTCN-PJM']['state_atol']=1e-3
        with self.assertRaises(ValueError):s.validate(c)

    def test_saved_four_members_replay_passed(self):
        r=replay_saved_v1();self.assertFalse(r['new_v2_admission']);self.assertFalse(r['old_payloads_modified'])
        self.assertEqual([x['dataset'] for x in r['rows']],list(EPF[:4]))
        maxima=[1.3441592454910278e-4,2.063065767288208e-4,2.4513527750968933e-4,1.7508119344711304e-4]
        for row,value in zip(r['rows'],maxima):
            self.assertTrue(row['passed']);self.assertEqual(row['state_max_abs'],value);self.assertTrue(row['exact_residual_state']);self.assertGreater(row['compared_elements'],0)

    def test_replay_finite_identity_reject(self):
        c=old_config('MS');t=next(t for t in c['tasks'] if t['model']=='ModernTCN' and t['dataset']=='PJM')
        x=json.loads((V1_RESULT/'probe/MS/ModernTCN-EPF-compatible-0/serial'/t['id']/'trajectory.json').read_text())
        from ch3_runner import compare_probe_trajectories
        for key,bad in (('finite',False),('initial','bad'),('initial_rng',{}),('batch_ids',[]),('steps',5),('final_rng',{})):
            y=copy.deepcopy(x);y[key]=bad
            with self.assertRaises(ValueError):compare_probe_trajectories(c,t,x,y)

    def test_scheduler_LR_beta1_remain_exact(self):
        c=old_config('MS');t=next(t for t in c['tasks'] if t['model']=='ModernTCN' and t['dataset']=='PJM')
        x=json.loads((V1_RESULT/'probe/MS/ModernTCN-EPF-compatible-0/serial'/t['id']/'trajectory.json').read_text())
        for key in ('lr','betas'):
            y=copy.deepcopy(x);y['scheduler_trace'][0]['groups'][0][key]=0
            with patch('ch3_runner.compare_probe_trajectories',return_value={'passed':True}):
                with self.assertRaisesRegex(ValueError,'scheduler LR/beta'):native.compare(c,t,x,y)

    def test_old_start_authorization_rejected(self):
        old=bound(dict(path=str(s.PARENT_PACKAGE/'start-review.json'),sha256='8389d06f9a3ffaa9b921e3665dd8894bf599427a7754956419ed6248ec4a0e1e'))
        with patch.object(q,'closure',return_value=old['closure_commit']):
            with self.assertRaisesRegex(PermissionError,'exact unified authorization'):q.validate_start(old)

    def test_v1_permit_and_paths_rejected(self):
        a=json.loads((V1_RESULT/'probe/MS/approval.json').read_text())
        with patch.object(q,'validate_start'),patch.object(q,'dynamic',return_value={}):
            with self.assertRaisesRegex(PermissionError,'exact machine-gate'):q.validate_permit(self.cs['MS'],a,True)
        from utils.ch3_baseline_unified_execution import read_config
        c=self.cs['MS'];t=c['tasks'][0]
        with self.assertRaises((PermissionError,ValueError)):
            read_config(dict(unified_scope='m6-baseline-unified96-oc01-v1',protocol_file=str(s.file('MS')),protocol_sha=digest(c),task=t['id']))

    def test_all_15_fresh_and_old_Passed_cannot_enter(self):
        groups=s.probe_groups(self.cs['MS']);self.assertEqual(len(groups),15)
        self.assertEqual(sum(len(g['representatives']) for g in groups),63)
        self.assertEqual(s.context(self.cs['MS'])['probe_root'],s.RESULT/'probe/MS')
        self.assertNotEqual(s.RESULT,V1_RESULT)
        self.assertFalse(s.RESULT.exists())
        self.assertFalse((s.PACKAGE/'start-review.json').exists())
        self.assertFalse((s.context(self.cs['MS'])['probe_root']/'complete.json').exists())
        for g in groups:self.assertTrue(all('v2' in t for t in g['representatives']))
        with patch.object(q,'bound',return_value={'complete_ref':{'path':str(V1_RESULT/'probe/MS/complete.json')}}):
            with self.assertRaises(PermissionError):q.validate_summary_light(self.cs['MS'],{'path':str(V1_RESULT/'queue/MS/admission-summary.json'),'sha256':'bad'})

    def test_budget_fresh_independent_and_no_refund(self):
        v=bound(dict(path=str(s.PACKAGE/'historical-probe-costs.json'),sha256=sha(s.PACKAGE/'historical-probe-costs.json')))
        self.assertEqual(v['v1_actual'],dict(adam=642,backward=642,forward=872));self.assertFalse(v['v1_budget_refund'])
        self.assertEqual(v['v2_starting_debit'],dict(adam=0,backward=0,forward=0))
        self.assertEqual(v['v2_ms_caps'],dict(adam=1122,backward=1122,forward=1520))
        self.assertEqual(v['historical_actual_plus_v2_ms_worst'],dict(adam=1764,backward=1764,forward=2392))
        self.assertEqual(s.probe_budget(self.cs['MS'])['caps'],v['v2_ms_caps'])

    def test_formal_and_M_caps_unchanged(self):
        for stage,c in self.cs.items():
            self.assertEqual(s.formal_budget(c)['total'],s.formal_budget(old_config(stage))['total'])
        self.assertEqual(s.probe_budget(self.cs['M'])['caps'],dict(adam=1512,backward=1512,forward=2016))
        self.assertEqual(len(s.probe_groups(self.cs['M'])),21)

    def test_numeric_never_resource_fallback(self):
        from utils.ch3_m_execution import resource_fallback
        for kind in ('numeric','finite','identity','scheduler','data','guard'):
            self.assertFalse(resource_fallback(dict(failure=kind+' gate failure',returncodes=[0],resource=True)))

    def test_full_payload_float_and_exact_residual(self):
        from ch3_runner import _compare_full_numeric_files
        with tempfile.TemporaryDirectory(dir=s.PACKAGE,prefix='fixture-policy-') as tmp:
            root=Path(tmp);entries=[];raw=b''
            paths=['model/parameter/p','model/buffer/b','gradient/p','optimizer/p/exp_avg','optimizer/p/exp_avg_sq','optimizer/p/step','model/buffer/count']
            for i,p in enumerate(paths):
                mode='bounded' if i<5 else 'exact';data=struct.pack('<f',0.0) if i<6 else struct.pack('<q',0)
                entries.append(dict(path=p,dtype='torch.float32' if i<6 else 'torch.int64',shape=[],mode=mode,offset=len(raw),nbytes=len(data),numel=1));raw+=data
            schema=root/'schema.json';schema.write_text(json.dumps(dict(entries=entries,optimizer_groups=[{'lr':.01,'betas':[.95,.999]}],optimizer_non_tensor_state={})))
            def point(name,data):
                p=root/name;p.write_bytes(data);return dict(schema_file=str(schema),schema_sha=sha(schema),data_file=str(p),data_sha=sha(p),bytes=len(data))
            a=point('a.bin',raw);b=point('b.bin',struct.pack('<f',0.0004)+raw[4:]);self.assertTrue(_compare_full_numeric_files(s.MODERNTCN_EPF_POLICY,a,b)['passed'])
            b=point('c.bin',struct.pack('<f',0.0006)+raw[4:]);self.assertFalse(_compare_full_numeric_files(s.MODERNTCN_EPF_POLICY,a,b)['passed'])
            b=point('step.bin',raw[:20]+struct.pack('<f',1.0)+raw[24:]);self.assertFalse(_compare_full_numeric_files(s.MODERNTCN_EPF_POLICY,a,b)['passed'])
            b=point('nonfloat.bin',raw[:24]+struct.pack('<q',1));self.assertFalse(_compare_full_numeric_files(s.MODERNTCN_EPF_POLICY,a,b)['passed'])
            b=point('nan.bin',struct.pack('<f',float('nan'))+raw[4:])
            with self.assertRaisesRegex(ValueError,'nonfinite'):_compare_full_numeric_files(s.MODERNTCN_EPF_POLICY,a,b)

    def test_no_new_executable_authorization_or_session(self):
        self.assertFalse((s.PACKAGE/'start-review.json').exists());self.assertFalse(s.RESULT.exists())
        self.assertEqual(q.SESSION,'ch3-baseline-unified96-oc01-v2')
        self.assertIn('baseline-unified96-onecycle001-v2',str(q.LOG))
        for stage in ('MS','M'):
            for kind in ('probe','formal'):
                p=json.loads((s.PACKAGE/(stage.lower()+'-'+kind+'-permit.template.json')).read_text())
                self.assertFalse(p['reviewed']);self.assertFalse(p['execution_permitted']);self.assertIsNone(p['commit'])


if __name__=='__main__':unittest.main()
