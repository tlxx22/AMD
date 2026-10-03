"""Reviewed v2 inheritance, UrbanEV 20/5, and fresh v3 gates; no models."""
import ast,copy,inspect,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from utils import ch3_baseline_unified_tasks as s
from utils import ch3_baseline_unified_chain as q
from utils import ch3_baseline_unified_execution as e
from utils.ch3_contract import BestState,digest,profile,step_arithmetic
from utils.ch3_native_recovery_records import bound,sha


class UnifiedV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cs=q.configs()

    def test_exact_MS_count(self):self.assertEqual(len(self.cs['MS']['tasks']),203)
    def test_exact_M_count(self):self.assertEqual(len(self.cs['M']['tasks']),84)
    def test_exact_unique_total(self):
        self.assertEqual(len({t['id'] for c in self.cs.values() for t in c['tasks']}),287)

    def test_models_order_domains_and_exclusions(self):
        self.assertEqual(s.MODELS,('AMD','DLinear','PatchTST','iTransformer','TimeMixer','ModernTCN','TimeXer'))
        for stage,c in self.cs.items():
            self.assertEqual(list(dict.fromkeys(t['model'] for t in c['tasks'])),list(s.MODELS))
            self.assertEqual(set(c['datasets']),set(s.MS_DOMAINS if stage=='MS' else s.M_DOMAINS))
            self.assertFalse({t['model'] for t in c['tasks']}&{'J','N','S'})
            self.assertNotIn('ECL',c['datasets'])

    def test_168_UrbanEV_profiles_20_5_and_fixed_inputs(self):
        selected=[t for t in self.cs['MS']['tasks'] if t['dataset']=='UrbanEV']
        self.assertEqual(len(selected),168)
        for t in selected:
            p=profile(self.cs['MS'],t);tr=p['training']
            self.assertEqual((tr['epochs'],tr['patience'],tr['scheduler']['epochs']),(20,5,20))
            self.assertEqual((p['T'],p['pred_len'],p['C'],p['input_variant']),(12,1,11,'F4'))
            self.assertEqual((tr['batch'],tr['eval_batch'],tr['weight_decay'],tr['seed']),(128,128,1e-7,2024))
            self.assertEqual(tr['scheduler']['steps_per_epoch'],step_arithmetic(self.cs['MS'],t)['train_batches'])

    def test_UrbanEV_scheduler_only_epochs_changes(self):
        for a,b in zip(s.direct_parent('MS')['tasks'],self.cs['MS']['tasks']):
            if b['dataset']!='UrbanEV':continue
            before=s.direct_parent('MS')['resolved_profiles'][a['id']]['training']['scheduler']
            after=profile(self.cs['MS'],b)['training']['scheduler']
            self.assertEqual({k:v for k,v in after.items() if k!='epochs'},{k:v for k,v in before.items() if k!='epochs'})
            self.assertEqual((before['epochs'],after['epochs']),(10,20))

    def test_35_EPF_profiles_exact_parent_and_20_5(self):
        count=0
        for a,b in zip(s.direct_parent('MS')['tasks'],self.cs['MS']['tasks']):
            if b['dataset']=='UrbanEV':continue
            count+=1;p=profile(self.cs['MS'],b)
            self.assertEqual(p,s.direct_parent('MS')['resolved_profiles'][a['id']])
            self.assertEqual((p['training']['epochs'],p['training']['patience']),(20,5))
        self.assertEqual(count,35)

    def test_84_M_profiles_exact_parent_and_10_None(self):
        for a,b in zip(s.direct_parent('M')['tasks'],self.cs['M']['tasks']):
            p=profile(self.cs['M'],b)
            self.assertEqual(p,s.direct_parent('M')['resolved_profiles'][a['id']])
            self.assertEqual((p['training']['epochs'],p['training']['patience']),(10,None))

    def test_both_numeric_registries_exact_v2(self):
        for stage,c in self.cs.items():
            self.assertEqual(c['baseline_unified']['numeric_policies'],s.direct_parent(stage)['baseline_unified']['numeric_policies'])

    def test_ModernTCN_EPF_policy_and_UrbanEV_exact(self):
        registry=self.cs['MS']['baseline_unified']['numeric_policies']
        self.assertIsNone(registry['ModernTCN-UrbanEV'])
        for domain in ('PJM','NP','BE','FR','DE'):self.assertEqual(registry['ModernTCN-'+domain],s.MODERNTCN_EPF_POLICY)
        self.assertEqual(s.MODERNTCN_EPF_POLICY['state_atol'],5e-4)

    def test_profile_diff_proof_exact_whitelist(self):
        for stage in ('MS','M'):
            proof=json.loads((s.PACKAGE/(stage.lower()+'-profile-diff.json')).read_text())
            self.assertEqual(len(proof['rows']),203 if stage=='MS' else 84)
            self.assertEqual(proof['urban_changed'],168 if stage=='MS' else 0)
            for r in proof['rows']:
                fields={x['field'] for x in r['diff']}
                self.assertEqual(fields,{'training.epochs','training.patience','training.scheduler.epochs'} if r['dataset']=='UrbanEV' else set())

    def test_data_metadata_only_task_keys_change(self):
        for stage,c in self.cs.items():
            before=bound(s.direct_parent(stage)['baseline_unified']['data_ref']);after=bound(c['baseline_unified']['data_ref'])
            mapping={a['id']:b['id'] for a,b in zip(s.direct_parent(stage)['tasks'],c['tasks'])}
            for key in ('metadata','data_bindings'):
                self.assertEqual(after[key],{d:{mapping[k]:v for k,v in rows.items()} for d,rows in before[key].items()})
            self.assertEqual({k:v for k,v in after.items() if k not in ('metadata','data_bindings')},{k:v for k,v in before.items() if k not in ('metadata','data_bindings')})

    def test_five_nonimprovements_stop_before_next_epoch(self):
        b=BestState(5);self.assertTrue(b.update(1.0,1))
        for epoch in range(2,6):b.update(1.0,epoch);self.assertFalse(b.stopped)
        b.update(1.0,6);self.assertTrue(b.stopped);self.assertEqual(b.bad,5)

    def test_strict_improvement_and_tie_keep_earlier_best(self):
        b=BestState(5);b.update(1.0,1)
        self.assertFalse(b.update(1.0,2));self.assertEqual(b.epoch,1)
        self.assertTrue(b.update(.99,3));self.assertEqual((b.epoch,b.bad),(3,0))

    def test_BestState_rejects_nonfinite(self):
        for value in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):BestState(5).update(value,1)

    def test_early_stop_test_and_update_math_AST_unchanged(self):
        import ch3_runner,subprocess
        old=ast.parse(subprocess.check_output(['git','show',s.BASE+':ch3_runner.py'],cwd=s.ROOT,text=True))
        new=ast.parse(inspect.getsource(ch3_runner))
        fs=lambda tree:{x.name:ast.dump(x,include_attributes=False) for x in tree.body if isinstance(x,ast.FunctionDef)}
        for name in ('formal_worker','update','evaluate','init_training','save_state','restore_state'):self.assertEqual(fs(old)[name],fs(new)[name])
        src=inspect.getsource(ch3_runner.formal_worker)
        self.assertIn('best.stopped',src);self.assertIn('best.update',src)
        self.assertIn('best.pt',src)

    def test_MS_fresh_probe_15_groups_63_representatives(self):
        groups=s.probe_groups(self.cs['MS'])
        self.assertEqual((len(groups),sum(len(g['representatives']) for g in groups)),(15,63))
        self.assertTrue(all('-u96-oc01-v3-' in r for g in groups for r in g['representatives']))

    def test_M_fresh_probe_21_groups_84_representatives(self):
        groups=s.probe_groups(self.cs['M'])
        self.assertEqual((len(groups),sum(len(g['representatives']) for g in groups)),(21,84))

    def test_MS_probe_nominal_and_caps(self):
        b=s.probe_budget(self.cs['MS'])
        self.assertEqual(b['nominal'],dict(adam=756,backward=756,forward=1024))
        self.assertEqual(b['caps'],dict(adam=1122,backward=1122,forward=1520))

    def test_M_probe_nominal_and_caps(self):
        b=s.probe_budget(self.cs['M'])
        self.assertEqual(b['nominal'],dict(adam=1008,backward=1008,forward=1344))
        self.assertEqual(b['caps'],dict(adam=1512,backward=1512,forward=2016))

    def test_formal_exact_authorized_budget(self):
        values={stage:s.formal_budget(c)['total'] for stage,c in self.cs.items()}
        self.assertEqual(values['MS'],dict(runs=203,run_epochs=4060,adam=15274280,backward=15274280,forward=17173829))
        self.assertEqual(values['M'],dict(runs=84,run_epochs=840,adam=300150,backward=300150,forward=359040))
        self.assertEqual({k:sum(v[k] for v in values.values()) for k in values['MS']},dict(runs=287,run_epochs=4900,adam=15574430,backward=15574430,forward=17532869))

    def test_v2_start_review_cannot_authorize_v3(self):
        a=json.loads((s.PACKAGE.with_name('baseline-unified96-onecycle001-v2')/'start-review.json').read_text())
        with patch.object(q,'closure',return_value=a['closure_commit']):
            with self.assertRaises(PermissionError):q.validate_start(a)

    def test_v2_config_cannot_enter_v3_worker(self):
        value=dict(unified_stage='MS',unified_scope='m6-baseline-unified96-oc01-v2',protocol_file=s.DIRECT_PARENT_REFS['MS']['path'])
        with self.assertRaises(PermissionError):e.read_config(value)

    def test_old_checkpoint_resume_forbidden_before_dispatch(self):
        with self.assertRaises(PermissionError):e.make_config(self.cs['MS'],'ch3_formal',s.PACKAGE/'never-output',task=self.cs['MS']['tasks'][0]['id'],approval={},resume=True)
        self.assertFalse((s.PACKAGE/'never-output').exists())

    def test_v3_namespaces_fresh_and_isolated(self):
        self.assertFalse(s.RESULT.exists());self.assertFalse((s.PACKAGE/'start-review.json').exists())
        self.assertEqual(q.SESSION,'ch3-baseline-unified96-oc01-v3')
        for stage,c in self.cs.items():
            self.assertTrue({t['id'] for t in c['tasks']}.isdisjoint({t['id'] for t in s.direct_parent(stage)['tasks']}))
        self.assertEqual(s.result_index(self.cs)['completed'],0)

    def test_template_unreviewed_unclosed_unexecutable(self):
        a=q.start_template()
        for k in ('reviewed','execution_permitted','structure_frozen','m6_authorized','budget_authorized'):self.assertIs(a[k],False)
        self.assertIsNone(a['closure_commit'])
        self.assertEqual(a,json.loads((s.PACKAGE/'start-approval.template.json').read_text()))

    def test_retirement_has_real_counts_no_refund_no_resume(self):
        r=bound(s.RETIREMENT)
        self.assertEqual(r['reason'],'user_authorized_protocol_supersession')
        self.assertFalse(r['scientific_failure']);self.assertFalse(r['budget_refund']);self.assertFalse(r['resume_into_v3'])
        self.assertEqual(sum(r['counts']['MS'].values()),203);self.assertEqual(sum(r['counts']['M'].values()),84)
        self.assertTrue(all(not row['original_instance_alive'] for row in r['owned_exit_proof']))

    def test_unauthorized_profile_change_rejected(self):
        c=copy.deepcopy(self.cs['MS']);c['resolved_profiles'][c['tasks'][0]['id']]['training']['patience']=6
        with self.assertRaises(ValueError):s.validate(c)
        c=copy.deepcopy(self.cs['M']);c['resolved_profiles'][c['tasks'][0]['id']]['training']['epochs']=20
        with self.assertRaises(ValueError):s.validate(c)

    def test_resource_only_fallback(self):
        from utils.ch3_m_execution import resource_fallback
        self.assertFalse(resource_fallback(dict(failure='numeric gate failure',returncodes=[0],resource=True)))
        self.assertFalse(resource_fallback(dict(failure='scheduler identity',returncodes=[0],resource=True)))

    def test_safe_stop_matches_owned_group_action(self):
        with tempfile.TemporaryDirectory(dir=s.PACKAGE,prefix='fixture-stop-') as tmp:
            root=Path(tmp);owner=dict(pid=111,start_ticks='1');child=dict(pid=222,start_ticks='2')
            args=['python',str(s.ROOT/'m6_baseline_unified_entry.py'),'group-child','--stage','MS']
            (root/'controller.json').write_text(json.dumps({'owner':owner}))
            (root/'current.json').write_text(json.dumps(dict(owner=owner,child=child,command=args)))
            (root/'cmdline').write_bytes(('\0'.join(args)+'\0').encode());(root/'stat').write_text('222 (python) S 111 0')
            def path(*parts):
                if parts[0]=='/proc':return root/parts[-1]
                return Path(*parts)
            with patch.object(q,'CONTROL',root),patch.object(q,'Path',side_effect=path),patch.object(q,'same',return_value=True),patch('utils.ch3_m_launch.command_has',return_value=True),patch.object(q,'signal_owned',return_value=True) as signal:
                result=q.safe_stop();self.assertTrue(result['STOP']);self.assertEqual([x.args[0] for x in signal.call_args_list],[child,owner])
                signal.reset_mock();(root/'cmdline').write_bytes(b'foreign\0')
                with self.assertRaises(PermissionError):q.safe_stop()
                signal.assert_not_called()


if __name__=='__main__':unittest.main()
