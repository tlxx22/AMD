"""Explicit synthetic retirement/combination gates; no model or GPU execution."""
import copy
import json
import os
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from utils import ch3_ms_retirement as retire
from utils import ch3_native_tasks as scope
from utils import ch3_native_chain as chain
from utils import ch3_native_execution as execution
from utils import ch3_m_tasks as source
from utils.ch3_m_launch import exclusive_json
from utils.ch3_contract import profile,digest
from tests.test_m6_native_chain import SavedProbeTests


class RetirementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=scope.parent_config();cls.receipt=source.bound(retire.RECEIPT)
        cls.inventory=source.bound(cls.receipt['run_inventory']);cls.cs=chain.configs()
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=scope.PACKAGE/'fixtures');self.root=Path(self.tmp.name);self.addCleanup(self.tmp.cleanup)
        self.r=copy.deepcopy(self.receipt);self.i=copy.deepcopy(self.inventory)
    def validate(self):return retire.validate_partition(self.r,self.i,self.original)

    def test_exact_552_partition_520_completed_32_retired(self):
        self.validate();self.assertEqual(len(self.r['completed_task_ids']),520);self.assertEqual(len(self.r['retired_task_ids']),32)
        self.assertEqual(len(self.r['N_completed']),16);self.assertEqual(len(self.r['incomplete_partial_N_task_ids']),4);self.assertEqual(len(self.r['N_not_started']),4)
        self.assertFalse(self.r['technical_complete']);self.assertFalse(self.r['budget_refund']);self.assertFalse(self.r['scientific_failure'])

    def test_partial_N_cannot_be_marked_Passed(self):
        run=self.r['incomplete_partial_N_task_ids'][0];self.i['runs'][run]['completed']=True
        with self.assertRaises(ValueError):self.validate()

    def test_S_must_be_not_executed(self):
        self.i['runs'][self.r['S_not_executed'][0]]['exists']=True
        with self.assertRaises(ValueError):self.validate()

    def test_missing_foreign_duplicate_count_rejected(self):
        for key in ('original_task_ids','completed_task_ids','retired_task_ids'):
            with self.subTest(key=key):
                self.r=copy.deepcopy(self.receipt);self.r[key]=self.r[key]+[self.r[key][0]]
                with self.assertRaises(ValueError):self.validate()

    def test_spent_partial_cost_no_refund_or_epoch_rounding(self):
        self.validate();spent=self.r['consumption']['recorded_N']
        self.assertEqual(spent,dict(adam=575959,forward=646897,backward=575959,recorded_completed_epochs=172))
        row=self.i['runs'][self.r['incomplete_partial_N_task_ids'][0]]
        self.assertGreater(row['counts']['adam'],row['last_history']['steps'])
        for key,value in [('budget_refund',True),('technical_complete',True),('scientific_failure',True)]:
            self.r=copy.deepcopy(self.receipt);self.r[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.validate()
        self.r=copy.deepcopy(self.receipt);self.r['consumption']['recorded_N']['adam']=0
        with self.assertRaises(ValueError):self.validate()

    def test_exact_receipt_SHA_and_changed_bytes_rejected_before_controller(self):
        p=self.root/'receipt.json';ip=self.root/'inventory.json';exclusive_json(ip,self.i)
        self.r['run_inventory']=source.ref(ip);exclusive_json(p,self.r);ref=source.ref(p)
        with patch.object(retire,'RECEIPT',ref):self.assertEqual(retire.validate_receipt(worker=True),self.r)
        p.write_text(p.read_text()+' ')
        with patch.object(retire,'RECEIPT',ref),patch.object(retire,'same')as controller:
            with self.assertRaises(ValueError):retire.validate_receipt()
            controller.assert_not_called()

    def test_ordinary_STOP_or_v1_evidence_cannot_authorize_v2(self):
        for kind in ('ordinary_STOP','old_MS_queue_technical_complete','old_M_waiter_retirement'):
            self.r=copy.deepcopy(self.receipt);self.r['kind']=kind
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.validate()
        p=self.root/'missing-receipt.json'
        with patch.object(retire,'RECEIPT',dict(path=str(p),sha256='0'*64)):
            with self.assertRaises(OSError):retire.validate_receipt()

    def test_retirement_boundary_not_old_technical_complete(self):
        with patch.object(retire,'validate_receipt',return_value=self.r):b=retire.boundary()
        self.assertEqual(b['kind'],retire.KIND);self.assertFalse(b['technical_complete']);self.assertTrue(b['retirement_accepted']);self.assertEqual(b['retirement_receipt'],retire.RECEIPT)
        p=self.root/'boundary.json';exclusive_json(p,b)
        with patch.object(chain,'old_path',return_value=p),patch.object(chain,'retirement_boundary',return_value=b):chain.check_boundary(self.cs['m'],source.ref(p))
        b['technical_complete']=True;p.write_text(json.dumps(b))
        with patch.object(chain,'old_path',return_value=p),patch.object(chain,'retirement_boundary',return_value=retire.boundary(worker=True)):
            with self.assertRaises(ValueError):chain.check_boundary(self.cs['m'],source.ref(p))

    def test_v3_namespace_ignores_v1_v2_without_overwriting(self):
        self.assertEqual(scope.ID,'m6-native-tmark-chain-v4');self.assertNotIn('chain-v1',str(chain.ROOT_CONTROL));self.assertNotIn('chain-v2',str(chain.ROOT_CONTROL))
        self.assertNotIn('chain-v1',str(chain.LOG));self.assertNotIn('chain-v2',str(chain.LOG));self.assertEqual(chain.SESSION,'ch3-native-time-mark-chain-v4')
        with patch.object(chain,'configs',return_value=self.cs),patch.object(chain,'binding',return_value={}),patch.object(chain,'retirement_boundary',return_value={}),patch.object(chain.subprocess,'run',return_value=subprocess.CompletedProcess([],1)):
            self.assertEqual(chain.preflight(),[])
        for c in self.cs.values():
            a=dict(successor_scope='m6-native-tmark-chain-v1-tmark-probe',reviewed=True)
            with patch.object(chain,'binding',return_value={}):
                with self.assertRaises(ValueError):chain.validate_permit(c,a,True)

    def test_87_and_M_science_and_M_numeric_unchanged(self):
        for path in ('configs/ch3_formal_profiles.json','models/ch3_adapter.py','utils/ch3_time_marks.py','utils/ch3_m_tasks.py'):
            old=subprocess.check_output(['git','show',scope.BASE+':'+path],cwd=scope.ROOT)
            self.assertEqual(old,(scope.ROOT/path).read_bytes())
        old=json.loads(subprocess.check_output(['git','show',scope.BASE+':configs/ch3_native_time_mark_profiles.json'],cwd=scope.ROOT))
        current=copy.deepcopy(self.cs['tmark']);old['native_replacement'].pop('numeric_policies');current['native_replacement'].pop('numeric_policies')
        self.assertEqual(old,current)
        self.assertEqual(len(self.cs['tmark']['tasks']),87);self.assertEqual(len(self.cs['m']['tasks']),84)

    def test_actual_concurrent_combo_coverage_and_exact_costs(self):
        c=self.cs['tmark'];p=scope.plan(c);gs=p['groups']
        self.assertEqual((p['computational_identities'],len(gs),p['representative_tasks']),(7,7,25))
        self.assertEqual((p['nominal_workers'],p['max_workers']),(50,73))
        self.assertEqual(p['nominal_cost'],dict(adam=300,forward=416,backward=300));self.assertEqual(p['caps'],dict(adam=438,forward=608,backward=438))
        covered=[r for g in gs for ids in g['coverage'].values() for r in ids]
        self.assertEqual(len(covered),len(set(covered)));self.assertEqual(set(covered),{t['id'] for t in c['tasks']})
        for g in gs:
            self.assertGreater(g['planned_q'],1)
            for run in g['representatives']:self.assertIn(run,g['coverage'])
        epf=[g for g in gs if '-EPF-' in g['id']]
        self.assertEqual([len(g['representatives']) for g in epf],[4,4,3,2])
        for g in epf:self.assertGreater(len({scope.task_by_run(c,r)['dataset'] for r in g['representatives']}),1)
        tx=[g for g in epf if g['model']=='TimeXer']
        self.assertEqual([{profile(c,scope.task_by_run(c,r))['training']['batch'] for r in g['representatives']} for g in tx],[{16},{4}])

    def test_formal_cross_market_waves_and_fallback_are_exact(self):
        c=self.cs['tmark'];gs=scope.probe_groups(c)
        for q in (4,2,1):
            report=dict(decisions={g['id']:dict(concurrency=min(q,g['planned_q'])) for g in gs})
            all_ids=[]
            for model in scope.MODELS:
                waves=scope.formal_waves(c,report,model);all_ids+=[r for wave in waves for r in wave]
                if q>1:self.assertTrue(any(len(w)>1 and len({scope.task_by_run(c,r)['dataset'] for r in w})>1 for w in waves))
            self.assertEqual(len(all_ids),87);self.assertEqual(len(set(all_ids)),87)
        for g in gs:
            self.assertEqual(scope.attempt_widths(g),(4,2) if g['planned_q']==4 else (2,))
        bad=dict(decisions={g['id']:dict(concurrency=4) for g in gs})
        with self.assertRaises(ValueError):scope.formal_waves(c,bad,'TimeXer')

    def test_M_21_groups_and_budget_unchanged(self):
        c=self.cs['m'];p=scope.plan(c)
        self.assertEqual(len(p['groups']),21);self.assertEqual((p['nominal_workers'],p['max_workers']),(168,252))
        self.assertEqual(p['nominal_cost'],dict(adam=1008,forward=1344,backward=1008));self.assertEqual(p['caps'],dict(adam=1512,forward=2016,backward=1512))
        self.assertEqual((p['run_budget'],p['max_optimizer_steps']),(dict(runs=84,run_epochs=840),294790))

    def test_worker_endpoint_scope_preserves_local_10_and_other_8(self):
        c=self.cs['tmark'];scope_id=scope.context(c)['probe_scope']
        for t in c['tasks']:
            s=dict(purpose='ch3_probe',successor_scope=scope_id,task=t['id'])
            expected=t['model']=='TimeMixer' and t['dataset']=='UrbanEV'
            self.assertEqual(execution.confirmation_endpoint(c,s),expected)
            self.assertEqual(scope.worker_counts(c,t)['forward'],10 if expected else 8)
            for wrong in ('m6-native-tmark-chain-v1-tmark-probe',scope.context(self.cs['m'])['probe_scope'],scope.context(c)['formal_scope']):
                self.assertFalse(execution.confirmation_endpoint(c,dict(s,successor_scope=wrong)))
        self.assertFalse(execution.confirmation_endpoint(self.cs['m'],dict(purpose='ch3_probe',successor_scope=scope.context(self.cs['m'])['probe_scope'],task=self.cs['m']['tasks'][0]['id'])))

    def test_guarded_positive_config_chain_and_v1_rejection_before_output(self):
        sys.path.insert(0,str(scope.ROOT/'tools/restricted_regression'));import m5_formal_entry
        c=self.cs['tmark'];ctx=scope.context(c);ctx['probe_root']=self.root/'probe'
        task=next(t for t in c['tasks'] if t['model']=='TimeMixer' and t['dataset']=='UrbanEV' and t['fold']==1)
        template=chain.approval_template(c,True)
        bound={k:template[k] for k in ('code','protocol_sha','environment','hardware','plan','data_binding_artifact','data_bindings','source_states')};bound['commit']='explicit-synthetic-closure'
        permit=self.root/'fixture-permit.json';out=ctx['probe_root']/execution.group_for(c,task['id'])['id']/'serial'/task['id']
        with patch.object(scope,'context',return_value=ctx),patch.object(chain,'ROOT_CONTROL',self.root),patch.object(chain,'binding',return_value=bound),patch.object(chain,'permit_path',return_value=permit),patch.object(chain,'check_boundary'),patch.object(chain,'readiness',side_effect=lambda c,a,probe:execution.authorization_reasons(c,a,probe)),patch.object(execution,'gpu_environment_reasons',return_value=[]),patch.dict(os.environ,{'TMPDIR':str(scope.PACKAGE/'fixtures')}):
            a=source.bound(chain.create_permit(c,True,dict(path='synthetic-boundary',sha256='fixture')))
            cfg=m5_formal_entry.make_config(c,'ch3_probe',out,task=task['id'],approval=a)
            execution.validate_worker(c,cfg);self.assertEqual(cfg['limits'],dict(adam=6,forward=10,backward=6,seconds=1800));self.assertEqual(cfg['prefix_files'],{})
            wrong=dict(a,successor_scope='m6-native-tmark-chain-v1-tmark-probe')
            other=ctx['probe_root']/execution.group_for(c,task['id'])['id']/'q4'/task['id']
            with self.assertRaises(PermissionError):m5_formal_entry.make_config(c,'ch3_probe',other,task=task['id'],approval=wrong)
            self.assertFalse(other.exists());self.assertEqual(json.loads((out/'budget.json').read_text())['counts'],dict(adam=0,forward=0,backward=0))


class Q2CombinationTests(SavedProbeTests):
    def test_initial_q2_resource_fail_uses_legal_serial_without_q4_retry(self):
        gs=[dict(g,planned_q=2) for g in scope.probe_groups(self.c)]
        stack,attempts=self.engine({'q2':'resource'})
        with stack,patch.object(scope,'probe_groups',return_value=gs):
            r=execution.run_probe(self.c,self.a);d=next(iter(r['decisions'].values()))
            self.assertEqual(d['concurrency'],1);self.assertEqual([x['q'] for x in d['attempts']],[2]);self.assertNotIn('q4',attempts)
            self.assertFalse(r['budget']['refund']);execution.validate_probe_completion(self.c,r)

    def test_initial_q2_nonresource_fail_cannot_use_q1(self):
        gs=[dict(g,planned_q=2) for g in scope.probe_groups(self.c)]
        stack,_=self.engine({'q2':'business'})
        with stack,patch.object(scope,'probe_groups',return_value=gs):
            with self.assertRaises(RuntimeError):execution.run_probe(self.c,self.a)
            self.assertFalse((self.ctx['probe_root']/'complete.json').exists())
