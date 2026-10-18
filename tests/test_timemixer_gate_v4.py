"""Pure saved-metric v2 gates: goal1 per fold, goal3 pooled, no training."""
import copy
import json
import math
import unittest
from unittest.mock import patch
from tests.timemixer_v4_fixtures import STORED, ROOT
from timemixer_dev_v4.summary import aggregate, DOMAINS
from timemixer_dev_v4.gate_policy import policy, validate_policy
from timemixer_dev_v4.training_config import validate
from timemixer_dev_v4.worker import source_binding, successful_identity
from timemixer_dev_v4.launcher import batch, validate_authorization


def saved_fixture():
    c=copy.deepcopy(STORED)
    for b in c['existing_B'].values():b.update(mse=10.,mae=10.)
    records=[]
    for t in c['tasks']:
        v=({'U':6.,'V':7.,'UV':3.}[t['arm']] if t['goal']==1 else
           {'F1':9.,'F2':6.,'F3':7.}[t['input_variant']] if t['goal']==3 else 5.)
        records.append((copy.deepcopy(t),dict(task_id=t['id'],mse=v,mae=v)))
    return c,records


class Gates(unittest.TestCase):
    def test_full77_separate_numeric_engineering_review_and_F4(self):
        c,rows=saved_fixture();r=aggregate(c,rows)
        self.assertTrue(r['coverage_complete']);self.assertTrue(r['numerical_conditions_met'])
        self.assertEqual(r['scientific_gate'],'Pending result review');self.assertEqual(r['result_review'],'Pending')
        self.assertEqual(r['Urban_inputs']['F4']['count'],8)
        self.assertEqual(set(r['Urban_inputs']['F4']['task_ids']),set(c['F4_reuse'].values()))
        self.assertFalse(r['Urban_module_pooled_is_gate'])
        for metric in ('mse','mae'):
            self.assertEqual(set(r['goals']['goal1']['metrics'][metric]['folds']),{'1','4'})

    def test_pooled_pass_but_one_fold_fails_must_fail_goal1(self):
        c,rows=saved_fixture()
        for t,m in rows:
            if t['goal']==1:
                v=({'U':1.,'V':2.,'UV':.5} if t['fold']==1 else {'U':11.,'V':12.,'UV':3.})[t['arm']]
                m.update(mse=v,mae=v)
        r=aggregate(c,rows);pooled=r['Urban_module']
        self.assertLess(pooled['UV']['mean']['mse'],pooled['U']['mean']['mse']);self.assertLess(pooled['V']['mean']['mse'],10.)
        for metric in ('mse','mae'):
            g=r['goals']['goal1']['metrics'][metric]
            self.assertTrue(g['folds']['1']['conditions_met']);self.assertFalse(g['folds']['4']['conditions_met']);self.assertFalse(g['conditions_met'])
        self.assertFalse(r['goals']['goal1']['numerical_conditions_met'])

    def test_goal1_each_metric_can_independently_fail(self):
        for metric in ('mse','mae'):
            c,rows=saved_fixture()
            for t,m in rows:
                if t['goal']==1 and t['arm']=='UV' and t['fold']==4:m[metric]=7.
            g=aggregate(c,rows)['goals']['goal1']
            self.assertFalse(g['metrics'][metric]['conditions_met']);self.assertTrue(g['metrics']['mae' if metric=='mse' else 'mse']['conditions_met'])
            self.assertFalse(g['numerical_conditions_met'])

    def test_goal3_keeps_eight_cell_average_and_independent_metric(self):
        c,rows=saved_fixture()
        for t,m in rows:
            if t['goal']==3 and t['input_variant']=='F2':m.update(mse=1. if t['fold']==1 else 8.,mae=1. if t['fold']==1 else 8.)
        g=aggregate(c,rows)['goals']['goal3']
        self.assertTrue(g['numerical_conditions_met']);self.assertEqual(g['metrics']['mse']['means']['F2'],4.5)
        for t,m in rows:
            if t['goal']==3 and t['input_variant']=='F2':m['mse']=2.
        g=aggregate(c,rows)['goals']['goal3'];self.assertFalse(g['metrics']['mse']['conditions_met']);self.assertTrue(g['metrics']['mae']['conditions_met'])

    def test_scheme_A_separate_winners_8_8_and_8_7_7_8(self):
        for nmse,nmae in ((8,8),(8,7),(7,8)):
            c,rows=saved_fixture();a=set(DOMAINS[:nmse]);b=set(DOMAINS[11-nmae:])
            for t,m in rows:
                if t['goal']==2:m.update(mse=5. if t['dataset'] in a else 10.,mae=5. if t['dataset'] in b else 10.)
            g=aggregate(c,rows)['goals']['goal2'];self.assertEqual(g['numerical_conditions_met'],nmse==nmae==8)
            self.assertEqual(g['metrics']['mse']['winner_count'],nmse);self.assertEqual(g['metrics']['mae']['winner_count'],nmae)
            if nmse==nmae==8:self.assertEqual(len(set(g['metrics']['mse']['winners'])&set(g['metrics']['mae']['winners'])),5)

    def test_unrounded_strict_equality_and_equal_H_weights(self):
        c,rows=saved_fixture()
        for t,m in rows:
            if t['goal']==1 and t['arm']=='U':m['mse']=float(t['h'])
            if t['goal']==2:m.update(mse=10.,mae=10.)
            if t['goal']==2 and t['dataset']=='DE':m['mse']=math.nextafter(10.,0.)
        r=aggregate(c,rows)
        for f in ('1','4'):self.assertEqual(r['goals']['goal1']['metrics']['mse']['folds'][f]['means']['U'],7.5)
        self.assertEqual(r['goals']['goal2']['metrics']['mse']['winners'],['DE']);self.assertEqual(r['goals']['goal2']['metrics']['mae']['winner_count'],0)

    def test_first24_only_goal1_missing_H_duplicate_and_wrong_version(self):
        c,rows=saved_fixture();first=[(t,m) for t,m in rows if t['goal']==1];r=aggregate(c,first)
        self.assertTrue(r['goals']['goal1']['numerical_conditions_met'])
        for goal in ('goal2','goal3'):self.assertEqual(r['goals'][goal]['numerical_status'],'Not evaluated')
        self.assertEqual(r['completed'],24);self.assertEqual(r['scientific_gate'],'Not evaluated')
        self.assertFalse(aggregate(c,first[1:])['goals']['goal1']['coverage_complete'])
        with self.assertRaises(ValueError):aggregate(c,first+[first[0]])
        bad=copy.deepcopy(first);bad[0][0]['candidate_id']='timemixer-uv-dev-v1'
        with self.assertRaises(ValueError):aggregate(c,bad)
        bad=copy.deepcopy(first);bad[0][1]['task_id']=bad[0][1]['task_id'].replace('-v4','-v2')
        with self.assertRaises(ValueError):aggregate(c,bad)

    def test_exact_same_version_F4_mapping_and_B_pairing(self):
        c,rows=saved_fixture();keys=list(c['F4_reuse']);c['F4_reuse'][keys[0]],c['F4_reuse'][keys[1]]=c['F4_reuse'][keys[1]],c['F4_reuse'][keys[0]]
        with self.assertRaisesRegex(ValueError,'fold/H'):validate(c)
        c,rows=saved_fixture();c['F4_reuse']['f1-H3']=c['F4_reuse']['f1-H3'].replace('-v4','-v2')
        with self.assertRaises(ValueError):validate(c)
        c,rows=saved_fixture();c['tasks'][0]['baseline_pair_cell_id']=c['tasks'][3]['baseline_pair_cell_id']
        with self.assertRaises(ValueError):aggregate(c,rows)


class Binding(unittest.TestCase):
    def test_stored_resolved_science_77_and_6_wave_source_identity(self):
        validate(STORED);self.assertEqual(STORED['source_binding'],source_binding())
        old=json.loads((ROOT/'configs/timemixer_uv_round2_resolved_v3.json').read_text())
        self.assertEqual(len(STORED['tasks']),77);self.assertEqual(len(STORED['existing_B']),37);self.assertEqual(STORED['existing_B'],old['existing_B'])
        allowed={'id','model','candidate_id','structural_revision'}
        for a,b in zip(old['tasks'],STORED['tasks']):
            self.assertEqual({k:v for k,v in a.items() if k not in allowed},{k:v for k,v in b.items() if k not in allowed})
        self.assertEqual(sum(t['arithmetic']['max_epochs'] for t in STORED['tasks']),1340)
        ids=[t['id'] for t in STORED['tasks'] if t['goal']==1];waves=batch(STORED,ids)
        self.assertEqual(len(ids),24);self.assertEqual(len(waves),6)
        self.assertTrue(all(len(w)==4 and len({next(t['arm'] for t in STORED['tasks'] if t['id']==i) for i in w})==1 for w in waves))
        self.assertEqual(sum(t['arithmetic']['max_epochs'] for t in STORED['tasks'] if t['id'] in ids),480)
        self.assertFalse(STORED['execution_authorized'])

    def test_launch_old_version_policy_or_unreviewed_rejected(self):
        from timemixer_dev.gate_policy import policy as old_policy
        ids=[t['id'] for t in STORED['tasks'] if t['goal']==1]
        for auth in ({'schema':'timemixer-uv-launch-v2','review_approved':False},
                     {'schema':'timemixer-uv-launch-v2','review_approved':True,'candidate_id':'timemixer-uv-dev-v1'},
                     {'schema':'timemixer-uv-launch-v2','review_approved':True,'candidate_id':STORED['candidate_id'],'structural_revision':STORED['structural_revision'],'gate_policy':old_policy()}):
            with self.assertRaises(PermissionError):validate_authorization(STORED,'unused',ids,auth)
        with self.assertRaises(ValueError):validate_policy(old_policy())


if __name__=='__main__':unittest.main(verbosity=2)
