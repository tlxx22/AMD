"""Frozen science/source/identity checks; no real data or model inference."""
import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
from tests import timemixer_v6_r3_fixtures as f
from timemixer_dev_v6_r3 import source,launcher,worker
from timemixer_dev_v6_r3.training_config import validate
from timemixer_dev_v6_r3.summary import aggregate,DOMAINS


def policy_records(config):
    rows=[]
    for t in config['tasks']:
        if t['goal']==3:continue
        b=config['existing_B'][t['baseline_pair_cell_id']]
        if t['goal']==1:
            ratio={'U':.3,'V':.4,'UV':.2}[t['arm']];mse=b['mse']*ratio;mae=b['mae']*ratio
        else:
            i=DOMAINS.index(t['dataset']);mse=b['mse']*(.9 if i<8 else 1.1);mae=b['mae']*(.9 if i>=3 else 1.1)
        rows.append((t,dict(task_id=t['id'],mse=mse,mae=mae,synthetic_metadata_fixture=True)))
    F4={metric:sum(m[metric] for t,m in rows if t['goal']==1 and t['arm']=='UV')/8 for metric in ('mse','mae')}
    for t in config['tasks']:
        if t['goal']==3:
            ratio={'F1':4,'F2':2,'F3':3}[t['input_variant']];rows.append((t,dict(task_id=t['id'],mse=F4['mse']*ratio,mae=F4['mae']*ratio,synthetic_metadata_fixture=True)))
    return rows


class Delivery(unittest.TestCase):
    def test_77_scientific_fields_B_pairs_F4_and_current_source_binding(self):
        old=json.loads((f.ROOT/'configs/timemixer_uv_round2_resolved_v6.json').read_text());new=validate(copy.deepcopy(f.STORED));self.assertEqual(new['source_binding'],worker.source_binding());self.assertEqual(len(new['existing_B']),37);self.assertEqual(new['existing_B'],old['existing_B']);self.assertEqual(len(new['F4_reuse']),8)
        mapping={}
        for a,b in zip(old['tasks'],new['tasks']):
            self.assertEqual(a.keys(),b.keys())
            for key in a:
                if key not in ('id','structural_revision'):self.assertEqual(a[key],b[key],key)
            self.assertEqual(b['structural_revision'],f.REVISION);self.assertIn('|r3|',b['id']);mapping[a['id']]=b['id']
        self.assertEqual(new['F4_reuse'],{k:mapping[v] for k,v in old['F4_reuse'].items()});self.assertEqual(new['gate_policy'],old['gate_policy'])
        self.assertEqual({g:sum(t['goal']==g for t in new['tasks']) for g in (1,3,2)},{1:24,3:24,2:29})
        self.assertEqual(sum(t['arithmetic']['max_epochs'] for t in new['tasks']),1340)
    def test_first24_order_exact_r2_mapping_six_waves(self):
        runtime=f.ROOT.parent/'amd-execution-evidence/m6/timemixer-uv-v6-development-runs/urbanev-f4-f1f4-uv-v6-first-batch-v1/urbanev-f4-f1f4-v6-first-r1/runtime-24486fc11ed440a8a0289f5d8735c05d.json';old=json.loads(runtime.read_text());ids=[s.replace(f.CANDIDATE_ID+'|',f.CANDIDATE_ID+'|r3|',1) for s in old['task_ids']];waves=launcher.batch(f.STORED,ids);self.assertEqual(len(ids),24);self.assertEqual(len(waves),6);self.assertEqual([s for group in waves for s in group],ids);self.assertTrue(all(len(group)==4 for group in waves));self.assertEqual(waves,[[s.replace(f.CANDIDATE_ID+'|',f.CANDIDATE_ID+'|r3|',1) for s in group] for group in old['waves']])
    def test_fixed_source_real_materials_and_mock_corrupt_material_rejected(self):
        verified=source.verify_training_materials();self.assertEqual(set(verified),{'XLinear','TimeBridge'});src=source.binding();self.assertEqual(src['XLinear']['commit'],'8398e656e945da84a79e8b2f64af8f3f68842e19');self.assertEqual(src['TimeBridge']['commit'],'0f9a83fbc3e1260c9ddd527c522dff0ce4b9554b')
        for kind in ('status','path','sha256'):
            bad=copy.deepcopy(src)
            if kind=='status':bad['XLinear']['paper']['status']='Pending'
            elif kind=='path':bad['XLinear']['paper'].pop('path')
            else:bad['XLinear']['paper']['sha256']='0'*64
            with patch.object(source,'binding',return_value=bad),self.assertRaises(PermissionError):source.verify_training_materials()
    def test_old_same_candidate_config_and_grant_and_unreviewed_grant_rejected(self):
        old=json.loads((f.ROOT/'configs/timemixer_uv_round2_resolved_v6.json').read_text())
        with self.assertRaises(ValueError):validate(old)
        ids=[t['id'] for t in f.STORED['tasks'] if t['goal']==1]
        for grant in [dict(schema='timemixer-uv-launch-v2',review_approved=False,candidate_id=f.CANDIDATE_ID,structural_revision=f.REVISION),dict(schema='timemixer-uv-launch-v2',review_approved=True,candidate_id=f.CANDIDATE_ID,structural_revision=old['structural_revision'])]:
            with patch.object(launcher,'git',side_effect=AssertionError('identity must reject before Git/network')),patch.object(launcher,'gpu_startup',side_effect=AssertionError('no GPU operation')),self.assertRaises(PermissionError):launcher.validate_authorization(f.STORED,'unused',ids,grant)
    def test_three_goals_foldwise_dual_A_and_eight_cell_policy(self):
        rows=policy_records(f.STORED);report=aggregate(f.STORED,rows);self.assertTrue(report['coverage_complete']);self.assertTrue(report['goals']['goal1']['numerical_conditions_met']);self.assertTrue(report['goals']['goal3']['numerical_conditions_met']);g2=report['goals']['goal2']['metrics'];self.assertEqual((g2['mse']['winner_count'],g2['mae']['winner_count']),(8,8));self.assertNotEqual(g2['mse']['winners'],g2['mae']['winners'])
        bad=copy.deepcopy(rows)
        for t,m in bad:
            if t['goal']==1 and t['fold']==4 and t['arm']=='U':
                base=f.STORED['existing_B'][t['baseline_pair_cell_id']];m.update(mse=base['mse']*1.1,mae=base['mae']*1.1)
        result=aggregate(f.STORED,bad)['goals']['goal1'];self.assertFalse(result['numerical_conditions_met']);self.assertTrue(result['metrics']['mse']['folds']['1']['conditions_met']);self.assertFalse(result['metrics']['mse']['folds']['4']['conditions_met'])
    def test_missing_duplicate_wrong_revision_and_old_F4_refused(self):
        rows=policy_records(f.STORED);self.assertFalse(aggregate(f.STORED,rows[1:])['goals']['goal1']['coverage_complete'])
        with self.assertRaises(ValueError):aggregate(f.STORED,rows+[rows[0]])
        bad=copy.deepcopy(rows);bad[0][0]['structural_revision']=dict(u='r2',v='r2',fusion='old')
        with self.assertRaises(ValueError):aggregate(f.STORED,bad)
        cfg=copy.deepcopy(f.STORED);key=next(iter(cfg['F4_reuse']));cfg['F4_reuse'][key]=cfg['F4_reuse'][key].replace('|r3|','|')
        with self.assertRaises(ValueError):validate(cfg)

if __name__=='__main__':unittest.main(verbosity=2)
