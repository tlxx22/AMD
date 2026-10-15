"""Saved synthetic metrics and bounded CPU fixtures for the dual gate contract."""
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from tests import test_timemixer_training as legacy
from timemixer_dev import launcher, worker
from timemixer_dev.gate_policy import policy, validate_policy
from timemixer_dev.summary import aggregate, summarize, DOMAINS
from timemixer_dev.training_config import validate, digest
from timemixer_dev.training_state import file_sha, read

ROOT = legacy.ROOT
STORED = json.loads((ROOT/'configs/timemixer_uv_round2_resolved_v1.json').read_text())
# Reuse the stored, reviewed task profiles, without deriving or auditing old runs.
legacy.REAL = STORED

def saved_fixture():
    config = copy.deepcopy(STORED)
    for b in config['existing_B'].values():
        b.update(mse=10., mae=10.)
    rows = []
    for t in config['tasks']:
        if t['goal']==1:
            value = {'U':6.,'V':7.,'UV':3.}[t['arm']]
        elif t['goal']==3:
            value = {'F1':9.,'F2':6.,'F3':7.}[t['input_variant']]
        else:
            value = 5.
        rows.append((copy.deepcopy(t), dict(task_id=t['id'],mse=value,mae=value)))
    return config, rows

class SavedMetrics(unittest.TestCase):
    def test_complete77_both_metrics_and_result_review_separate(self):
        c, rows = saved_fixture()
        # Actual save/read of metric-only fixtures, never formal artifacts.
        with tempfile.TemporaryDirectory(prefix='.uv-runner-',dir=ROOT) as d:
            p=Path(d)/'synthetic-saved-metrics.json';p.write_text(json.dumps(rows))
            result=aggregate(c,json.loads(p.read_text()))
        self.assertTrue(result['coverage_complete']);self.assertTrue(result['numerical_conditions_met'])
        self.assertEqual(result['engineering_completion'],'Complete')
        self.assertEqual(result['scientific_gate'],'Pending result review');self.assertEqual(result['result_review'],'Pending')
        for name in ('goal1','goal2','goal3'):
            self.assertTrue(result['goals'][name]['numerical_conditions_met'])
            self.assertEqual(set(result['goals'][name]['metrics']),{'mse','mae'})
        self.assertEqual(result['Urban_inputs']['F4']['count'],8)

    def test_goal1_mse_pass_mae_fail(self):
        c,rows=saved_fixture()
        for t,m in rows:
            if t['goal']==1:m['mae']=9.
        g=aggregate(c,rows)['goals']['goal1']
        self.assertTrue(g['metrics']['mse']['conditions_met']);self.assertFalse(g['metrics']['mae']['conditions_met']);self.assertFalse(g['numerical_conditions_met'])

    def test_goal3_mae_pass_mse_fail(self):
        c,rows=saved_fixture()
        for t,m in rows:
            if t['goal']==3 and t['input_variant']=='F2':m['mse']=2.
        g=aggregate(c,rows)['goals']['goal3']
        self.assertTrue(g['metrics']['mae']['conditions_met']);self.assertFalse(g['metrics']['mse']['conditions_met']);self.assertFalse(g['numerical_conditions_met'])

    def goal2(self,nmse,nmae):
        c,rows=saved_fixture();a=set(DOMAINS[:nmse]);b=set(DOMAINS[11-nmae:])
        for t,m in rows:
            if t['goal']==2:m.update(mse=5. if t['dataset'] in a else 10.,mae=5. if t['dataset'] in b else 10.)
        return aggregate(c,rows)['goals']['goal2']

    def test_goal2_A_independent8_8_intersection5(self):
        g=self.goal2(8,8);m=g['metrics']
        self.assertTrue(g['numerical_conditions_met'])
        self.assertEqual(m['mse']['winner_count'],8);self.assertEqual(m['mae']['winner_count'],8)
        self.assertNotEqual(m['mse']['winners'],m['mae']['winners'])
        self.assertEqual(len(set(m['mse']['winners'])&set(m['mae']['winners'])),5)

    def test_goal2_8_7_fails(self):
        self.assertFalse(self.goal2(8,7)['numerical_conditions_met'])

    def test_goal2_7_8_fails(self):
        self.assertFalse(self.goal2(7,8)['numerical_conditions_met'])

    def test_equal_not_improved_and_unrounded_strict_difference(self):
        c,rows=saved_fixture()
        for t,m in rows:
            if t['goal']==2:m.update(mse=10.,mae=10.)
            if t['goal']==1:m.update(mse=6.,mae=6.)
        r=aggregate(c,rows)
        self.assertFalse(r['goals']['goal1']['numerical_conditions_met'])
        self.assertEqual(r['goals']['goal2']['metrics']['mse']['winner_count'],0)
        for t,m in rows:
            if t['goal']==2 and t['dataset']=='DE':m.update(mse=math.nextafter(10.,0.),mae=math.nextafter(10.,0.))
        r=aggregate(c,rows)
        self.assertEqual(r['goals']['goal2']['metrics']['mse']['winners'],['DE'])
        self.assertEqual(r['goals']['goal2']['metrics']['mae']['winners'],['DE'])

    def test_first24_only_goal1_evaluated_and_exact_F4_reuse(self):
        c,rows=saved_fixture();rows=[(t,m) for t,m in rows if t['goal']==1];r=aggregate(c,rows)
        self.assertEqual(r['completed'],24);self.assertTrue(r['goals']['goal1']['numerical_conditions_met'])
        for g in ('goal2','goal3'):
            self.assertEqual(r['goals'][g]['numerical_status'],'Not evaluated');self.assertIsNone(r['goals'][g]['numerical_conditions_met'])
        self.assertFalse(r['coverage_complete']);self.assertIsNone(r['numerical_conditions_met']);self.assertEqual(r['scientific_gate'],'Not evaluated')
        self.assertEqual(set(r['Urban_inputs']['F4']['task_ids']),set(c['F4_reuse'].values()))

    def test_missing_exact_task_and_forged_count_cannot_complete(self):
        c,rows=saved_fixture();r=aggregate(c,rows[1:]);self.assertFalse(r['goals']['goal1']['coverage_complete'])
        self.assertIsNone(r['numerical_conditions_met'])
        with self.assertRaisesRegex(ValueError,'duplicate'):aggregate(c,rows[1:]+[rows[1]])
        altered=copy.deepcopy(rows);altered[0][0]['fold']=4
        with self.assertRaisesRegex(ValueError,'exact summary task'):aggregate(c,altered)
        altered=copy.deepcopy(c);altered['tasks'][0]['id']='fabricated-but-still-77'
        with self.assertRaises(ValueError):aggregate(altered,rows)

    def test_wrong_B_pair_and_profile_rejected(self):
        c,rows=saved_fixture();t=c['tasks'][0];t['baseline_pair_cell_id']=c['tasks'][3]['baseline_pair_cell_id']
        with self.assertRaises(ValueError):aggregate(c,rows)
        c,rows=saved_fixture();key=c['tasks'][0]['baseline_cell_id'];c['existing_B'][key]['profile_sha']='wrong'
        with self.assertRaisesRegex(ValueError,'B paired profile'):aggregate(c,rows)

    def test_unequal_H_values_use_equal_cell_weights(self):
        c,rows=saved_fixture()
        for t,m in rows:
            if t['goal']==2 and t['dataset']=='ETTh1':m.update(mse={96:1.,192:3.,336:5.,720:7.}[t['h']])
        r=aggregate(c,rows);self.assertEqual(r['nonUrban']['ETTh1']['candidate']['mse'],4.)
        for t,m in rows:
            if t['goal']==1 and t['arm']=='U':m['mse']=float(t['fold']+t['h'])
        self.assertEqual(aggregate(c,rows)['Urban_module']['U']['mean']['mse'],10.)

    def test_invalid_metrics_rejected(self):
        for value in (float('nan'),float('inf'),-1.,True):
            c,rows=saved_fixture();rows[0][1]['mse']=value
            with self.subTest(value=value),self.assertRaises(ValueError):aggregate(c,rows)

class PolicyRouting(unittest.TestCase):
    def test_stored_config_and_confirmed24_exact_source_binding(self):
        validate(STORED);self.assertEqual(STORED['source_binding'],worker.source_binding())
        before=json.loads((ROOT.parent/'amd-execution-evidence/m6/M6-Timemixer-launch-preparation-r1/first-batch-proposal.json').read_text())
        ids=before['proposed_task_ids'];self.assertEqual(len(ids),24);self.assertEqual(len(launcher.batch(STORED,ids)),6)
        self.assertEqual(sum(next(t for t in STORED['tasks'] if t['id']==id)['arithmetic']['max_epochs'] for id in ids),480)

    def test_dual_authorization_metadata_and_legacy_rejection(self):
        c=copy.deepcopy(STORED);ids=[t['id'] for t in c['tasks'][:3]];cp=ROOT/'configs/timemixer_uv_round2_resolved_v1.json'
        evidence=ROOT.parent/'amd-execution-evidence/m6/M6-Timemixer-runner-integration-r1/repair-r1/acceptance.md'
        auth=dict(schema='timemixer-uv-launch-v2',review_approved=True,commit='synthetic-metadata-only',config_sha256=file_sha(cp),task_ids=ids,source_binding=worker.source_binding(),gate_policy=policy(),resource_plan=dict(q_max=4,waves=launcher.batch(c,ids),mandatory_probe=False,reviewed_evidence_refs=[dict(path=str(evidence),sha256=file_sha(evidence))]))
        def fake_git(*args):return '' if args[0]=='status' else 'synthetic-metadata-only refs/heads/M6-Timemixer' if args[0]=='ls-remote' else 'synthetic-metadata-only'
        with patch.object(launcher,'git',side_effect=fake_git):
            self.assertIs(launcher.validate_authorization(c,cp,ids,auth),auth)
            for mode in ('legacy_schema','legacy_metric','both_string','wrong_policy'):
                bad=copy.deepcopy(auth)
                if mode=='legacy_schema':bad.update(schema='timemixer-uv-launch-v1',main_metric='mse')
                elif mode=='legacy_metric':bad['main_metric']='mae'
                elif mode=='both_string':bad['gate_policy']='both'
                else:bad['gate_policy']['goal2']['intersection_requirement']=True
                with self.subTest(mode=mode),self.assertRaises(PermissionError):launcher.validate_authorization(c,cp,ids,bad)
        self.assertFalse(torch.cuda.is_initialized())

    def test_policy_only_changes_never_grant_retraining(self):
        t=STORED['tasks'][0]
        spec=legacy.adapter.StructuralSpec.from_profile(t['structural_source_profile'],use_u=True,use_v=False,task='MS',fold=1).value
        identity=dict(candidate_id=t['candidate_id'],task=t,profile=t['structural_source_profile'],structural_spec=spec,data_sha='same-data',purpose='development',reference_original=False,training_protocol=STORED['training_protocol'],config_sha='old',source={'summary.py':'old'},gate_policy=policy(),producer_commit='old')
        changed=copy.deepcopy(identity);changed.update(config_sha='new-report-config',producer_commit='new',source={'summary.py':'new'},gate_policy={'future':'report only'})
        changed['structural_spec']['implementation']['summary.py']='new-report-code'
        self.assertEqual(worker.successful_identity(identity),worker.successful_identity(changed))
        changed['structural_spec']['implementation']['modules.py']='different-model'
        self.assertNotEqual(worker.successful_identity(identity),worker.successful_identity(changed))

    def test_unknown_claim_checkpoint_unchanged_and_no_resume_updates(self):
        with tempfile.TemporaryDirectory(prefix='.uv-runner-',dir=ROOT) as d:
            folder=Path(d);c,id,data=legacy.fixture(folder,epochs=1)
            with self.assertRaisesRegex(RuntimeError,'test uncertainty'):worker.run_worker(c,id,folder/'results','claim',path_override=data,fault='after_test_claim')
            stage=next((folder/'results/claim/staging').iterdir());names=('best.pt','last.pt','updates.jsonl','history.jsonl','test-claim.json')
            before={name:file_sha(stage/name) for name in names};state=read(stage/'last.pt');self.assertEqual(state['steps'],2)
            with self.assertRaisesRegex(ValueError,'cannot repeat'):worker.run_worker(c,id,folder/'results','claim',path_override=data,resume=True)
            self.assertEqual(before,{name:file_sha(stage/name) for name in names})
            legacy._assert_nested_equal(self,state,read(stage/'last.pt'))
            self.assertFalse((stage/'test-result.json').exists())

    def test_actual_receipt_policy_and_runtime_mismatch_rejected(self):
        with tempfile.TemporaryDirectory(prefix='.uv-runner-',dir=ROOT) as d:
            folder=Path(d);c,id,data=legacy.fixture(folder,epochs=1)
            receipt=worker.run_worker(c,id,folder/'results','valid',path_override=data)
            self.assertEqual(receipt['manifest']['identity']['gate_policy'],policy())
            self.assertNotIn('gate_main_metric',receipt['manifest']['identity'])
            rp=folder/'receipt.json';rp.write_text(json.dumps(receipt))
            runtime=dict(schema='timemixer-uv-owned-runtime-v2',source_binding=worker.source_binding(),commit=receipt['manifest']['identity']['producer_commit'],gate_policy=policy(),synthetic=True)
            child=dict(receipt=rp,expected_task_id=id);attempt=folder/'results/valid'
            self.assertEqual(launcher.verify_child_receipt(child,c,attempt,runtime,None),id)
            wrong_version=copy.deepcopy(runtime);wrong_version['schema']='timemixer-uv-owned-runtime-v1'
            with self.assertRaisesRegex(ValueError,'runtime v2'):launcher.verify_child_receipt(child,c,attempt,wrong_version,None)
            changed=copy.deepcopy(runtime);changed['gate_policy']['goal2']['intersection_requirement']=True
            with self.assertRaisesRegex(ValueError,'gate policy'):launcher.verify_child_receipt(child,c,attempt,changed,None)
            legacy_runtime=copy.deepcopy(runtime);legacy_runtime.pop('gate_policy');legacy_runtime['gate_main_metric']='mse'
            with self.assertRaisesRegex(ValueError,'gate policy'):launcher.verify_child_receipt(child,c,attempt,legacy_runtime,None)
            altered=copy.deepcopy(receipt);altered['manifest']['identity']['gate_policy']='both'
            with patch('timemixer_dev.artifacts.audit',return_value=altered):
                rp.write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError,'gate policy'):launcher.verify_child_receipt(child,c,attempt,runtime,None)
            # Exercise the actual worker-context entry before any CUDA check or write.
            grant=folder/'metadata-invalid-grant.json'
            grant.write_text(json.dumps(dict(schema='timemixer-uv-launch-v2',review_approved=True,commit='metadata-fixture-only',task_ids=[id],gate_policy=policy())))
            parent=os.getppid();context_runtime=dict(schema='timemixer-uv-owned-runtime-v2',owner_pid=parent,owner_start=launcher.proc_start(parent),task_ids=[id],config_sha=digest(c),source_binding=worker.source_binding(),commit='metadata-fixture-only',authorization_ref=dict(path=str(grant),sha256=file_sha(grant)),gate_policy='both')
            path=folder/'metadata-runtime.json';path.write_text(json.dumps(context_runtime))
            def fake_git(*args):return '' if args[0]=='status' else 'metadata-fixture-only'
            with patch.object(launcher,'git',side_effect=fake_git),self.assertRaisesRegex(PermissionError,'runtime dual gate policy'):
                launcher.validate_worker_context(c,id,dict(path=str(path),sha256=file_sha(path)))
            # Same round's synthetic state: policy mismatch rejects before any live writes.
            from timemixer_dev.training_state import restore
            from tests.test_timemixer_runner_repair import live_state
            p=c['tasks'][0]['structural_source_profile'];tr=p['training']
            model=legacy.adapter.build(p,use_u=True,use_v=True,task='MS',fold=1)
            opt=torch.optim.Adam(model.parameters(),lr=tr['lr'],betas=tuple(tr['betas']),eps=tr['eps'],weight_decay=tr['weight_decay'])
            schedule=legacy.Schedule(opt,tr['scheduler']);generator=torch.Generator().manual_seed(17)
            state=read(Path(receipt['path'])/'last.pt');bad_identity=copy.deepcopy(state['identity']);bad_identity['gate_policy']['goal2']['intersection_requirement']=True
            before=live_state(model,opt,schedule,generator)
            with self.assertRaises(ValueError):restore(Path(receipt['path'])/'last.pt',model,opt,schedule,generator,bad_identity)
            legacy._assert_nested_equal(self,before,live_state(model,opt,schedule,generator))
            self.assertEqual(state['steps'],2)
            self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main(verbosity=2)
