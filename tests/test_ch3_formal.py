"""Exact-ID CH3 acceptance; fixtures never use project observations."""
import copy
import csv
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from utils.ch3_contract import (read_profiles,validate_manifest,profile,task_by_id,
    BestState,digest,summarize,validate_amd_declaration)


class ContractTests(unittest.TestCase):
    def setUp(self):self.c=read_profiles()
    def test_unique_counts_and_epochs(self):
        c=validate_manifest(self.c);expected={'AMD':(113,1180),'J':(113,1180),'TimeMixer':(16,200),'N':(24,240),'S':(24,240)}
        self.assertEqual(len(c['tasks']),495)
        self.assertEqual(sum(profile(c,t)['training']['epochs'] for t in c['tasks']),5340)
        for m in ('DLinear','PatchTST','iTransformer','ModernTCN','TimeXer'):expected[m]=(41,460)
        for m,(n,e) in expected.items():
            rows=[t for t in c['tasks'] if t['model']==m]
            self.assertEqual((len(rows),sum(profile(c,t)['training']['epochs'] for t in rows)),(n,e))
    def test_group_coverage_and_waves(self):
        from ch3_runner import verified_waves
        self.assertEqual((len(self.c['groups']),sum(g['q'] for g in self.c['groups'])),(54,195))
        self.assertEqual(sum(g['q']==4 for g in self.c['groups']),47)
        self.assertEqual(sum(g['q']==1 for g in self.c['groups']),7)
        for g in self.c['groups']:
            for q in (1,2,4):
                if q>g['q']:continue
                waves=verified_waves(g,dict(concurrency=q,status='Passed',representatives=g['representatives']))
                self.assertEqual(sum(waves,[]),g['task_ids'])
                self.assertTrue(all(len(w)<=q for w in waves))
    def test_duplicate_task_rejected(self):
        self.c['tasks'].append(self.c['tasks'][0])
        with self.assertRaises(ValueError):validate_manifest(self.c)
    def test_unverified_wave_rejected(self):
        from ch3_runner import verified_waves
        g=self.c['groups'][0]
        for decision in ({'concurrency':3},{'concurrency':4,'status':'Passed','representatives':[]}):
            with self.assertRaises(ValueError):verified_waves(g,decision)
    def test_family_target_and_kernel_mapping(self):
        for t in self.c['tasks']:
            p=profile(self.c,t)
            self.assertEqual(p['features'][p['target_idx']],self.c['datasets'][t['dataset']]['target'])
            self.assertEqual(sorted(p['aux_idx']+[p['target_idx']]),list(range(p['C'])))
            if t['model'] in ('AMD','J','N','S'):
                s=p['structure'];self.assertEqual((s['kernel_large'],s['kernel_small']),(7,3) if t['dataset']=='UrbanEV' else (31,5))
                self.assertEqual(s['layernorm'],t['dataset']!='ECL')
    def test_declaration_negative_identity(self):
        t=task_by_id(self.c,'J-ECL-MS-f1-h96-s2024');p=profile(self.c,t)
        args=dict(input_shape=(p['T'],p['C']),pred_len=96,patch=16,layernorm=False,target_idx=320,aux_idx=p['aux_idx'],norm=True,task_mode='target_exogenous',s2=True,thls=True)
        validate_amd_declaration(p,**args)
        for key,val in [('layernorm',True),('patch',12),('s2',False),('target_idx',0)]:
            bad=dict(args);bad[key]=val
            with self.assertRaises(ValueError):validate_amd_declaration(p,**bad)
    def test_training_fairness(self):
        for t in self.c['tasks']:
            p=profile(self.c,t);r=p['training']
            self.assertEqual(r,dict(self.c['training_common'],**self.c['datasets'][t['dataset']]['training']))
            self.assertEqual((r['seed'],r['search'],r['workers'],r['threads']),(2024,0,0,4))
    def test_pjm_fit_and_source_scope_confirmed(self):
        d=self.c['datasets']['PJM'];self.assertEqual(d['endpoints'],[36691,41933,52416])
        self.assertEqual(d['mandatory_blockers'],[])
        self.assertIn('not record-level vintage audit',d['source_admission']['source_availability'])
    def test_time_xer_reorder_inverse(self):
        from models.ch3_adapter import reversible_target_order
        for name in ('UrbanEV','Weather','ETTh1'):
            t=next(t for t in self.c['tasks'] if t['model']=='TimeXer' and t['dataset']==name)
            p=profile(self.c,t);order,inverse=reversible_target_order(p)
            self.assertEqual([order[i] for i in inverse],list(range(p['C'])))
            self.assertEqual(order[-1],p['target_idx'])
    def test_best_ties_and_patience(self):
        b=BestState(5);self.assertTrue(b.update(2.,1))
        for e in range(2,6):self.assertFalse(b.update(2.,e));self.assertFalse(b.stopped)
        self.assertFalse(b.update(3.,6));self.assertTrue(b.stopped);self.assertEqual(b.epoch,1)
        self.assertTrue(b.update(1.,7));self.assertEqual(b.bad,0)
    def test_best_rejects_nonfinite(self):
        for v in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):BestState().update(v,1)
    def test_fixed_epochs_no_early_stop(self):
        b=BestState();b.update(1.,1)
        for e in range(2,22):b.update(2.,e)
        self.assertFalse(b.stopped);self.assertEqual(b.epoch,1)
    def test_summary_complete_identity_and_no_duplicate(self):
        g=self.c['groups'][0]
        rows=[dict(id=i,input_variant=g['input_variant'],purpose='ch3_formal',protocol_sha=digest(self.c),mse=float(k),mae=float(k)/2) for k,i in enumerate(g['task_ids'])]
        r=summarize(self.c,rows)[g['id']];self.assertEqual(r['mse'],11.5);self.assertIsNone(r['std'])
        for bad in (rows[:-1],rows+[rows[0]],[dict(rows[0],purpose='M4')]):
            with self.assertRaises(ValueError):summarize(self.c,bad)
    def test_preflight_requires_explicit_freeze(self):
        from ch3_runner import preflight
        c=copy.deepcopy(self.c);c['sources']={}
        with patch('ch3_runner.git',return_value=''):reasons=preflight(c,'AMD')
        self.assertIn('explicit review/closure authorization missing',reasons)


class IncrementalTests(unittest.TestCase):
    def setUp(self):self.c=read_profiles()
    def test_cancelled_scope_and_reuse(self):
        tasks=self.c['tasks']
        self.assertFalse(any(t['input_variant'] in ('F0','TargetOnly') for t in tasks))
        self.assertEqual(sum(t['input_variant'] in ('F4','MS') and t['model'] not in ('N','S') for t in tasks),303)
        self.assertEqual(sum(t['model'] in ('N','S') for t in tasks),48)
        self.assertEqual(sum(t['model']=='J' and t['input_variant'] in ('F1','F2','F3') for t in tasks),72)
        self.assertEqual(sum(t['dataset']=='PJM' and t['input_variant']=='MS' for t in tasks),7)
        self.assertEqual({t['input_variant'] for t in tasks if t['model']=='AMD' and t['dataset']=='UrbanEV'},{'F1','F2','F3','F4'})
        self.assertEqual(sum(t['model']=='AMD' and t['input_variant'] in ('F1','F2','F3') for t in tasks),72)
        self.assertEqual(sum(t['model']=='AMD' and t['input_variant']=='F4' for t in tasks),24)
        self.assertEqual(sum(t['model']=='J' and t['input_variant']=='F4' for t in tasks),24)
    def test_parallel_input_variants(self):
        bank=self.c['urban_input_variants']
        for name,C in [('F1',6),('F2',8),('F3',9),('F4',11)]:
            p=profile(self.c,task_by_id(self.c,f'J-UrbanEV-{name}-f1-h3-s2024'))
            self.assertEqual((p['C'],len(p['aux_idx'])),(C,C-1))
        self.assertFalse(set(bank['F2']).issubset(bank['F3']))
        self.assertEqual(set(bank['F2'])|set(bank['F3']),set(bank['F4']))
        self.assertEqual(bank['F0'],['volume'])
    def test_empty_aux_and_j_f0_rejected_before_construction(self):
        t=task_by_id(self.c,'J-UrbanEV-F4-f1-h3-s2024')
        for arm in ('J','S'):
            with self.assertRaisesRegex(ValueError,'nonempty'):profile(self.c,dict(t,model=arm,input_variant='F0'))
            p=profile(self.c,t);p.update(model=arm,aux_idx=[])
            with self.assertRaisesRegex(ValueError,'nonempty'):
                validate_amd_declaration(p,input_shape=(12,11),pred_len=1,patch=12,layernorm=True,
                    target_idx=0,aux_idx=[],norm=True,task_mode='target_exogenous',s2=True,thls=arm=='J')
        with self.assertRaises(ValueError):task_by_id(self.c,'J-UrbanEV-F0-f1-h3-s2024')
    def test_cancelled_group_and_old_schema_rejected(self):
        c=copy.deepcopy(self.c);c['groups'].append(dict(c['groups'][0],id='AMD-UrbanEV-F0'))
        with self.assertRaises(ValueError):validate_manifest(c)
        c=copy.deepcopy(self.c);c['contract']='ch3-target-ms-formal-v1'
        with self.assertRaises(ValueError):validate_manifest(c)
        self.assertNotIn('preset',json.dumps(self.c))
    def test_old_approval_rejected(self):
        from ch3_runner import preflight
        c=copy.deepcopy(self.c);c['sources']={}
        a=dict(commit='head',protocol_sha=c['supersedes']['protocol_sha'],code={},hardware={},environment={},purpose='ch3_resource_probe',reviewed=True)
        with patch('ch3_runner.git',side_effect=lambda *a:'head' if a[0]=='rev-parse' else ''),patch('ch3_runner.code_binding',return_value={}),patch('ch3_runner.hardware_binding',return_value={}),patch('ch3_runner.environment_binding',return_value={}):
            for old_sha in (c['supersedes']['protocol_sha'],c['supersedes']['earlier_protocol_sha']):
                with self.subTest(protocol_sha=old_sha):
                    self.assertIn('approval configuration mismatch',preflight(c,None,dict(a,protocol_sha=old_sha),True))
    def test_probe_reports_require_current_coverage(self):
        from ch3_runner import validate_probe_report
        report=dict(protocol_sha=digest(self.c),Q=195,decisions={g['id']:{'status':'Blocked'} for g in self.c['groups']})
        validate_probe_report(self.c,report)
        for bad in (dict(report,Q=200),dict(report,protocol_sha=self.c['supersedes']['protocol_sha']),
                    dict(report,protocol_sha=self.c['supersedes']['earlier_protocol_sha']),
                    dict(report,decisions=dict(report['decisions'],**{'J-UrbanEV-F0':{'status':'Passed'}}))):
            with self.assertRaises(ValueError):validate_probe_report(self.c,bad)
    def test_exact_retained_and_added_plan(self):
        old=read_profiles(Path(self.c['execution']['evidence'])/'before/configs/ch3_formal_profiles.json')
        old_tasks={t['id']:t for t in old['tasks']};new_tasks={t['id']:t for t in self.c['tasks']}
        expected={f'AMD-UrbanEV-F{v}-f{f}-h{h}-s2024' for v in (1,2,3) for f in range(1,7) for h in (3,6,9,12)}
        self.assertEqual(len(old_tasks),423);self.assertEqual(set(new_tasks)-set(old_tasks),expected)
        self.assertEqual(set(old_tasks)-set(new_tasks),set())
        for run,t in old_tasks.items():
            with self.subTest(run=run):
                self.assertEqual(new_tasks[run],t);self.assertEqual(profile(self.c,new_tasks[run]),profile(old,t))
        with self.assertRaises(ValueError):validate_manifest(old)
        old520=read_profiles(Path(self.c['execution']['evidence'])/'reference/section11-plan.json')
        with self.assertRaises(ValueError):validate_manifest(old520)
        self.assertTrue(expected.issubset({t['id'] for t in old520['tasks']}))
        # Even a foreign list with today's count cannot pass by count alone.
        foreign=copy.deepcopy(self.c);foreign['tasks'][0]=dict(foreign['tasks'][0],input_variant='F0')
        with self.assertRaises(ValueError):validate_manifest(foreign)
    def test_aj_paired_inputs_and_summary_reuse(self):
        for v in ('F1','F2','F3','F4'):
            rows=[]
            for fold in range(1,7):
                for h in (3,6,9,12):
                    with self.subTest(input_variant=v,fold=fold,h=h):
                        a=task_by_id(self.c,f'AMD-UrbanEV-{v}-f{fold}-h{h}-s2024');j=task_by_id(self.c,f'J-UrbanEV-{v}-f{fold}-h{h}-s2024')
                        pa=profile(self.c,a);pj=profile(self.c,j)
                        self.assertEqual(dict(pa,model='paired'),dict(pj,model='paired'))
                        for t in (a,j):rows.append(dict(id=t['id'],input_variant=v,purpose='ch3_formal',protocol_sha=digest(self.c),mse=1.,mae=1.))
            result=summarize(self.c,rows)
            self.assertEqual(set(result),{f'AMD-UrbanEV-{v}',f'J-UrbanEV-{v}'})
            self.assertTrue(all(x['input_variant']==v for x in result.values()))
            with self.assertRaises(ValueError):summarize(self.c,rows+[rows[0]])
            bad=copy.deepcopy(rows);bad[0]['input_variant']='F0'
            with self.assertRaises(ValueError):summarize(self.c,bad)
    def test_probe_monitor_blocker_before_queue(self):
        from ch3_runner import preflight
        c=copy.deepcopy(self.c);c['sources']={}
        a=dict(commit='head',protocol_sha=digest(c),code={},hardware={},environment={},purpose='ch3_resource_probe',reviewed=True)
        with patch('ch3_runner.git',side_effect=lambda *a:'head' if a[0]=='rev-parse' else ''),patch('ch3_runner.code_binding',return_value={}),patch('ch3_runner.hardware_binding',return_value={}),patch('ch3_runner.environment_binding',return_value={}):
            reasons=preflight(c,None,a,True)
        self.assertNotIn('active_worker_pid_attribution_unresolved_reviewed_admission_path_required',reasons)
        self.assertEqual(c['execution']['probe']['resource_diagnostic_status'],'Passed on owned 16MiB tensor processes; formal groups untested')
    def test_formal_identity_and_completion_input_variant(self):
        from ch3_runner import formal_identity,validate_model_completion
        t=task_by_id(self.c,'J-UrbanEV-F2-f1-h3-s2024')
        with patch('ch3_runner.code_binding',return_value={}):i=formal_identity(self.c,t,{},dict(commit='synthetic'))
        self.assertEqual(i['input_variant'],'F2');self.assertEqual(i['profile_sha'],digest(profile(self.c,t)))
        report=dict(model='AMD',protocol_sha=digest(self.c),task_ids=[t['id'] for t in self.c['tasks'] if t['model']=='AMD'],
                    results={g['id']:{'input_variant':g['input_variant']} for g in self.c['groups'] if g['model']=='AMD'})
        validate_model_completion(self.c,'AMD',report)
        for bad in (dict(report,protocol_sha=self.c['supersedes']['protocol_sha']),dict(report,task_ids=report['task_ids'][:-1])):
            with self.assertRaises(ValueError):validate_model_completion(self.c,'AMD',bad)
    def sample(self,**overrides):
        gib=1024**3
        return dict(dict(uuid='GPU-fixture',total=80*gib,used=10*gib,free=67*gib,driver_reserved=3*gib,
                    process_table_reliable=True,nvml_processes={},owned_host_pids=[],process_gpu={}),**overrides)
    def test_nvml_empty_table_unknown_worker(self):
        from m5_formal_entry import resource_assessment
        s=self.sample();self.assertTrue(resource_assessment(s,[])['admission'])
        a=resource_assessment(s,[42],s)
        self.assertFalse(a['admission'])
        self.assertEqual(a['process_attribution'],'Not verified')
        # Empty process table does not measure any worker peak.
        self.assertNotIn('42',s['process_gpu'])
    def test_nvml_partial_unknown_competitor_denied(self):
        from m5_formal_entry import resource_assessment
        s=self.sample(nvml_processes={'42':1.,'99':2.},owned_host_pids=['42'],process_gpu={'42':1.})
        a=resource_assessment(s,[42,43],self.sample())
        self.assertFalse(a['admission']);self.assertEqual(a['unknown_pids'],['99'])
        self.assertEqual(a['process_attribution'],'Not verified')
    def test_nvml_whole_card_fallback_and_bad_samples(self):
        from m5_formal_entry import resource_assessment
        s=self.sample(nvml_processes={'42':None},owned_host_pids=['42'],process_gpu={'42':None})
        a=resource_assessment(s,[42],self.sample())
        self.assertTrue(a['admission']);self.assertEqual(a['measurement'],'whole-card only')
        self.assertEqual(a['process_attribution'],'Not verified')
        for bad in (dict(s,uuid='other'),dict(s,process_table_reliable=False),dict(s,free=1),dict(s,used=float('nan'))):
            self.assertFalse(resource_assessment(bad,[42],self.sample())['admission'])
    def test_nvml_parser_namespace_and_na(self):
        from m5_formal_entry import gpu_sample
        with patch('m5_formal_entry.subprocess.check_output',side_effect=['GPU-fixture, 81920, 10240, 68608, 3072','900, [N/A]\n']),patch('m5_formal_entry.owned_pid_metadata',return_value={'host_pid':900,'rss':32768}):
            s=gpu_sample([42])
        self.assertIsNone(s['process_gpu']['42']);self.assertEqual(s['owned_host_pids'],['900'])
        self.assertEqual(s['pid_candidates']['42'],[900]);self.assertEqual(s['uuid'],'GPU-fixture')
        self.assertEqual(s['driver_reserved'],3072*1024**2)
    def test_weather_blocker_is_machine_readable(self):
        self.assertEqual(self.c['datasets']['Weather']['time_index_policy'],'weather_original_records')
        self.assertTrue(self.c['datasets']['Weather']['source_admission']['unverified_explanation'])


class ResourceAdmissionTests(unittest.TestCase):
    def sample(self,**kw):return IncrementalTests().sample(**kw)
    def assess(self,s,pids=(42,),baseline=None):
        from m5_formal_entry import resource_assessment
        return resource_assessment(s,list(pids),self.sample() if baseline is None else baseline)
    def test_known_pid(self):
        s=self.sample(nvml_processes={'900':123},owned_host_pids=['900'],process_gpu={'42':123})
        self.assertTrue(self.assess(s)['admission']);self.assertEqual(self.assess(s)['process_attribution'],'Measured')
    def test_unknown_pid(self):
        self.assertFalse(self.assess(self.sample(nvml_processes={'999':123},owned_pid_metadata={'42':{'host_pid':900}}))['admission'])
    def test_cpu_initializing(self):
        s=self.sample(owned_pid_metadata={'42':{'host_pid':900}},process_states={'42':'alive_not_registered'})
        self.assertTrue(self.assess(s)['admission']);self.assertEqual(self.assess(s)['process_attribution'],'Not verified')
        self.assertNotIn('42',s['process_gpu'])
    def test_unmapped_initialization_rejected(self):
        self.assertFalse(self.assess(self.sample())['admission'])
    def test_exited_process(self):
        from m5_formal_entry import owned_pid_metadata
        with patch('pathlib.Path.read_text',side_effect=FileNotFoundError):m=owned_pid_metadata(42)
        self.assertEqual(m['state'],'exited_during_sample');self.assertIsNone(m['host_pid'])
        self.assertFalse(self.assess(self.sample(owned_pid_metadata={'42':m}))['admission'])
    def test_sampling_failure(self):
        from m5_formal_entry import gpu_sample
        import subprocess
        with patch('m5_formal_entry.subprocess.check_output',side_effect=subprocess.CalledProcessError(1,'nvidia-smi')):
            with self.assertRaises(subprocess.CalledProcessError):gpu_sample([])
        self.assertFalse(self.assess(self.sample(process_table_reliable=False),())['admission'])
    def test_driver_reserved(self):
        self.assertTrue(self.assess(self.sample(),())['card_reliable'])
        self.assertFalse(self.assess(self.sample(driver_reserved=0),())['card_reliable'])
    def test_margin(self):
        gib=1024**3;s=self.sample(used=70*gib,free=7*gib)
        a=self.assess(s,());self.assertEqual(a['reserve'],8*gib);self.assertFalse(a['admission'])
        s=self.sample(total=100*gib,used=88*gib,free=9*gib)
        self.assertEqual(self.assess(s,())['reserve'],10*gib);self.assertFalse(self.assess(s,())['admission'])
    def test_uuid_change(self):
        self.assertFalse(self.assess(self.sample(uuid='other',process_gpu={'42':1}))['admission'])
    def test_unknown_memory_never_zero(self):
        s=self.sample(nvml_processes={'900':None},owned_host_pids=['900'],process_gpu={'42':None})
        a=self.assess(s);self.assertTrue(a['admission']);self.assertEqual(a['process_attribution'],'Not verified')
        self.assertIsNone(s['process_gpu']['42'])
    def test_sched_mapping(self):
        from m5_formal_entry import owned_pid_metadata
        stat='42 (a b) '+' '.join(['S']+['0']*18+['123'])
        with patch('pathlib.Path.read_text',side_effect=[stat,'NSpid:\t42\nVmRSS:\t32 kB\n','python (900, #threads: 4)\n',stat]),patch('os.readlink',return_value='pid:[fixture]'):
            m=owned_pid_metadata(42)
        self.assertEqual((m['host_pid'],m['start_ticks'],m['rss']),(900,'123',32768));self.assertEqual(m['nspid'],['42'])
    def test_pid_reuse_rejected(self):
        from m5_formal_entry import owned_pid_metadata
        stat='42 (a) '+' '.join(['S']+['0']*18)
        with patch('pathlib.Path.read_text',side_effect=[stat+' 123','Pid:\t42','python (900, #threads: 1)',stat+' 124']):m=owned_pid_metadata(42)
        self.assertIsNone(m['host_pid']);self.assertEqual(m['state'],'metadata_unavailable')
    def test_parser_na(self):
        from m5_formal_entry import gpu_sample
        with patch('m5_formal_entry.subprocess.check_output',side_effect=['GPU-fixture, 81920, 10240, 68608, 3072','900, [N/A]\n']),patch('m5_formal_entry.owned_pid_metadata',return_value={'host_pid':900,'rss':32768}):s=gpu_sample([42])
        self.assertIsNone(s['process_gpu']['42']);self.assertEqual(s['owned_host_pids'],['900'])
        self.assertEqual(s['pid_candidates']['42'],[900]);self.assertEqual(s['driver_reserved'],3072*1024**2)
    def test_bad_process_memory(self):
        from m5_formal_entry import gpu_sample
        for value in ('nan','-1'):
            with self.subTest(value=value),patch('m5_formal_entry.subprocess.check_output',side_effect=['GPU-fixture, 81920, 10240, 68608, 3072',f'900, {value}\n']):s=gpu_sample([])
            self.assertFalse(s['process_table_reliable']);self.assertIsNone(s['nvml_processes']['900'])
    def test_duplicate_host_mapping(self):
        from m5_formal_entry import gpu_sample
        with patch('m5_formal_entry.subprocess.check_output',side_effect=['GPU-fixture, 81920, 10240, 68608, 3072','900, 1']),patch('m5_formal_entry.owned_pid_metadata',return_value={'host_pid':900}):s=gpu_sample([42,43])
        self.assertFalse(s['process_table_reliable'])
    def test_diagnostic_binding_and_no_probe_waiver(self):
        from m5_formal_entry import validate_config
        from restricted_io_guard import require_installed
        s=copy.deepcopy(require_installed());s.update(purpose='ch3_resource_diagnostic',ids=['ch3.resource_diagnostic'],diagnostic='bad')
        with self.assertRaisesRegex(ValueError,'exact resource diagnostic'):validate_config(s)
        from ch3_runner import preflight
        c=read_profiles();c['sources']={}
        with patch('ch3_runner.git',return_value=''):reasons=preflight(c,None,None,True)
        self.assertIn('explicit review/closure authorization missing',reasons)
        self.assertNotIn('active_worker_pid_attribution_unresolved_reviewed_admission_path_required',reasons)
        self.assertEqual(c['execution']['probe']['resource_diagnostic_status'],'Passed on owned 16MiB tensor processes; formal groups untested')


class DataTests(unittest.TestCase):
    def setUp(self):
        from restricted_io_guard import require_installed
        self.c=read_profiles();self.root=Path(self.c['execution']['fixture'])/Path(require_installed()['output']).name/self.id().split('.')[-1]
        self.root.mkdir(parents=True,exist_ok=False)
    def fixture(self):
        c=copy.deepcopy(self.c);d=c['datasets']['ETTh1'];d.update(T=4,endpoints=[16,24,32])
        d['training'].update(batch=5,eval_batch=5)
        t=dict(next(t for t in c['tasks'] if t['model']=='AMD' and t['dataset']=='ETTh1'),h=3)
        f=self.root/'prefix.csv'
        with f.open('w',newline='') as stream:
            w=csv.writer(stream);w.writerow(['date']+d['features'])
            for i in range(24):w.writerow([f'2022-01-{i+1:02d} 00:00:00']+[i+j for j in range(7)])
            w.writerow(['forbidden-test-poison','must never parse'])
        return c,t,f
    def test_prefix_stops_before_poison(self):
        from utils.ch3_data import load
        c,t,f=self.fixture();ds,m=load(c,t,path_override=f)
        self.assertEqual(m['parsed_records'],24);self.assertFalse(m['test_observations_accessed']);self.assertNotIn('test',ds)
        self.assertFalse(m['full_file_verified']);self.assertEqual(m['window_counts'],{'train':10,'validation':6})
    def test_scaler_train_only_and_val_context(self):
        from utils.ch3_data import load
        c,t,f=self.fixture();ds,m=load(c,t,path_override=f)
        self.assertEqual(m['scaler']['mean'],[7.5+j for j in range(7)])
        x,y=ds['validation'][0];scale=m['scaler']['scale'][0]
        self.assertAlmostEqual(float(x[0,0]),(12-7.5)/scale,places=6)
        self.assertAlmostEqual(float(y[0,0]),(16-7.5)/scale,places=6)
    def test_tail_and_shuffle_policy(self):
        from utils.ch3_data import load,batches
        c,t,f=self.fixture();ds,m=load(c,t,path_override=f);p=profile(c,t)
        self.assertEqual([len(x) for x,y in batches(ds['validation'],p,'validation')],[5,1])
        self.assertTrue(batches(ds['train'],p,'train').drop_last)
    def test_no_implicit_test_capability(self):
        from utils.ch3_data import load
        c,t,f=self.fixture()
        with self.assertRaises(PermissionError):load(c,t,path_override=f,test_capability={})
        from ch3_runner import audit_resume
        with self.assertRaises(PermissionError):audit_resume(self.root,{},t['id'])
        with self.assertRaises(FileExistsError):audit_resume(self.root,{'resume_audits':{t['id']:{'mode':'not_started'}}},t['id'])
    def test_ordered_header_rejected(self):
        from utils.ch3_data import prefix_csv
        c,t,f=self.fixture()
        with self.assertRaises(ValueError):prefix_csv(f,1,list(reversed(c['datasets']['ETTh1']['features'])))
    def test_short_prefix_rejected(self):
        from utils.ch3_data import prefix_csv
        f=self.root/'short.csv';f.write_text('date,x\n2022,1\n')
        with self.assertRaises(ValueError):prefix_csv(f,2)
    def test_weather_diagnostic_and_input_metadata(self):
        from utils.ch3_data import adjacent_timestamps,load
        c,t,f=self.fixture();r=adjacent_timestamps(f,22,24)
        self.assertEqual([x['record_index'] for x in r['rows']],[22,23])
        self.assertEqual(r['numeric_observations_parsed'],0);self.assertFalse(r['test_observations_accessed'])
        with self.assertRaises(ValueError):adjacent_timestamps(f,0,42158)
        ds,metadata=load(c,t,path_override=f)
        self.assertEqual(metadata['input_variant'],'MS');self.assertNotIn('preset',metadata)
    def test_resume_rejects_old_identity_before_checkpoint(self):
        from ch3_runner import audit_resume,formal_identity
        t=task_by_id(self.c,'J-UrbanEV-F1-f1-h3-s2024')
        with patch('ch3_runner.code_binding',return_value={}):identity=formal_identity(self.c,t,{},dict(commit='fixture'))
        manifest=dict(identity=identity,task=t,profile=profile(self.c,t))
        record={'resume_audits':{t['id']:{'mode':'resume','sha256':{}}}}
        for field,value in [('protocol_sha',self.c['supersedes']['protocol_sha']),
                            ('protocol_sha',self.c['supersedes']['earlier_protocol_sha']),('input_variant','F0')]:
            bad=copy.deepcopy(manifest);bad['identity'][field]=value
            (self.root/'manifest.json').write_text(json.dumps(bad))
            with patch('pathlib.Path.read_bytes',side_effect=AssertionError('checkpoint must not be read')):
                with self.assertRaisesRegex(ValueError,'before checkpoint'):audit_resume(self.root,record,t['id'])
    def test_all_urban_fold_h_labels(self):
        import numpy as np
        from utils.ch3_data import Windows
        x=np.zeros((4344,2,3),dtype='float32');y=np.broadcast_to(np.arange(4344,dtype='float32')[:,None],(4344,2)).copy()
        for train,val,test in self.c['urban_folds']:
            for h in (3,6,9,12):
                ds=Windows(x,y,train,val,12,1,urban=True,label_horizon=h)
                self.assertEqual(len(ds),(val-train-12-h+1)*2)
                self.assertEqual(ds[0][1].item(),train+12+h-1)
                self.assertEqual(ds[len(ds)-1][1].item(),val-1)
        # Distinct source timestamp spellings must compare by actual time.
        import pandas as pd
        from utils.ch3_data import load
        c=copy.deepcopy(self.c);c['urban_folds'][0]=[24,48,72]
        for name in ('volume','e_price','s_price','weather_central'):
            with (self.root/(name+'.csv')).open('w',newline='') as stream:
                w=csv.writer(stream);weather=name=='weather_central'
                w.writerow(['time']+(['T','P','U'] if weather else ['a','b']))
                for i,time in enumerate(pd.date_range('2022-09-01',periods=48,freq='h')):
                    stamp=time.isoformat() if weather else str(time)
                    w.writerow([stamp]+[float(i+j) for j in range(3 if weather else 2)])
                w.writerow(['never parse this test row'])
        t=task_by_id(c,'AMD-UrbanEV-F4-f1-h3-s2024');ds,m=load(c,t,path_override=self.root)
        self.assertEqual(ds['validation'][0][0].shape,(12,11));self.assertFalse(m['test_observations_accessed'])
    def test_streaming_metrics_target_and_tail(self):
        import torch
        from ch3_runner import evaluate
        class Predict:
            def eval(self):pass
            def __call__(self,x):return x
        p=dict(model='DLinear',target_idx=1,pred_len=1)
        data=[(torch.tensor([[[99.,1.]],[[99.,2.]]]),torch.zeros(2,1,1)),(torch.tensor([[[99.,4.]]]),torch.zeros(1,1,1))]
        r=evaluate(Predict(),data,p,'cpu');self.assertEqual(r['elements'],3);self.assertEqual(r['mse'],7.)
    def test_guard_exact_id_rejection(self):
        from restricted_io_guard import require_installed
        from m5_formal_entry import validate_config
        s=dict(require_installed());s['ids']=s['ids']+['unapproved.case']
        with self.assertRaises(Exception):validate_config(s)
    def test_guard_purpose_source_binding_rejection(self):
        from restricted_io_guard import require_installed
        from m5_formal_entry import validate_config
        s=dict(require_installed());s['purpose']='ch3_probe';s['task']='not-a-task'
        with self.assertRaises(ValueError):validate_config(s)
    def test_protected_path_policy_without_asset_read(self):
        from restricted_io_guard import require_installed
        from m5_formal_entry import check_access
        def reject(*args):raise PermissionError(args)
        with self.assertRaises(PermissionError):check_access('/not-read/toy.ckpt',False,require_installed(),reject,None)


class DataAdmissionTests(unittest.TestCase):
    def setUp(self):
        from restricted_io_guard import require_installed
        self.c=read_profiles()
        self.root=Path(self.c['execution']['fixture'])/Path(require_installed()['output']).name/self.id().split('.')[-1]
        self.root.mkdir(parents=True,exist_ok=False)

    def weather_fixture(self, reverse=False):
        c=copy.deepcopy(self.c);d=c['datasets']['Weather'];d.update(T=4,endpoints=[16,24,32])
        t=dict(next(t for t in c['tasks'] if t['model']=='AMD' and t['dataset']=='Weather'),h=3)
        f=self.root/'weather.csv'
        with f.open('w',newline='') as stream:
            w=csv.writer(stream);w.writerow(['date']+d['features'])
            for i in range(24):
                day=(9 if reverse else 10) if i==10 else i+1
                w.writerow([f'2022-01-{day:02d} 00:00:00']+[i+j for j in range(21)])
            w.writerow(['test-poison','must not parse'])
        return c,t,f

    def test_weather_duplicate_records(self):
        from utils.ch3_data import load
        c,t,f=self.weather_fixture();ds,m=load(c,t,path_override=f)
        self.assertEqual(m['time_index']['duplicate_record_indices'],[10])
        self.assertEqual(m['parsed_records'],24);self.assertEqual(m['scaler_fit_records'],16)
        self.assertEqual(m['window_counts'],{'train':10,'validation':6});self.assertNotIn('test',ds)
        self.assertEqual(m['scaler']['mean'],[7.5+j for j in range(21)])
        # Distinct records sharing a timestamp retain their distinct observations.
        self.assertNotEqual(float(ds['train'].x[9,1]),float(ds['train'].x[10,1]))

    def test_weather_reverse_rejected(self):
        from utils.ch3_data import load
        c,t,f=self.weather_fixture(reverse=True)
        with self.assertRaisesRegex(ValueError,'reverse time forbidden'):load(c,t,path_override=f)

    def test_other_domain_duplicates_rejected(self):
        import pandas as pd
        from utils.ch3_data import validate_times
        times=pd.DatetimeIndex(['2022-01-01','2022-01-01'])
        for domain in ('ETTh1','ECL','Exchange','PJM'):
            with self.subTest(domain=domain):
                with self.assertRaisesRegex(ValueError,'unique'):validate_times(times,domain,self.c['datasets'][domain])

    def test_weather_policy_not_global(self):
        import pandas as pd
        from utils.ch3_data import validate_times
        times=pd.DatetimeIndex(['2022-01-01'])
        with self.assertRaisesRegex(ValueError,'Weather-only'):
            validate_times(times,'ECL',dict(time_index_policy='weather_original_records'))
        with self.assertRaisesRegex(ValueError,'unknown time index'):
            validate_times(times,'Weather',dict(time_index_policy='anything'))

    def test_source_status_and_limits(self):
        for domain in ('Weather','PJM','ECL','Exchange'):
            with self.subTest(domain=domain):
                d=self.c['datasets'][domain];a=d['source_admission']
                self.assertEqual(d['mandatory_blockers'],[])
                self.assertEqual(len(a['local_sha256_inherited']),64)
                self.assertTrue(a['unverified_explanation']);self.assertIn('user confirmed',a['accepted_scope'])
                self.assertNotEqual(a['source_availability'],'Passed')
        self.assertIn('not record-level vintage audit',self.c['datasets']['PJM']['source_admission']['source_availability'])

    def test_changed_source_state_rejected(self):
        from utils.ch3_data import verify_source_state
        f=self.root/'state.txt';f.write_text('original');s=f.stat()
        state=dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
        d=dict(path=str(f),source_admission=dict(local_state=state))
        self.assertEqual(verify_source_state(d),state)
        f.write_text('different length')
        with self.assertRaisesRegex(ValueError,'state changed'):verify_source_state(d)

    def test_pjm_endpoints_windows_tails(self):
        d=self.c['datasets']['PJM'];n=d['declared_rows'];a,b,z=d['endpoints'];T=d['T'];H=d['horizons'][0]
        self.assertEqual((n,a,b,z,T,H),(52416,36691,41933,52416,168,24))
        self.assertEqual((a,b),(int(.7*n),n-int(.2*n)))
        self.assertEqual((a-T-H+1,b-a-H+1,z-b-H+1),(36500,5219,10460))
        self.assertEqual([divmod(v,128) for v in (36500,5219,10460)],[(285,20),(40,99),(81,92)])

    def test_old_protocol_rejected(self):
        from ch3_runner import validate_probe_report,preflight
        before=json.loads((Path(self.c['execution']['evidence'])/'before/configs/ch3_formal_profiles.json').read_text())
        self.assertNotEqual(digest(before),digest(self.c))
        with self.assertRaisesRegex(ValueError,'configuration'):validate_probe_report(self.c,dict(protocol_sha=digest(before),Q=195))
        fixture=copy.deepcopy(self.c);fixture['sources']={}
        with patch('ch3_runner.git',return_value='fixture'),patch('ch3_runner.code_binding',return_value={}),patch('ch3_runner.environment_binding',return_value={}),patch('ch3_runner.hardware_binding',return_value={}):
            reasons=preflight(fixture,None,dict(commit='fixture',protocol_sha=digest(before),code={},environment={},hardware={},purpose='ch3_resource_probe',reviewed=True),True)
        self.assertIn('approval configuration mismatch',reasons)

    def test_prefix_binding_before_access(self):
        from restricted_io_guard import require_installed
        from m5_formal_entry import validate_config
        s=copy.deepcopy(require_installed());s.update(purpose='ch3_prefix',case='Weather',task=None,diagnostic='train_validation_connectivity',ids=['ch3.prefix.Weather.train_validation'],prefix_files={self.c['datasets']['Weather']['path']:42157})
        validate_config(s)
        for field,value in [('prefix_files',{self.c['datasets']['Weather']['path']:42158}),('case','ECL'),('ids',['unapproved.prefix'])]:
            with self.subTest(field=field):
                bad=dict(s);bad[field]=value
                with self.assertRaises((ValueError,PermissionError)):validate_config(bad)

    def test_unchanged_manifest_and_profiles(self):
        before=json.loads((Path(self.c['execution']['evidence'])/'before/configs/ch3_formal_profiles.json').read_text())
        validate_manifest(self.c)
        self.assertEqual(self.c['tasks'],before['tasks']);self.assertEqual(self.c['groups'],before['groups'])
        self.assertEqual((len(self.c['tasks']),len(self.c['groups']),sum(g['q'] for g in self.c['groups'])),(495,54,195))
        self.assertEqual([profile(self.c,t) for t in self.c['tasks']],[profile(before,t) for t in before['tasks']])


class BaselineFollowupTests(unittest.TestCase):
    def setUp(self):
        self.c=read_profiles()
        self.old=json.loads((Path(self.c['execution']['evidence'])/'before/configs/ch3_formal_profiles.json').read_text())
    def task(self,model='DLinear',domain='ETTh1',h=96):
        return next(t for t in self.c['tasks'] if t['model']==model and t['dataset']==domain and t['h']==h)
    def entry(self,c=None):return (c or self.c)['baseline_training_overrides']['DLinear']['ETTh1']['96']
    def test_counts(self):
        c=validate_manifest(self.c);self.assertEqual((len(c['tasks']),len(c['groups']),sum(g['q']for g in c['groups'])),(495,54,195))
        f=c['execution']['followup'];self.assertEqual((len(f['inherit_groups']),len(f['retest_groups']),f['Q'],f['planned_adam']),(10,44,170,3036))
        self.assertEqual(set(f['inherit_groups'])|set(f['retest_groups']),{g['id']for g in c['groups']})
        self.assertFalse(set(f['inherit_groups'])&set(f['retest_groups']))
        self.assertEqual((f['remaining_first_adam'],f['additional_first_adam_proposed']),(2327,709))
    def test_preserved_tasks(self):
        self.assertEqual(self.c['tasks'],self.old['tasks']);self.assertEqual(self.c['groups'],self.old['groups'])
    def test_family_training(self):
        for t in self.c['tasks']:
            if t['model'] in ('AMD','J','N','S'):
                with self.subTest(run=t['id']):self.assertEqual(profile(self.c,t),profile(self.old,t))
    def test_only_three_fields(self):
        for t in self.c['tasks']:
            with self.subTest(run=t['id']):
                a,b=profile(self.old,t)['training'],profile(self.c,t)['training']
                self.assertLessEqual({k for k in a if a[k]!=b[k]},{'batch','eval_batch','lr'})
    def test_reject_family_override(self):
        for m in ('AMD','J','N','S'):
            with self.subTest(model=m):
                c=copy.deepcopy(self.c);c['baseline_training_overrides'][m]={}
                with self.assertRaises(ValueError):profile(c,self.task())
    def test_reject_unauthorized_field(self):
        for k in ('T','epochs','patience','scheduler','structure','dtype'):
            with self.subTest(field=k):
                c=copy.deepcopy(self.c);self.entry(c)['values'][k]=1
                with self.assertRaises(ValueError):profile(c,self.task())
    def test_reject_invalid_value(self):
        for v in (0,-1,True,1.5,float('inf')):
            with self.subTest(value=str(v)):
                c=copy.deepcopy(self.c);self.entry(c)['values']['batch']=v
                with self.assertRaises(ValueError):profile(c,self.task())
    def test_require_source(self):
        c=copy.deepcopy(self.c);self.entry(c)['source']=[]
        with self.assertRaises(ValueError):profile(c,self.task())
    def test_pending_blocks(self):
        from utils.ch3_contract import training_blockers
        c=copy.deepcopy(self.c);c['baseline_training_overrides']['PatchTST']['ETTh1']['96']['pending']={'lr':'synthetic unresolved source'}
        self.assertTrue(training_blockers(c,self.task('PatchTST')))
        self.assertFalse(training_blockers(self.c,self.task()))
    def test_dlinear_horizons(self):
        for h,b in ((96,8),(192,8),(336,32),(720,32)):
            with self.subTest(H=h):
                tr=profile(self.c,self.task('DLinear','Exchange',h))['training']
                self.assertEqual((tr['batch'],tr['eval_batch'],tr['lr']),(b,b,.0005))
    def test_time_mixer_paper(self):
        for d,b in (('ETTh1',128),('Weather',128),('ECL',32)):
            with self.subTest(dataset=d):
                p=profile(self.c,self.task('TimeMixer',d));self.assertEqual((p['training']['batch'],p['training']['lr']),(b,.01))
    def test_itransformer_partial(self):
        from utils.ch3_contract import training_blockers
        p=profile(self.c,self.task('iTransformer'));self.assertEqual(p['training']['batch'],32)
        self.assertEqual(p['training']['lr'],1e-4);self.assertFalse(training_blockers(self.c,self.task('iTransformer')))
        self.assertEqual(profile(self.c,self.task('iTransformer','ECL'))['training']['batch'],16)
    def test_native_options_six_domains(self):
        from models.ch3_adapter import native_options
        for d in self.c['datasets']:
            with self.subTest(dataset=d):
                t=next(t for t in self.c['tasks']if t['model']=='iTransformer' and t['dataset']==d)
                p=profile(self.c,t);v=native_options(p);self.assertIs(v['output_attention'],False);self.assertEqual(v['seq_len'],p['T'])
                self.assertTrue({'use_norm','d_model','embed','freq','dropout','class_strategy','factor','n_heads','d_ff','activation','e_layers'}<=set(v))
    def test_uniform_T_epochs(self):
        for t in self.c['tasks']:
            with self.subTest(run=t['id']):
                a,b=profile(self.c,t),profile(self.old,t)
                self.assertEqual({k:v for k,v in a.items()if k!='training'},{k:v for k,v in b.items()if k!='training'})
                self.assertEqual((a['training']['epochs'],a['training']['patience']),(b['training']['epochs'],b['training']['patience']))
    def test_arithmetic_pjm(self):
        from utils.ch3_contract import step_arithmetic
        a=step_arithmetic(self.c,self.task('AMD','PJM',24))
        self.assertEqual((a['train_windows'],a['train_batches'],a['train_dropped'],a['validation_tail'],a['test_tail']),(36500,285,20,99,92))
    def test_arithmetic_urban(self):
        from utils.ch3_contract import step_arithmetic
        t=self.task('AMD','UrbanEV',3);a=step_arithmetic(self.c,t)
        self.assertEqual(a['train_windows'],(576-12-3+1)*275)
        self.assertEqual(a['max_optimizer_steps'],a['train_windows']//128*10)
    def test_old_report_rejected(self):
        from ch3_runner import validate_probe_report
        with self.assertRaises(ValueError):validate_probe_report(self.c,dict(protocol_sha=digest(self.old),Q=195,decisions={}))
    def test_old_identity_changed(self):
        self.assertNotEqual(digest(self.c),digest(self.old))
        self.assertNotEqual(digest(profile(self.c,self.task())),digest(profile(self.old,self.task())))
    def test_eval_profile_consistency(self):
        rows=json.loads((Path(self.c['execution']['evidence'])/'source-training-table.json').read_text())
        for row in rows:
            with self.subTest(model=row['model'],dataset=row['dataset'],H=row['H']):
                p=profile(self.c,self.task(row['model'],row['dataset'],row['H']))
                self.assertEqual({k:p['training'][k]for k in ('batch','eval_batch','lr')},row['effective'])
    def exit_fixture(self):
        from m5_formal_entry import ExitObservation
        s=dict(uuid='same',time=1.,nvml_processes={'101':12},owned_pid_metadata={'1':dict(host_pid=101,start_ticks='42')})
        x=ExitObservation(s);self.assertEqual(x.classify(s,[1],[1]),[])
        return x,s
    def test_exit_pending(self):
        x,s=self.exit_fixture();s.update(time=2.,owned_pid_metadata={})
        self.assertEqual(x.classify(s,[1],[1]),['101'])
        self.assertEqual(x.classify(s,[1],[]),['101']);self.assertTrue(x.pending)
        s['time']=6.
        with self.assertRaises(ValueError):x.classify(s,[1],[])
    def test_exit_resolved(self):
        x,s=self.exit_fixture();x.classify(s,[1],[]);s.update(time=2.,nvml_processes={})
        self.assertEqual(x.classify(s,[1],[]),[]);self.assertFalse(x.pending)
    def test_exit_unknown_rejected(self):
        from m5_formal_entry import resource_assessment
        x,s=self.exit_fixture();s.update(total=80*1024**3,used=1024**3,free=79*1024**3,driver_reserved=0,process_table_reliable=True,
                                      nvml_processes={'999':1024},owned_host_pids=[],process_gpu={})
        self.assertEqual(x.classify(s,[1],[1]),[])
        a=resource_assessment(s,[1],dict(s,nvml_processes={}))
        self.assertFalse(a['admission']);self.assertEqual(a['unknown_pids'],['999'])
        for changes in (dict(process_table_reliable=False),dict(free=0),dict(driver_reserved=999999999)):
            with self.subTest(changes=changes):
                self.assertFalse(resource_assessment(dict(s,**changes),[1],dict(s,nvml_processes={}))['admission'])
    def test_exit_reuse_rejected(self):
        x,s=self.exit_fixture();s['owned_pid_metadata']['1']['start_ticks']='43'
        with self.assertRaises(ValueError):x.classify(s,[1],[1])
        s['uuid']='other'
        with self.assertRaises(ValueError):x.classify(s,[1],[1])
    def test_growth_rule_preserved(self):
        import hashlib,torch
        from ch3_runner import tensor_digest
        source=(Path(__file__).resolve().parents[1]/'ch3_runner.py').read_text()
        self.assertIn("for key in ('allocated','rss_before_hash')",source)
        for a in (torch.tensor(1.),torch.arange(6).reshape(2,3).T,torch.zeros(0,2),torch.tensor([True,False])):
            with self.subTest(shape=tuple(a.shape),dtype=str(a.dtype)):
                v=a.contiguous();h=hashlib.sha256();h.update(str((v.dtype,tuple(v.shape))).encode());h.update(v.numpy().tobytes())
                self.assertEqual(tensor_digest(a),h.hexdigest())


class ModelTests(unittest.TestCase):
    def exercise(self):
        import torch
        from restricted_io_guard import require_installed
        from ch3_runner import init_training,update,evaluate,save_state,restore_state,tensor_digest,formal_identity
        s=require_installed();c=read_profiles();task=task_by_id(c,s['task']);out=Path(s['output'])
        p,model,opt,g=init_training(c,task,'cuda:0')
        x=torch.randn(2,p['T'],p['C'],generator=g);y=torch.randn(2,p['pred_len'],1,generator=g)
        before=tensor_digest(model.state_dict());update(model,opt,x,y,p,'cuda:0')
        self.assertNotEqual(before,tensor_digest(model.state_dict()))
        measured=evaluate(model,[(x,y),(x[:1],y[:1])],p,'cuda:0')
        best=BestState(p['training']['patience']);self.assertTrue(best.update(measured['mse'],1))
        identity=formal_identity(c,task,{'synthetic':True},{'commit':'synthetic-current-package'})
        identity['purpose']='ch3_model_acceptance'
        self.assertEqual(identity['input_variant'],task['input_variant'])
        save_state(out/'best.pt',model,opt,identity,best,1,1,g)
        save_state(out/'last.pt',model,opt,identity,best,1,1,g)
        state=tensor_digest((model.state_dict(),opt.state_dict()))
        with self.assertRaises(ValueError):restore_state(out/'last.pt',model,opt,dict(identity,purpose='M4'),g)
        restored,e,steps=restore_state(out/'last.pt',model,opt,identity,g)
        self.assertEqual((e,steps,restored.epoch),(1,1,1));self.assertEqual(state,tensor_digest((model.state_dict(),opt.state_dict())))
        self.assertEqual(measured,evaluate(model,[(x,y),(x[:1],y[:1])],p,'cuda:0'))
        update(model,opt,x,y,p,'cuda:0')
        self.assertNotEqual(state,tensor_digest((model.state_dict(),opt.state_dict())))
    def test_amd_f4_resume(self):self.exercise()
    def test_j_ecl_new_declaration(self):self.exercise()
    def test_time_mixer_native_width(self):self.exercise()
    def test_time_xer_true_ms(self):self.exercise()
    def test_itransformer_defaults(self):self.exercise()
    def test_time_mixer_ecl_batch(self):
        import torch
        from restricted_io_guard import require_installed
        from ch3_runner import init_training,update,evaluate,tensor_digest
        s=require_installed();c=read_profiles();t=task_by_id(c,s['task'])
        p,model,opt,g=init_training(c,t,'cuda:0');b=p['training']['batch']
        self.assertEqual((p['T'],b,p['training']['lr']),(512,32,.01))
        x=torch.randn(b,p['T'],p['C'],generator=g);y=torch.randn(b,p['pred_len'],1,generator=g)
        before=tensor_digest(model.state_dict())
        update(model,opt,x,y,p,'cuda:0');update(model,opt,x,y,p,'cuda:0')
        self.assertNotEqual(before,tensor_digest(model.state_dict()))
        self.assertTrue(any(param.grad is not None and torch.isfinite(param.grad).all()for param in model.parameters()))
        self.assertTrue(torch.isfinite(torch.tensor(evaluate(model,[(x,y)],p,'cuda:0')['mse'])))


class SourceClosureTests(unittest.TestCase):
    """Sixteen exact methods for this approved increment; no test-to-test calls."""
    def setUp(self):
        self.c=read_profiles();self.e=Path(self.c['execution']['evidence'])
        self.old=json.loads((self.e/'before/configs/ch3_formal_profiles.json').read_text())
    def one(self,m='AMD',d='ETTh1',h=96):
        return next(t for t in self.c['tasks'] if (t['model'],t['dataset'],t['h'])==(m,d,h))
    def test_counts_and_allowance(self):
        from utils.ch3_contract import followup_limits
        c=validate_manifest(self.c);f=c['execution']['followup']
        self.assertEqual((len(c['tasks']),len(c['groups']),sum(g['q'] for g in c['groups'])),(495,54,195))
        self.assertEqual((f['remaining_first_adam'],f['approved_extra_adam'],followup_limits(c)),(2327,709,3036))
        self.assertEqual((len(f['inherit_groups']),len(f['retest_groups']),f['Q']),(10,44,170))
    def test_family_unchanged(self):
        family=[t for t in self.c['tasks'] if t['model'] in ('AMD','J','N','S')]
        self.assertEqual(len(family),274)
        for t in family:
            with self.subTest(run=t['id']):self.assertEqual(profile(self.c,t),profile(self.old,t))
    def test_57_source_values(self):
        from utils.ch3_contract import training_blockers
        rows=json.loads((self.e/'source-decisions.json').read_text());self.assertEqual(len(rows),57)
        for row in rows:
            with self.subTest(model=row['model'],dataset=row['dataset'],H=row['H']):
                t=self.one(row['model'],row['dataset'],row['H']);p=profile(self.c,t)
                self.assertEqual({k:p['training'][k] for k in ('batch','eval_batch','lr')},row['after']['values'])
                self.assertFalse(training_blockers(self.c,t));self.assertFalse(row['after']['pending'])
                self.assertTrue(all(x['category']=='user-approved official-code supplement' for x in row['after']['source']))
        self.assertFalse([t['id'] for t in self.c['tasks'] if training_blockers(self.c,t)])
    def test_tasks_T_epochs_preserved(self):
        self.assertEqual(self.c['tasks'],self.old['tasks']);self.assertEqual(self.c['groups'],self.old['groups'])
        for key in ('datasets','amd','structures','training_common','urban_folds','urban_input_variants','sources'):
            with self.subTest(key=key):self.assertEqual(self.c[key],self.old[key])
        for t in self.c['tasks']:
            with self.subTest(run=t['id']):
                a,b=profile(self.c,t),profile(self.old,t)
                self.assertEqual({k:v for k,v in a.items() if k!='training'},{k:v for k,v in b.items() if k!='training'})
                self.assertLessEqual({k for k in a['training'] if a['training'][k]!=b['training'][k]},{'batch','eval_batch','lr'})
    def test_unauthorized_overrides_rejected(self):
        for model in ('AMD','J','N','S'):
            with self.subTest(model=model):
                c=copy.deepcopy(self.c);c['baseline_training_overrides'][model]={}
                with self.assertRaises(ValueError):profile(c,self.one())
        for key in ('T','epochs','patience','scheduler','precision'):
            with self.subTest(field=key):
                c=copy.deepcopy(self.c);c['baseline_training_overrides']['PatchTST']['ETTh1']['96']['values'][key]=1
                with self.assertRaises(ValueError):profile(c,self.one('PatchTST'))
    def test_optimizer_steps_and_tails(self):
        from utils.ch3_contract import step_arithmetic
        a=step_arithmetic(self.c,self.one('TimeXer','PJM',24))
        self.assertEqual((a['train_windows'],a['train_batches'],a['train_dropped'],a['validation_tail']),(36500,2281,4,3))
        self.assertEqual(sum(profile(self.c,t)['training']['epochs'] for t in self.c['tasks']),5340)
        for t in self.c['tasks']:
            with self.subTest(run=t['id']):
                a=step_arithmetic(self.c,t);p=profile(self.c,t)['training']
                self.assertEqual(a['max_optimizer_steps'],a['train_batches']*p['epochs'])
                self.assertLess(a['validation_tail'],p['eval_batch']);self.assertEqual(p['batch'],p['eval_batch'])
    def test_pending_and_old_report_rejected(self):
        from utils.ch3_contract import training_blockers
        from ch3_runner import validate_probe_report
        c=copy.deepcopy(self.c);c['baseline_training_overrides']['PatchTST']['ETTh1']['96']['pending']={'lr':'missing'}
        self.assertTrue(training_blockers(c,self.one('PatchTST')))
        with self.assertRaises(ValueError):validate_probe_report(self.c,dict(protocol_sha=digest(self.old),Q=195,decisions={}))
    def test_inherited_profiles_unchanged(self):
        for gid in self.c['execution']['followup']['inherit_groups']:
            for t in (x for x in self.c['tasks'] if x['group']==gid):
                with self.subTest(run=t['id']):self.assertEqual(profile(self.c,t),profile(self.old,t))
    def test_digest_parity(self):
        import torch,hashlib
        from ch3_runner import tensor_digest
        for a in (torch.tensor(1.),torch.arange(6).reshape(2,3).T,torch.zeros(0,2),torch.tensor([True,False])):
            with self.subTest(shape=tuple(a.shape)):
                v=a.contiguous();h=hashlib.sha256();h.update(str((v.dtype,tuple(v.shape))).encode());h.update(v.numpy().tobytes())
                self.assertEqual(tensor_digest(a),h.hexdigest())
    def test_growth_true_model_rejected(self):
        from ch3_runner import memory_growth_review
        for key in ('allocated','rss_before_hash'):
            rows=[dict(allocated=10,rss_before_hash=20,rss_after_hash=30) for _ in range(6)]
            for i,row in enumerate(rows):row[key]+=i
            with self.subTest(field=key):self.assertIn(key,memory_growth_review(rows)['triggers'])
    def test_growth_hash_only_not_gpu_leak(self):
        from ch3_runner import memory_growth_review
        rows=[dict(allocated=10,rss_before_hash=20,rss_after_hash=30+i) for i in range(6)]
        x=memory_growth_review(rows);self.assertFalse(x['blocked']);self.assertTrue(x['after_hash_grows'])
    def test_previous_hash_carryover_not_exonerated(self):
        from ch3_runner import memory_growth_review
        rows=[dict(allocated=10,rss_before_hash=20+i,rss_after_hash=30+i) for i in range(6)]
        x=memory_growth_review(rows);self.assertTrue(x['blocked']);self.assertIn('earlier hash',x['scope'])
    def test_budget_approval_and_excess_rejected(self):
        from utils.ch3_contract import followup_limits
        f=self.c['execution']['followup'];good=dict(followup_sha=digest(f),approved_extra_adam=709)
        self.assertEqual(followup_limits(self.c,good),3036)
        for bad in ({},dict(good,approved_extra_adam=0),dict(good,followup_sha='old')):
            with self.subTest(approval=bad):
                with self.assertRaises(ValueError):followup_limits(self.c,bad)
        c=copy.deepcopy(self.c);c['execution']['followup']['approved_extra_adam']=0
        with self.assertRaises(ValueError):followup_limits(c)
    def test_bad_budget_before_output(self):
        from m5_formal_entry import probe_all
        c=copy.deepcopy(self.c);root=Path(c['execution']['fixture'])/'rejected-probe-root'
        c['execution']['evidence']=str(root);c['execution']['followup']['approved_extra_adam']=0
        self.assertFalse(root.exists())
        with patch('m5_formal_entry.preflight',return_value=[]):
            with self.assertRaises(ValueError):probe_all(c,{})
        self.assertFalse(root.exists())
    def test_diagnostic_capability_not_general_probe(self):
        import m5_formal_entry as entry
        d=self.c['execution']['diagnostics']
        self.assertEqual(d['total_limits'],dict(adam=32,forward=48,backward=32))
        self.assertEqual(set(d['cases']),{'amd_rss','timemixer_exact'})
        s=dict(protocol_sha=digest(self.c),bound_files=entry.repository_files(),session_root=self.c['execution']['evidence'],fixture_root=self.c['execution']['fixture'],purpose='ch3_step_diagnostic',case='not_authorized',task='foreign',limits={'seconds':180})
        with patch.dict(os.environ,TMPDIR=s['fixture_root']):
            with self.assertRaisesRegex(ValueError,'exact current-package'):entry.validate_config(s)
    def test_exit_transition_positive_and_negative(self):
        from m5_formal_entry import ExitObservation
        sample=dict(uuid='gpu',time=1.,nvml_processes={'101':12},owned_pid_metadata={'1':dict(host_pid=101,start_ticks='42')})
        obs=ExitObservation(sample);self.assertEqual(obs.classify(sample,[1],[1]),[])
        stale=dict(sample,time=2.,owned_pid_metadata={});self.assertEqual(obs.classify(stale,[1],[]),['101'])
        self.assertEqual(obs.classify(dict(stale,time=3.,nvml_processes={}),[1],[]),[]);self.assertFalse(obs.pending)
        with self.assertRaises(ValueError):obs.classify(dict(sample,uuid='other'),[1],[1])
        reused=copy.deepcopy(sample);reused['owned_pid_metadata']['1']['start_ticks']='43'
        with self.assertRaises(ValueError):obs.classify(reused,[1],[1])
