import contextlib
import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from ch4 import queue,reporting
from ch4.runtime_io import read,write_new
from ch4.imports import sha
from ch4 import lifecycle


class Supervisor(unittest.TestCase):
    def fixture(self,base,failure=None):
        config=dict(scope='synthetic-only',fixture_only=True,environment={'fixture_only':True},models=['fixture'],tasks=[
            dict(id=f'fixture-H{h}',model='fixture',domain='UrbanEV',H=h,P=1,seed=2024,metadata_sha256='synthetic-only') for h in (3,6,9,12)])
        if failure:config['tasks'][1]['fail_phase']=failure
        auth=dict(commit='synthetic-parent',resource_plan=dict(global_lock=str(base/'fixture.lock'),waves=[[t['id']] for t in config['tasks']]))
        write_new(base/'auth.json',auth);return config,auth

    def execute(self,base,config,auth,*,seal_hook=None,checksum_failure=False,summary_tamper=False):
        real_popen=subprocess.Popen;real_seal=reporting.seal;real_csv=reporting.csv_new
        def process(command,**kwargs):
            command=list(command)
            if 'ch4.queue' in command:command[command.index('ch4.queue')]='tests.ch4_synthetic_worker'
            return real_popen(command,**kwargs)
        def seal(c,folder):
            if seal_hook:seal_hook(c,folder)
            return real_seal(c,folder)
        def csv(path,*args):
            real_csv(path,*args)
            if summary_tamper and Path(path).name=='comparison-standardized.csv':
                with Path(path).open('a') as f:f.write('tampered-summary\n')
        with contextlib.ExitStack() as stack:
            # Only external admission and synthetic scope are replaced. Actual
            # supervise, subprocess lifecycle, freeze/claim, seal and sha256sum run.
            stack.enter_context(patch.object(queue,'admission',return_value=(config,auth)))
            stack.enter_context(patch.object(queue.subprocess,'Popen',side_effect=process))
            stack.enter_context(patch.object(reporting,'validate_plan',side_effect=lambda c:self.assertTrue(c['fixture_only'])))
            stack.enter_context(patch.object(reporting,'seal',side_effect=seal))
            stack.enter_context(patch.object(reporting,'csv_new',side_effect=csv))
            if checksum_failure:
                stack.enter_context(patch.object(reporting.subprocess,'run',side_effect=subprocess.CalledProcessError(1,['sha256sum','-c'])))
            queue.supervise(base/'unused-config.json',base/'auth.json',base/'attempt')

    def temporary(self):
        root=Path(os.environ.get('M7_FIXTURE_ROOT',tempfile.gettempdir()))
        return tempfile.TemporaryDirectory(dir=root,prefix='synthetic-supervisor-')

    def test_success_full_supervisor_then_seal_and_claim_refusal(self):
        with self.temporary() as tmp:
            base=Path(tmp);config,auth=self.fixture(base);self.execute(base,config,auth)
            folder=base/'attempt';names=[t['id'] for t in config['tasks']]
            events=(folder/'fixture-events.jsonl').read_text().splitlines()
            self.assertEqual(events,['train '+n for n in names]+['test '+n for n in names])
            complete=read(folder/'complete.json');self.assertEqual((complete['training_tasks'],complete['formal_test_evaluations']),(4,4))
            self.assertEqual(len(list((folder/'receipts').iterdir())),8)
            manifest=read(folder/'manifest.json')['files']
            self.assertIn('comparison-standardized.csv',manifest);self.assertIn('source-ledger.json',manifest)
            for excluded in ('manifest.json','checksums.sha256','complete.json'):self.assertNotIn(excluded,manifest)
            self.assertEqual(complete['manifest_sha256'],sha(folder/'manifest.json'))
            self.assertEqual(complete['checksums_sha256'],sha(folder/'checksums.sha256'))
            self.assertEqual(complete['system_checksum_returncode'],0)
            self.assertFalse((folder/'complete.json.staging').exists())
            self.assertEqual(list((folder/'failures').glob('*.json')),[])
            self.assertEqual(len(list((folder/'lifecycles').glob('*-end.json'))),8)
            with self.assertRaises(FileExistsError):queue.claim_test(folder,names[0])
            # Ambiguous claim contents still prohibit evaluation, without parsing it.
            (folder/'claims'/(names[1]+'.json')).write_text('interrupted unknown claim')
            with self.assertRaises(FileExistsError):queue.claim_test(folder,names[1])
            self.assertEqual((folder/'fixture-events.jsonl').read_text().splitlines(),events)

    def test_technical_failure_stops_later_dispatch(self):
        with self.temporary() as tmp:
            base=Path(tmp);config,auth=self.fixture(base,'train')
            with self.assertRaisesRegex(RuntimeError,'technical failure'):self.execute(base,config,auth)
            folder=base/'attempt'
            self.assertEqual((folder/'fixture-events.jsonl').read_text().splitlines(),['train fixture-H3','train fixture-H6'])
            self.assertTrue((folder/'failure.json').exists());self.assertTrue((folder/'fixture-H3/train-result.json').exists())
            self.assertFalse((folder/'frozen-best.json').exists());self.assertFalse((folder/'complete.json').exists())
            self.check_failure(folder,'fixture-H6',reason='worker_exception')

    def check_failure(self,folder,name,reason,complete=True,returncode=1):
        record=read(folder/'failures'/(name+'-train.json'))
        self.assertEqual(record['failure_reason'],reason);self.assertEqual(record['returncode'],returncode)
        self.assertEqual((record['completed_updates'],record['completed_epochs']),(3,1))
        self.assertEqual((record['last_confirmed_total_update'],record['last_completed_epoch']),(103,7))
        self.assertEqual(record['accounting_complete'],complete)
        self.assertEqual(bool(record['accounting_errors']),not complete)
        self.assertGreater(record['elapsed_seconds'],0)
        self.assertGreaterEqual(record['ended_at']['unix'],record['started_at']['unix'])
        for key in ('worker_log','access_log','history'):
            self.assertEqual(record[key+'_sha256'],sha(record[key+'_path']))
        self.assertEqual(record['config_sha256'],sha(folder/'config.json'))
        self.assertEqual(record['runtime_sha256'],sha(folder/'runtime.json'))
        end=read(folder/'lifecycles'/(name+'-train-end.json'))
        self.assertEqual(end['failure_receipt']['sha256'],sha(folder/'failures'/(name+'-train.json')))
        failure=read(folder/'failure.json')
        self.assertIn(end['failure_receipt'],failure['failure_receipts'])
        self.assertGreater(failure['supervisor_elapsed_seconds'],0)
        self.assertFalse((folder/'complete.json').exists())
        with self.assertRaises(FileExistsError):write_new(folder/'failures'/(name+'-train.json'),{})
        return record

    def test_stop_corrupt_missing_receipt_and_signal_costs(self):
        for mode,reason,code,complete in [('stop','stop_interruption',1,True),('corrupt','worker_exception',1,False),
                                         ('missing_receipt','missing_success_receipt',0,True),('signal','signal_exit',-15,True)]:
            with self.subTest(mode=mode),self.temporary() as tmp:
                base=Path(tmp);config,auth=self.fixture(base);config['tasks'][0]['failure_mode']=mode
                expected=InterruptedError if mode=='stop' else RuntimeError
                with self.assertRaisesRegex(expected,'STOP' if mode=='stop' else 'technical failure'):self.execute(base,config,auth)
                folder=base/'attempt';record=self.check_failure(folder,'fixture-H3',reason,complete,code)
                self.assertEqual((folder/'fixture-events.jsonl').read_text().splitlines(),['train fixture-H3'])
                if mode=='stop':self.assertEqual(record['stop_record']['record']['reason'],'synthetic user STOP')
                if mode=='signal':self.assertIsNone(record['exception_type'])

    def test_same_wave_failure_stops_peer_and_later_wave(self):
        with self.temporary() as tmp:
            base=Path(tmp);config,auth=self.fixture(base)
            config['tasks'][0]['failure_mode']='exception';config['tasks'][1]['failure_mode']='wait_stop'
            auth['resource_plan']['waves']=[['fixture-H3','fixture-H6'],['fixture-H9'],['fixture-H12']]
            with self.assertRaisesRegex(RuntimeError,'technical failure'):self.execute(base,config,auth)
            folder=base/'attempt';self.check_failure(folder,'fixture-H3','worker_exception');self.check_failure(folder,'fixture-H6','stop_interruption')
            self.assertEqual(set((folder/'fixture-events.jsonl').read_text().splitlines()),{'train fixture-H3','train fixture-H6'})
            self.assertEqual(len(read(folder/'failure.json')['failure_receipts']),2)

    def test_invalid_success_receipt_structure_is_accounted_once(self):
        for invalid in ({'invalid_files':[]},{'invalid_files':None},{'invalid_root':[]},{'invalid_root':None}):
            with self.subTest(invalid=invalid),self.temporary() as tmp:
                base=Path(tmp);config,auth=self.fixture(base)
                config['tasks'][0].update(failure_mode='invalid_receipt',**invalid)
                with self.assertRaisesRegex(RuntimeError,'technical failure'):self.execute(base,config,auth)
                folder=base/'attempt';record=self.check_failure(folder,'fixture-H3','invalid_success_receipt',returncode=0)
                self.assertEqual(record['exception_type'],'builtins.ValueError')
                self.assertEqual(record['exception_message'],'receipt root must be a dict' if 'invalid_root' in invalid else 'receipt files must be a dict')
                self.assertEqual(len(read(folder/'failure.json')['failure_receipts']),1)
                self.assertEqual(len(list((folder/'failures').glob('*.json'))),1)
                self.assertEqual((folder/'fixture-events.jsonl').read_text().splitlines(),['train fixture-H3'])

    def test_sequence_gaps_keep_confirmed_counts_and_offset_baseline(self):
        with self.temporary() as tmp:
            base=Path(tmp);config,auth=self.fixture(base);config['tasks'][0]['failure_mode']='sequence_gap'
            with self.assertRaisesRegex(RuntimeError,'technical failure'):self.execute(base,config,auth)
            folder=base/'attempt';record=read(folder/'failures/fixture-H3-train.json')
            self.assertEqual((record['completed_updates'],record['completed_epochs']),(2,2))
            self.assertEqual((record['last_confirmed_total_update'],record['last_completed_epoch']),(103,9))
            self.assertFalse(record['accounting_complete'])
            self.assertEqual(record['accounting_errors'],[
                'access: new record 2: missing updates range 102..102 (between 101 and 103)',
                'history: new record 2: missing epochs range 8..8 (between 7 and 9)'])
            self.assertTrue((folder/'failure.json').exists());self.assertFalse((folder/'complete.json').exists())
            for kind,rows in [('updates',[{'event':'optimizer_completed','updates':101},{'event':'optimizer_completed','updates':102}]),
                              ('epochs',[{'epoch':7,'updates':102},{'epoch':8,'updates':104}])]:
                path=base/(kind+'.jsonl');path.write_text('{"epoch":1,"updates":1}\n')
                position=lifecycle.log_position(path)
                with path.open('a') as stream:
                    for row in rows:stream.write(json.dumps(row)+'\n')
                result=lifecycle.progress(path,position,kind)
                self.assertEqual(result['count'],2);self.assertEqual(result['errors'],[])

    def test_supervisor_exception_waits_and_accounts_launched_worker(self):
        real_attach=lifecycle.attach
        def fail_after_attach(folder,record,child):
            real_attach(folder,record,child)
            raise ValueError('synthetic supervisor exception after launch')
        with self.temporary() as tmp:
            base=Path(tmp);config,auth=self.fixture(base);config['tasks'][0]['failure_mode']='wait_stop'
            with patch.object(lifecycle,'attach',side_effect=fail_after_attach):
                with self.assertRaisesRegex(ValueError,'supervisor exception'):self.execute(base,config,auth)
            folder=base/'attempt';self.check_failure(folder,'fixture-H3','stop_interruption')
            self.assertEqual(read(folder/'failure.json')['failure_type'],'builtins.ValueError')

    def test_progress_offsets_and_corrupt_history(self):
        with self.temporary() as tmp:
            path=Path(tmp)/'history.jsonl';path.write_text('{"epoch":1,"updates":2}\n')
            start=lifecycle.log_position(path)
            with path.open('a') as f:f.write('{"epoch":7,"updates":102}\ninvalid\n{"epoch":')
            value=lifecycle.progress(path,start,'epochs')
            self.assertEqual((value['count'],value['last'],value['max_update']),(1,7,102));self.assertEqual(len(value['errors']),2)
            path.write_text('')
            self.assertIn('truncated',lifecycle.progress(path,start,'epochs')['errors'][0])
            path.write_text('{"event":"optimizer_completed","updates":101}\n{"updates":102}\n')
            value=lifecycle.progress(path,dict(offset=0,inode=None),'updates')
            self.assertEqual((value['count'],value['max_update']),(1,101));self.assertEqual(len(value['errors']),1)

    def test_missing_tampered_payload_and_checksum_failure_no_complete(self):
        def missing(c,f):(f/c['tasks'][0]['id']/'test_predictions/truth.npy').unlink()
        def tamper(c,f):(f/c['tasks'][0]['id']/'last.pt').write_bytes(b'changed synthetic bytes')
        for kind,hook in [('missing',missing),('tamper',tamper),('checksum',None),('summary',None)]:
            with self.subTest(kind=kind),self.temporary() as tmp:
                base=Path(tmp);config,auth=self.fixture(base)
                with self.assertRaises((ValueError,FileNotFoundError,subprocess.CalledProcessError)):
                    self.execute(base,config,auth,seal_hook=hook,checksum_failure=kind=='checksum',summary_tamper=kind=='summary')
                folder=base/'attempt';self.assertFalse((folder/'complete.json').exists())
                self.assertTrue((folder/'failure.json').exists());self.assertEqual(len(list((folder/'receipts').iterdir())),8)
                if kind=='checksum':self.assertTrue((folder/'checksums.sha256').exists())

    def test_production_seal_rejects_fixture_scope_without_mock(self):
        with self.temporary() as tmp:
            config,_=self.fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError,'exact216'):reporting.seal(config,Path(tmp)/'attempt')

    def test_actual_system_checksum_rejects_corrupted_control(self):
        real_run=subprocess.run
        def corrupt_then_check(command,**kwargs):
            path=Path(kwargs['cwd'])/'checksums.sha256';text=path.read_text()
            path.write_text(('0' if text[0]!='0' else '1')+text[1:])
            return real_run(command,**kwargs)
        with self.temporary() as tmp:
            base=Path(tmp);config,auth=self.fixture(base)
            with patch.object(reporting.subprocess,'run',side_effect=corrupt_then_check):
                with self.assertRaises(subprocess.CalledProcessError) as error:self.execute(base,config,auth)
            self.assertNotEqual(error.exception.returncode,0)
            self.assertFalse((base/'attempt/complete.json').exists())
            self.assertTrue((base/'attempt/checksums.sha256').exists())


if __name__=='__main__':unittest.main()
