"""A forty-run driver stops; B is physically separate and cannot execute yet."""
import ast,contextlib,copy,io,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from utils import ch3_patchtst_width_reproduction as a,ch3_third_round_after_selection as b
from utils import ch3_type1_chain as q,ch3_type1_tasks as tasks
from utils.ch3_contract import ROOT,digest,profile
from utils.ch3_native_recovery_records import bound,exclusive

a.activate()


class SeparateEntries(unittest.TestCase):
    def test_distinct_scripts_python_entries_namespaces_no_cross_call(self):
        self.assertNotEqual(a.WRAPPER,b.WRAPPER);self.assertNotEqual(a.ENTRY,b.ENTRY)
        self.assertNotEqual(a.RESULT,b.RESULT);self.assertNotEqual(a.LOG,b.LOG);self.assertNotEqual(a.SESSION,b.SESSION)
        left=a.WRAPPER.read_text();right=b.WRAPPER.read_text()
        self.assertNotIn(b.ENTRY.name,left);self.assertNotIn(b.WRAPPER.name,left)
        self.assertNotIn(a.ENTRY.name,right);self.assertNotIn(a.WRAPPER.name,right)
        for name,forbidden in((a.ENTRY,'ch3_third_round_after_selection'),(b.ENTRY,'ch3_patchtst_width_reproduction')):
            parsed=ast.parse(name.read_text());imports=[x.module for x in ast.walk(parsed)if isinstance(x,ast.ImportFrom)]
            self.assertFalse(any(forbidden in(x or'')for x in imports))

    def test_B_source_stage_counts_and_skip_completed_depth(self):
        v=b.preparation();self.assertEqual(v['stage_order'],['URBAN_SUBSET','EPF_ALL','M_ALL']);self.assertEqual(v['stage_runs'],dict(URBAN_SUBSET=84,EPF_ALL=35,M_ALL=168))
        source=bound(a.SOURCE_REF);self.assertEqual(v['source_ref'],a.SOURCE_REF)
        self.assertEqual(v['r6_Urban_complete_ref'],source['refs']['probe/URBAN_SUBSET/complete.json'])
        self.assertEqual(v['retained_depth_boundaries'],{k:source['refs'][k]for k in('PATCH_ENC1','PATCH_ENC2')})
        self.assertFalse(any(x in v['stage_order']for x in('PATCH_ENC1','PATCH_ENC2','M_BASE','M_AMEND')))
        self.assertEqual(v['third_formal'],287)
        state=b.status();self.assertEqual(state['planned_resume_at'],'URBAN_AUTO_AUDIT_AFTER_REVIEWED_SOURCE_ADOPTION');self.assertFalse(state['redispatch_depth'])

    def test_B_no_selection_or_executable_default_config(self):
        v=b.status();self.assertIsNone(v['selected_nonWeather_d_model']);self.assertIsNone(v['executable_config_refs']);self.assertFalse(v['execution_permitted']);self.assertFalse(v['READY_TO_ARM_HANDOFF']);self.assertFalse(v['READY_FOR_GPU_EXECUTION'])
        self.assertEqual(v['future_main_index'],dict(total=371,switched_nonWeather_PatchTST=20,preserved_Weather=4,preserved_other=347,preserved_total=351))
        c=bound(b.preparation()['parent_config_refs']['M_ALL'])
        for t in c['tasks']:
            if t['model']=='PatchTST'and t['dataset']=='Weather':self.assertEqual([profile(c,t)['structure'][k]for k in('d_model','n_heads','e_layers','d_ff')],[128,16,2,256])

    def test_B_all_launch_permission_GPU_actions_refused_no_subprocess_or_writes(self):
        # Status reads immutable preparation records only; an execution request
        # cannot reach any process launcher, GPU subprocess, mkdir or writer.
        with patch.object(subprocess,'Popen',side_effect=AssertionError('process launch forbidden')),patch.object(subprocess,'run',side_effect=AssertionError('GPU/process query forbidden')),patch.object(Path,'mkdir',side_effect=AssertionError('no materialization')),patch.object(Path,'write_text',side_effect=AssertionError('no write')):
            for action in b.EXECUTION_ACTIONS:
                out=io.StringIO()
                with contextlib.redirect_stdout(out):code=b.cli([action,'--approval','/not-read/old-A-permission.json','--approval-sha','f'*64])
                self.assertEqual(code,2);v=json.loads(out.getvalue());self.assertFalse(v['GPU_preflight_executed']);self.assertFalse(v['executable_permission_bound']);self.assertFalse(v['READY_TO_ARM_HANDOFF'])

    def test_B_never_reads_provided_old_A_approval(self):
        original=b.bound;reads=[]
        def observe(value):reads.append(value['path']);return original(value)
        out=io.StringIO()
        with patch.object(b,'bound',side_effect=observe),contextlib.redirect_stdout(out):self.assertEqual(b.cli(['arm','--approval',str(a.PACKAGE/'start-review.json'),'--approval-sha','0'*64]),2)
        self.assertNotIn(str(a.PACKAGE/'start-review.json'),reads)

    def test_B_cannot_unfreeze_by_argument_environment_or_fake_preparation(self):
        with patch.dict(os.environ,{'M6_SELECTED_D_MODEL':'4','PATCHTST_D_MODEL':'8','M6_THIRD_EXECUTION_PERMITTED':'true'}):self.assertIsNone(b.status()['selected_nonWeather_d_model'])
        for width in(4,8,16):
            v=b.preparation();v['selected_nonWeather_d_model']=width
            with patch.object(b,'bound',return_value=v):self.assertRaises(PermissionError,b.preparation)
        for key,value in(('execution_permitted',True),('reviewed_closure','f'*40),('start_authorization_ref',{'path':'x'}),('reviewed_config_refs',{}),('Weather_d_model',16)):
            v=b.preparation();v[key]=value
            with patch.object(b,'bound',return_value=v):self.assertRaises(PermissionError,b.preparation)
        with contextlib.redirect_stderr(io.StringIO()):self.assertRaises(SystemExit,b.cli,['arm','--selected-d-model','4'])

    def test_B_complete_is_pending_and_never_calls_A_or_scheduler(self):
        with patch.object(a,'activate',side_effect=AssertionError('A activation forbidden')),patch.object(q,'run',side_effect=AssertionError('scheduler forbidden')),patch.object(a,'width_results',side_effect=AssertionError('A results not automatically selected')):
            out=io.StringIO()
            with contextlib.redirect_stdout(out):code=b.cli(['complete'])
            self.assertEqual(code,2);self.assertEqual(json.loads(out.getvalue())['state'],'NON_EXECUTABLE_WAIT_USER_SELECTION_AND_REVIEW')

    def test_B_actual_shell_actions_no_GPU_tmux_checkpoint_or_result(self):
        with tempfile.TemporaryDirectory(prefix='m6-B-nonexecute-')as raw:
            root=Path(raw);marker=root/'forbidden-called';binpath=root/'bin';binpath.mkdir()
            for name in('nvidia-smi','tmux'):
                p=binpath/name;p.write_text(chr(10).join(('#!/bin/sh','printf called >> '+str(marker),'exit 77','')));p.chmod(0o700)
            # Executable fixture files are temporary CPU test tools; repository modes are unchanged.
            for action in('arm','start','preflight','prepare-permission','complete','status'):
                r=subprocess.run(['bash',str(b.WRAPPER),action,'--approval','/missing/old-A.json','--approval-sha','f'*64],cwd=ROOT,capture_output=True,text=True,timeout=30,env={**os.environ,'PATH':str(binpath)+':'+os.environ['PATH'],'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''})
                self.assertEqual(r.returncode,0 if action=='status'else 2,r.stderr);v=json.loads(r.stdout);self.assertFalse(v['GPU_preflight_executed']);self.assertIsNone(v['selected_nonWeather_d_model'])
            self.assertFalse(marker.exists());self.assertFalse(b.RESULT.exists());self.assertFalse(b.LOG.exists());self.assertFalse((b.PACKAGE/'third-round-start-review.json').exists())

    def test_A_actual_shared_driver_complete40_cannot_dispatch_B(self):
        # Only stage computations are synthetic; the shared driver, fixed state
        # order, completion creation and physical A/B separation are real.
        with tempfile.TemporaryDirectory(prefix='m6-A-complete-isolation-')as raw,contextlib.ExitStack()as stack:
            root=Path(raw);stack.enter_context(patch.object(tasks,'RESULT',root/'results'));stack.enter_context(patch.object(q,'CONTROL',root/'controller'));q.CONTROL.mkdir()
            start=exclusive(root/'synthetic-start.json',{'synthetic':True});events=[]
            stack.enter_context(patch.object(q,'wait_upstream',side_effect=lambda:exclusive(root/'synthetic-upstream.json',{'synthetic':True})))
            stack.enter_context(patch.object(q,'validate_start',side_effect=lambda v:v));stack.enter_context(patch.object(q,'stop_check'))
            def permit(c,start_ref,probe,*args,**kwargs):
                stage=c['baseline_unified']['stage'];events.append((stage,'probe'if probe else'formal'))
                return exclusive(root/(stage+('-probe'if probe else'-formal')+'.json'),{'synthetic':True,'stage':stage})
            def wait(stage,value,probe,model=None,runtime=None):
                if not probe:exclusive(tasks.context(q.configs()[stage])['control']/('group-'+model)/'complete.json',{'synthetic':True})
            stack.enter_context(patch.object(q,'create_permit',side_effect=permit));stack.enter_context(patch.object(q,'wait_owned',side_effect=wait))
            stack.enter_context(patch.object(q,'audit_probe',side_effect=lambda c:exclusive(root/(c['baseline_unified']['stage']+'-audit.json'),{'synthetic':True})))
            stack.enter_context(patch.object(q,'seal_runtime',side_effect=lambda c,value:value))
            stack.enter_context(patch.object(q,'seal_boundary',side_effect=lambda c,r:exclusive(root/(c['baseline_unified']['stage']+'-boundary.json'),{'synthetic':True,'tasks':len(c['tasks'])})))
            stack.enter_context(patch.object(b,'cli',side_effect=AssertionError('no B dispatch')));stack.enter_context(patch.object(subprocess,'Popen',side_effect=AssertionError('no automatic launch')))
            q.run(start);complete=bound({'path':str(q.CONTROL/'complete.json'),'sha256':__import__('hashlib').sha256((q.CONTROL/'complete.json').read_bytes()).hexdigest()})
            self.assertEqual(events,[('PATCH_DM4','probe'),('PATCH_DM4','formal'),('PATCH_DM8','probe'),('PATCH_DM8','formal')]);self.assertEqual(complete['total_runs'],40);self.assertEqual(complete['executed_new_formal_runs'],40);self.assertTrue(complete['third_round_paused']);self.assertTrue(complete['awaiting_user_width_selection']);self.assertFalse(complete['automatic_adoption'])
            self.assertEqual(set(complete['boundaries']),{'PATCH_DM4','PATCH_DM8'});self.assertFalse(any('URBAN' in x for x in q.STATES));self.assertFalse(b.RESULT.exists())


if __name__=='__main__':unittest.main()
