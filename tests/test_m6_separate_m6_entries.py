"""Physical entries remain isolated after the explicit dm4 user selection."""
import ast,contextlib,copy,io,json,os,subprocess,sys,unittest
from pathlib import Path
from unittest.mock import patch
from utils import ch3_patchtst_width_reproduction as a,ch3_third_round_after_selection as b,ch3_type1_chain as q
from utils.ch3_contract import ROOT,profile
from utils.ch3_native_recovery_records import bound,ref
b.activate()


class SeparateEntries(unittest.TestCase):
    def test_distinct_scripts_python_entries_namespaces_no_cross_call(self):
        for x,y in((a.WRAPPER,b.WRAPPER),(a.ENTRY,b.ENTRY),(a.RESULT,b.RESULT),(a.LOG,b.LOG),(a.SESSION,b.SESSION)):self.assertNotEqual(x,y)
        self.assertNotIn(b.ENTRY.name,a.WRAPPER.read_text());self.assertNotIn(a.ENTRY.name,b.WRAPPER.read_text())
        for name,forbidden in((a.ENTRY,'ch3_third_round_after_selection'),(b.ENTRY,'ch3_patchtst_width_reproduction')):
            self.assertFalse(any(forbidden in(n.module or'')for n in ast.walk(ast.parse(name.read_text()))if isinstance(n,ast.ImportFrom)))

    def test_B_source_stage_counts_and_skip_completed_depth(self):
        v=b.contract();self.assertEqual(v['stage_order'],['URBAN_SUBSET','EPF_ALL','M_ALL']);self.assertEqual(v['stage_runs'],dict(URBAN_SUBSET=84,EPF_ALL=35,M_ALL=168));self.assertTrue(v['no_depth_redispatch']);self.assertTrue(v['no_A_dispatch'])
        self.assertFalse(any(x in v['stage_order']for x in('PATCH_ENC1','PATCH_ENC2','M_BASE','M_AMEND')))
        self.assertEqual(v['remaining_formal_runs'],287)

    def test_B_no_executable_default_after_explicit_selection(self):
        # Selection is now explicit; executable permission and closure remain separate gates.
        v=b.status();self.assertEqual(v['selected_nonWeather_d_model'],4);self.assertFalse(v['permission_exists']);self.assertFalse(v['READY_FOR_GPU_EXECUTION'])
        c=bound(b.contract()['config_refs']['M_ALL'])
        for t in c['tasks']:
            if t['model']=='PatchTST'and t['dataset']=='Weather':self.assertEqual([profile(c,t)['structure'][k]for k in('d_model','n_heads','e_layers','d_ff')],[128,16,2,256])

    def test_B_launch_requires_exact_reviewed_closure_before_GPU_or_writes(self):
        with patch.object(q,'run',side_effect=AssertionError('training launch forbidden')),patch.object(Path,'mkdir',side_effect=AssertionError('no result materialization')):
            # Actual closure validator rejects this uncommitted candidate. No GPU preflight is run.
            self.assertRaises((ValueError,PermissionError),q.validate_start,None)
            from utils.ch3_event_resources import prestart
            with self.assertRaises((ValueError,PermissionError)):
                with prestart(None):self.fail('invalid permission cannot reach GPU')

    def test_B_never_accepts_provided_old_A_approval(self):
        old=bound(ref(a.PACKAGE/'separate-entrypoints-v1/start-review.json'))
        with patch.object(q,'closure',return_value=old['closure_commit']):self.assertRaises(PermissionError,q.validate_start,old)

    def test_B_cannot_change_selection_by_environment_or_fake_contract(self):
        with patch.dict(os.environ,{'M6_SELECTED_D_MODEL':'8','PATCHTST_D_MODEL':'16'}):self.assertEqual(b.contract()['selected_d_model'],4)
        read=b.bound
        for width in(8,16):
            bad=read(b.CONTRACT_REF);bad['selected_d_model']=width
            with patch.object(b,'bound',side_effect=lambda r:bad if r==b.CONTRACT_REF else read(r)):self.assertRaises(PermissionError,b.contract)
        for key,value in(('execution_permitted',True),('Weather_d_model',16),('remaining_formal_runs',335)):
            bad=read(b.CONTRACT_REF);bad[key]=value
            with patch.object(b,'bound',side_effect=lambda r:bad if r==b.CONTRACT_REF else read(r)):self.assertRaises(PermissionError,b.contract)

    def test_B_complete_is_pending_and_never_calls_A_or_scheduler(self):
        with patch.object(a,'activate',side_effect=AssertionError('no A activation')),patch.object(q,'run',side_effect=AssertionError('no scheduler')),patch.object(a,'width_results',side_effect=AssertionError('no metric selection')):
            out=io.StringIO()
            with contextlib.redirect_stdout(out):code=b.cli(['complete'])
            self.assertEqual(code,2);self.assertFalse(json.loads(out.getvalue())['complete'])

    def test_B_actual_shell_status_no_GPU_tmux_checkpoint_or_result(self):
        # Status is read-only and is not GPU preflight.
        r=subprocess.run(['bash',str(b.WRAPPER),'status'],cwd=ROOT,capture_output=True,text=True,timeout=30,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''})
        self.assertEqual(r.returncode,0,r.stderr);v=json.loads(r.stdout);self.assertFalse(v['running']);self.assertFalse(v['complete']);self.assertFalse(b.RESULT.exists());self.assertFalse(b.LOG.exists());self.assertFalse((b.PACKAGE/'start-review.json').exists())

    def test_A_actual_shared_driver_complete40_cannot_dispatch_B(self):
        # Retain the historical real shared-driver isolation test in a separate
        # process, as A and B activation belongs to distinct physical processes.
        old=bound(ref(b.PACKAGE/'historical-test-source.json'))['source'];tree=ast.parse(old)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));method=next(n for n in cls.body if n.name==self._testMethodName)
        header=old[:old.index('class SeparateEntries')];lines=old.splitlines(True);body=''.join(lines[method.lineno-1:method.end_lineno])
        child=header+'class IsolationTest(unittest.TestCase):\n'+body+'\nif __name__=="__main__": unittest.main()\n'
        r=subprocess.run([sys.executable,'-B','-c',child],cwd=ROOT,capture_output=True,text=True,timeout=60,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
        self.assertEqual(r.returncode,0,r.stdout+r.stderr);self.assertIn('Ran 1 test',r.stderr)


if __name__=='__main__':unittest.main()
