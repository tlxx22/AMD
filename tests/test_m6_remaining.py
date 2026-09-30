"""Exact no-model remaining-queue tests; fixtures never enter formal roots."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from utils.ch3_contract import read_profiles,digest,profile,task_by_id
from utils import ch3_remaining as q
import m6_remaining_entry as entry

class RemainingTests(unittest.TestCase):
    def setUp(self):
        self.c=read_profiles();self.decisions=q.read(q.PARENT)['decisions'];self.plan=q.plan(self.c,self.decisions)
        self.tmp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']);self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def child(self,m):
        g=next(g for g in self.plan['groups']if g['model']==m)
        a=dict(queue_id=q.QUEUE,models=[m],authorized_task_ids=g['task_ids'],profile_shas=g['profile_shas'],reviewed=True,execution_permitted=True,run_budget={'runs':g['runs'],'run_epochs':g['run_epochs']},max_optimizer_steps=g['max_optimizer_steps'],seed_list=[2024],additional_search=0,data_bindings={})
        for r in g['task_ids']:a['data_bindings'].setdefault(task_by_id(self.c,r)['dataset'],{})[r]='a'*64
        if m in q.MODELS[:4]:a['extension_batch']=self.c['extension']['id']
        return a
    def small(self):return dict(groups=[{'model':'one'},{'model':'two'}],task_ids=['one','two'])
    def commands(self,first=0):
        return {m:[sys.executable,'-B','-c',"import time,sys; time.sleep(.1); sys.exit("+str(first if m=='one'else 0)+")"]for m in ('one','two')}
    def receipt(self,g):return dict(status='technical-complete',result_review='pending',model=g['model'])
    def test_plan(self):
        p=self.plan;self.assertEqual((p['runs'],p['run_epochs'],p['max_optimizer_steps'],p['waves']),(228,2640,7438600,73))
        self.assertEqual([len(g['waves'])for g in p['groups']],[15,15,16,15,6,6]);self.assertEqual(p['order'],list(q.MODELS))
        self.assertFalse(set(p['task_ids'])&set(self.c['extension']['catchup_ids']))
        self.assertEqual([g['runs']for g in p['groups']],[45]*4+[24]*2)
    def test_decision_not_candidate_q(self):
        d=copy.deepcopy(self.decisions);d['ModernTCN-ECL-MS']['concurrency']=4
        with self.assertRaises(ValueError):q.plan(self.c,d)
    def test_children_exact(self):
        for m in q.MODELS:
            with self.subTest(model=m):self.assertEqual(q.child_reasons(self.c,m,self.child(m)),[])
    def test_cross_scope_reject(self):
        for m in q.MODELS:
            with self.subTest(model=m):
                a=self.child(m);a['authorized_task_ids']=self.c['extension']['catchup_ids'];self.assertTrue(q.child_reasons(self.c,m,a))
        a=self.child('N');a['extension_batch']=self.c['extension']['id'];self.assertTrue(q.child_reasons(self.c,'N',a))
    def test_template_reject(self):
        with self.assertRaises(PermissionError):q.authorization(self.c,{'queue_id':q.QUEUE,'reviewed':False})
        a=self.child('S');a['execution_permitted']=False;self.assertTrue(q.child_reasons(self.c,'S',a))
    def test_worker_output(self):
        m='N';a=self.child(m);r=a['authorized_task_ids'][0];out=str(q.result_path(self.c,task_by_id(self.c,r)))
        s=dict(task=r,approval=a,output=out,artifact_root=out,resume=False);q.worker_scope(self.c,s)
        s['output']=str(self.root)
        with self.assertRaises(PermissionError):q.worker_scope(self.c,s)
    def test_true_exit_then_handoff(self):
        order=[]
        def before(g):
            if g['model']=='two':self.assertTrue((self.root/'one-handoff.json').exists())
            order.append('start-'+g['model'])
        def check(g):order.append('check-'+g['model']);return self.receipt(g)
        entry.sequence(self.small(),self.root,self.commands(),before,check)
        self.assertEqual(order,['start-one','check-one','start-two','check-two']);self.assertTrue((self.root/'complete.json').exists())
    def test_nonzero_stops_successor(self):
        with self.assertRaises(RuntimeError):entry.sequence(self.small(),self.root,self.commands(3),lambda g:None,self.receipt)
        self.assertFalse((self.root/'two.log').exists());self.assertFalse((self.root/'complete.json').exists());self.assertTrue((self.root/'failure.json').exists())
    def test_invalid_handoff_stops(self):
        for kind in ('missing','identity','budget'):
            with self.subTest(kind=kind):
                d=self.root/kind;d.mkdir()
                def fail(g):raise ValueError(kind)
                with self.assertRaises(ValueError):entry.sequence(self.small(),d,self.commands(),lambda g:None,fail)
                self.assertFalse((d/'two.log').exists());self.assertFalse((d/'complete.json').exists())
    def test_stop_between_groups(self):
        def check(g):entry.stop_marker(self.root);return self.receipt(g)
        with self.assertRaises(InterruptedError):entry.sequence(self.small(),self.root,self.commands(),lambda g:None,check)
        self.assertFalse((self.root/'two.log').exists());self.assertTrue((self.root/'STOP').exists())
    def test_stop_before_spawn(self):
        def before(g):entry.stop_marker(self.root)
        with self.assertRaises(InterruptedError):entry.sequence(self.small(),self.root,self.commands(),before,self.receipt)
        self.assertFalse((self.root/'one.log').exists())
    def test_stop_inside_child(self):
        cmds=self.commands();cmds['one']=[sys.executable,'-B','-c',"from pathlib import Path;import time;Path("+repr(str(self.root/'STOP'))+").touch();time.sleep(5)"]
        with self.assertRaises(RuntimeError):entry.sequence(self.small(),self.root,cmds,lambda g:None,self.receipt)
        self.assertFalse((self.root/'two.log').exists());self.assertFalse(q.read(self.root/'failure.json')['child_still_live'])
    def test_lock_no_self_deadlock(self):
        with entry.lock(self.root/'queue.lock'):
            with entry.lock(self.root/'gpu.lock'):pass
            with self.assertRaises(BlockingIOError):
                with entry.lock(self.root/'queue.lock'):pass
    def test_repeat_start_rejected(self):
        with patch.object(q,'control_root',return_value=self.root):
            with self.assertRaises(FileExistsError):q.reject_existing(self.c,self.plan)
    def test_no_early_next_group(self):
        p=self.small();cmds=self.commands();cmds['one']=[sys.executable,'-B','-c',"from pathlib import Path;import time;time.sleep(.2);Path("+repr(str(self.root/'really-exited'))+").touch()"]
        def before(g):
            if g['model']=='two':self.assertTrue((self.root/'really-exited').exists())
        entry.sequence(p,self.root,cmds,before,self.receipt)
    def test_admission_parent_immutable(self):
        self.assertEqual(q.sha(q.PARENT),q.PARENT_SHA)
        with self.assertRaises(ValueError):q.validate_resources(self.c,{'purpose':q.RESOURCE_PURPOSE,'reviewed':False})
    def test_handoff_missing_file_reject(self):
        g=copy.deepcopy(self.plan['groups'][0]);g['control']=str(self.root)
        with self.assertRaises(FileNotFoundError):q.technical_handoff(self.c,g,self.child('DLinear'))
    def test_training_paths_and_profiles(self):
        self.assertEqual(len(self.c['tasks']),552)
        for g in self.plan['groups']:
            for r,path in g['result_paths'].items():
                t=task_by_id(self.c,r);self.assertEqual(g['profile_shas'][r],digest(profile(self.c,t)))
                self.assertIn('/supplements/' if t['dataset']in q.MODELS[:0]+('NP','BE','FR','DE') else '/formal-',path)
        self.assertEqual(sum(profile(self.c,t)['training']['epochs']for t in self.c['tasks']),6240)
    def test_technical_gate_actual_json(self):
        m='N';a=self.child(m);r=a['authorized_task_ids'][0];t=task_by_id(self.c,r);p=profile(self.c,t);ar=q.step_arithmetic(self.c,t);d=self.root/r;d.mkdir();a.update(commit=q.BASE,code={});meta={'fixed':'fixture'};a['data_bindings'][t['dataset']][r]=digest(meta)
        def put(p,x):p.write_text(json.dumps(x))
        def metric(elements):return dict(mse=.5,mae=.25,sse=.5*elements,sae=.25*elements,elements=elements)
        identity=dict(run_id=r,profile_sha=digest(p),data_sha=digest(meta),commit=q.BASE,protocol_sha=digest(self.c),source={})
        put(d/'manifest.json',dict(task=t,profile=p,identity=identity,metadata=meta));res=dict(id=r,purpose='ch3_formal',protocol_sha=digest(self.c),input_variant=t['input_variant'],seed=2024,metric_space='train-standardized',best_epoch=1,**metric(ar['test_windows_arithmetic_only']*p['pred_len']));put(d/'result.json',res)
        hist=[dict(epoch=i,steps=i*ar['train_batches'],best_epoch=1,validation=metric(ar['validation_windows']*p['pred_len']))for i in range(1,11)];(d/'history.jsonl').write_text('\n'.join(json.dumps(x)for x in hist))
        import math
        n=10*ar['train_batches'];f=10*(ar['train_batches']+math.ceil(ar['validation_windows']/p['training']['eval_batch']))+math.ceil(ar['test_windows_arithmetic_only']/p['training']['eval_batch'])
        put(d/'budget.json',dict(counts=dict(adam=n,forward=f,backward=n),by_pid={'99999999':dict(adam=n,forward=f,backward=n)}));put(d/'runtime.json',dict(pid=99999999,error=None));put(d/'config.json',dict(approval=a,resume=False,task=r,output=str(d)))
        put(self.root/'complete.json',dict(task_ids=[r],protocol_sha=digest(self.c)));put(self.root/'controller.json',dict(pid=99999999,start_ticks='1',model='N'))
        w=self.root/'wave-000';w.mkdir();put(w/'process.json',dict(returncodes=[0],resource_admission=True,exit_transitions_resolved=True,failure=None,process_peaks={'99999999':1}));(w/'memory.jsonl').write_text(json.dumps({'owned_pid_metadata':{'99999999':{'start_ticks':'1'}}})+'\n')
        g=dict(model=m,control=str(self.root),task_ids=[r],waves=[[r]])
        with patch.object(q,'result_path',return_value=d):
            got=q.technical_handoff(self.c,g,a);self.assertEqual(got['status'],'technical-complete');self.assertEqual(got['totals']['adam'],n)
            for field,value in [('best_epoch',2),('mse',float('nan')),('protocol_sha','wrong')]:
                with self.subTest(field=field):
                    bad=dict(res);bad[field]=value;put(d/'result.json',bad)
                    with self.assertRaises(ValueError):q.technical_handoff(self.c,g,a)
            put(d/'result.json',res)
            (self.root/'failure.json').write_text('{}')
            with self.assertRaises(ValueError):q.technical_handoff(self.c,g,a)
            (self.root/'failure.json').unlink()
            hist[-1]['steps']+=1;(d/'history.jsonl').write_text('\n'.join(json.dumps(x)for x in hist))
            with self.assertRaises(ValueError):q.technical_handoff(self.c,g,a)
    def test_authorized_positive_dispatch_chain(self):
        # Explicit fixture: exercise total/child scope checks, never real start.
        from unittest.mock import patch
        from contextlib import ExitStack
        a=dict(queue_id=q.QUEUE,purpose='m6_remaining_queue',reviewed=True,execution_permitted=True,commit=q.BASE,code={},protocol_sha=digest(self.c),environment={},hardware={},resource_report={'tag':'resources','sha256':'r'},plan={'tag':'plan','sha256':'p'},catchup_completion_audit={'tag':'audit'},task_ids=self.plan['task_ids'],run_budget={'runs':228,'run_epochs':2640},max_optimizer_steps=7438600,children={m:{'tag':m}for m in q.MODELS})
        children={m:self.child(m)for m in q.MODELS}
        for x in children.values():x.update(commit=q.BASE,code={},protocol_sha=digest(self.c),environment={},hardware={},plan_sha256='p',probe_report_sha='r')
        def bound(ref):return {'decisions':self.decisions}if ref['tag']=='resources'else self.plan if ref['tag']=='plan'else children[ref['tag']]
        with ExitStack()as stack:
            for name,value in [('code_binding',{}),('environment_binding',{}),('hardware_binding',{}),('preflight',[])]:stack.enter_context(patch('ch3_runner.'+name,return_value=value))
            stack.enter_context(patch('ch3_runner.git',side_effect=lambda *x:''if x[0]=='status'else q.BASE))
            stack.enter_context(patch.object(q,'bound',side_effect=bound));stack.enter_context(patch.object(q,'validate_resources'));stack.enter_context(patch.object(q,'catchup_audit'));stack.enter_context(patch.object(q,'reject_existing'));stack.enter_context(patch('utils.ch3_extension.extension_reasons',return_value=[]))
            p,cs=q.authorization(self.c,a);self.assertEqual(p,self.plan);self.assertEqual(set(cs),set(q.MODELS))
            for m in q.MODELS:
                with self.subTest(model=m):
                    saved=children[m]['authorized_task_ids'];children[m]['authorized_task_ids']=[]
                    with self.assertRaises(PermissionError):q.authorization(self.c,a)
                    children[m]['authorized_task_ids']=saved
    def test_source_carry_forward_positive_and_negative(self):
        from ch3_runner import code_binding
        parent=q.read(q.PARENT);code=code_binding()
        proof=dict(base_commit=q.BASE,protocol_sha=digest(self.c),before_code=parent['code'],after_code=code,profile_shas={t['id']:digest(profile(self.c,t))for t in self.c['tasks']})
        q.source_proof(self.c,proof,code)
        for k in ('protocol_sha','after_code','profile_shas'):
            with self.subTest(field=k):
                bad=dict(proof);bad[k]={}if k!='protocol_sha'else 'wrong'
                with self.assertRaises(ValueError):q.source_proof(self.c,bad,code)
    def test_catchup_raw_not_accepted(self):
        raw=q.PACKAGE/'catchup41-completed-review-v1/completion-audit.json'
        with self.assertRaises(ValueError):q.catchup_audit(self.c,q.ref(raw))
    def test_cli_fixture_lifecycle_and_accounting(self):
        import types
        from contextlib import ExitStack
        fake=types.ModuleType('m5_formal_entry');fake.gpu_sample=lambda x:{};fake.resource_assessment=lambda x,y:{'admission':True}
        for exitcode in (0,3):
            with self.subTest(exitcode=exitcode),ExitStack()as st:
                root=self.root/str(exitcode)/'queue';root.parent.mkdir();app=self.root/('app'+str(exitcode)+'.json')
                p=dict(groups=[dict(model=m,result_paths={},control=str(root.parent/m))for m in ('DLinear','N')],task_ids=['fixture-a','fixture-b'])
                a=dict(commit=q.BASE,protocol_sha=digest(self.c),plan={'sha256':'fixture'},resource_report={'path':'fixture-report'},children={m:{'path':'fixture-child'}for m in ('DLinear','N')});app.write_text(json.dumps(a))
                st.enter_context(patch.dict(sys.modules,{'m5_formal_entry':fake}));st.enter_context(patch.object(q,'control_root',return_value=root));st.enter_context(patch.object(q,'authorization',return_value=(p,{'DLinear':{},'N':{}})))
                st.enter_context(patch('ch3_runner.GPULock',side_effect=lambda c:entry.lock(root.parent/'gpu.lock')))
                st.enter_context(patch.object(q,'group_control',side_effect=lambda c,m:root.parent/m));st.enter_context(patch.object(q,'result_path',return_value=root.parent/'never-created'))
                st.enter_context(patch.object(q,'technical_handoff',side_effect=lambda c,g,a:self.receipt(g)))
                st.enter_context(patch.object(entry,'command',side_effect=lambda g,a,r:[sys.executable,'-B','-c','import time,sys;time.sleep(.15);sys.exit('+str(exitcode)+')']))
                if exitcode:
                    with self.assertRaises(RuntimeError):entry.cli(['start','--approval',str(app)])
                    self.assertFalse((root/'N.log').exists())
                else:self.assertEqual(entry.cli(['start','--approval',str(app)]),0);self.assertTrue((root/'complete.json').exists())
                self.assertFalse(q.read(root/'execution-accounting.json')['budget_refund'])
