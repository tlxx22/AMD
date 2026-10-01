"""Synthetic saved-artifact admission fixtures; never execute a model/worker."""
import contextlib,copy,json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from utils import ch3_m_tasks as s,ch3_m_execution as e,ch3_m_handoff as h,ch3_m_auto as a,ch3_m_launch as l
from utils.ch3_contract import read_profiles,validate_manifest,digest,profile
from ch3_runner import code_binding

def put(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');return s.ref(path)

class MAutoTests(unittest.TestCase):
 def setUp(self):
  self.c=validate_manifest(read_profiles());self.tmp=tempfile.TemporaryDirectory(dir=s.PACKAGE/'fixtures');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.package=self.root/'package';self.package.mkdir();self.supervisor=self.package/'supervisor';self.supervisor.mkdir();self.children=[]
  data=s.bound(self.c['m_experiment']['data_binding_artifact']);self.context=dict(commit='a'*40,code=code_binding(),protocol_sha=digest(self.c),environment={'synthetic_fixture':True},hardware={'synthetic_fixture':True},plan=s.ref(s.PACKAGE/'plan.json'),data_binding_artifact=self.c['m_experiment']['data_binding_artifact'],data_bindings=data['data_bindings'],source_states=data['source_states'])
  self.stack=contextlib.ExitStack();self.addCleanup(self.stack.close)
  for obj,name,val in [(s,'PACKAGE',self.package),(s,'RESULT',self.root/'M-results'),(e,'PROBE_ROOT',self.package/'probe'),(e,'CONTROL',self.root/'M-results/queues/m-baselines-v1'),(h,'ROOT',self.supervisor)]:self.stack.enter_context(patch.object(obj,name,val))
  self.stack.enter_context(patch.object(a,'binding',return_value=self.context));self.stack.enter_context(patch('ch3_runner.code_binding',return_value=self.context['code']));self.stack.enter_context(patch('ch3_runner.environment_binding',return_value=self.context['environment']));self.stack.enter_context(patch('ch3_runner.hardware_binding',return_value=self.context['hardware']));self.stack.enter_context(patch('ch3_runner.git',side_effect=lambda *args:'m6/m-baselines-v1'if args[0]=='branch'else''if args[0]=='status'else self.context['commit']))
  self.boundary=dict(kind='old_MS_queue_technical_complete',result_review='pending',synthetic_fixture=True,task_ids=['fixture'])
  self.boundary_ref=put(self.package/'old-queue-technical-boundary.json',self.boundary);self.stack.enter_context(patch.object(e,'old_boundary',return_value=self.boundary));self.stack.enter_context(patch.object(e,'gpu_environment_reasons',return_value=[]))
  self.stack.enter_context(patch('m6_m_entry.GPULock',return_value=contextlib.nullcontext()));self.stack.enter_context(patch('m5_formal_entry.gpu_sample',return_value={'synthetic_fixture':True}));self.stack.enter_context(patch('m5_formal_entry.resource_assessment',return_value=dict(admission=True)))
  original=subprocess.run
  self.stack.enter_context(patch('subprocess.run',side_effect=lambda args,**kw:subprocess.CompletedProcess(args,1)if args[0]=='tmux'else original(args,**kw)))
  self.stack.enter_context(patch.object(e,'compare',side_effect=self.compare));self.stack.enter_context(patch('ch3_runner._compare_full_numeric_files',side_effect=self.full_compare));self.stack.enter_context(patch('m6_remaining_entry.same',return_value=False))
 def compare(self,c,t,x,y):
  for k in ('id','profile_sha','initial','initial_rng','batch_ids','steps','final_rng'):
   if x[k]!=y[k]:raise ValueError('fixture identity mismatch')
  if not x['finite']or not y['finite']:raise ValueError('fixture finite failure')
  return dict(passed=x['numeric_passed']and y['numeric_passed'],synthetic_fixture=True)
 def full_compare(self,rule,x,y):
  for row in (x,y):
   if s.sha(row['schema_file'])!=row['schema_sha']or s.sha(row['data_file'])!=row['data_sha']:raise ValueError('fixture state payload binding')
  return dict(passed=True,synthetic_fixture=True)
 def probe_permit(self):return a.create_permit(self.c,True,self.boundary_ref,self.supervisor)
 def make_report(self,permit,*,q=4,kind='resource'):
  root=e.PROBE_ROOT;root.mkdir();put(root/'approval.json',permit['value']);put(root/'controller.json',dict(pid=99999999,start_ticks='1',approval=dict(path=permit['path'],sha256=permit['sha256']),synthetic_fixture=True));evidence={};artifacts={};decisions={};reserved=dict(adam=0,forward=0,backward=0);actual=dict(reserved)
  def wave(g,phase,n,ids,fail=False):
   v=dict(returncodes=[1 if fail else 0 for r in ids],resource_admission=not fail,failure='fixture failure'if fail else None,failure_kind=kind if fail else None,elapsed=3 if phase=='serial'else 1,synthetic_fixture=True)
   evidence[g['id']+'/'+phase+'/'+str(n)]=put(root/g['id']/phase/('wave-'+str(n))/'process.json',v)
   for run in ids:
    out=root/g['id']/phase/run;cfg=dict(task=run,m_phase=phase,protocol_sha=digest(self.c),approval=permit['value'],m_scope=s.PROBE,output=str(out),prefix_files={},limits=dict(adam=6,forward=8,backward=6,seconds=1800),synthetic_fixture=True);counts=dict(adam=0,forward=0,backward=0)if fail else dict(adam=6,forward=8,backward=6)
    artifacts[str(out/'config.json')]=put(out/'config.json',cfg);artifacts[str(out/'budget.json')]=put(out/'budget.json',dict(counts=counts,synthetic_fixture=True))
    for k in reserved:reserved[k]+=dict(adam=6,forward=8,backward=6)[k];actual[k]+=counts[k]
    schema=out/'synthetic-schema.json';put(schema,dict(synthetic_fixture=True));payload=out/'synthetic-state.bin';payload.write_bytes(b'synthetic finite state');artifacts[str(schema)]=s.ref(schema);artifacts[str(payload)]=s.ref(payload)
    t=next(t for t in self.c['tasks']if t['id']==run);points=[dict(step=i,schema_file=str(schema),schema_sha=s.sha(schema),data_file=str(payload),data_sha=s.sha(payload),bytes=payload.stat().st_size)for i in range(1,7)]
    tr=dict(id=run,profile_sha=digest(profile(self.c,t)),finite=True,initial='fixture',initial_rng='fixture',batch_ids=list(range(6)),steps=6,final_rng='fixture',M_full_state_trace=points,numeric_passed=True,synthetic_fixture=True)
    artifacts[str(out/'trajectory.json')]=put(out/'trajectory.json',tr);artifacts[str(out/'runtime.json')]=put(out/'runtime.json',dict(synthetic_fixture=True));log=out/'audit.jsonl';log.write_text('{"synthetic_fixture":true}\n');artifacts[str(log)]=s.ref(log)
   return v
  for index,g in enumerate(self.c['groups']):
   selected=q if index==0 else 4;serial=[wave(g,'serial',n,[r])for n,r in enumerate(g['task_ids'])];q4=[wave(g,'q4',0,g['task_ids'],selected!=4)];attempts=[dict(phase='q4',waves=q4,resource_failed=selected!=4)];parallel=q4
   if selected!=4:
    parallel=[wave(g,'q2',n,ids,selected==1)for n,ids in enumerate(e.wave_ids(g['task_ids'],2)[:1 if selected==1 else 2])];attempts.append(dict(phase='q2',waves=parallel,resource_failed=selected==1))
   comparisons=[]if selected==1 else[dict(passed=True,synthetic_fixture=True)for _ in g['task_ids']]
   decisions[g['id']]=dict(status='Passed',concurrency=selected,serial=serial,parallel=parallel,attempts=attempts,numerical_comparisons=comparisons,policy_sha=digest(self.c['m_experiment']['numeric_policies'][g['id']]),q1_only_if_serial_valid=True)
  budget=dict(caps=s.CAPS,reserved=reserved,actual=actual,refund=False);put(root/'budget.json',budget)
  report=dict(purpose='M_resource_admission',m_scope=s.PROBE,reviewed=False,admission_granted=False,protocol_sha=digest(self.c),commit=self.context['commit'],code=self.context['code'],environment=self.context['environment'],hardware=self.context['hardware'],approval=s.ref(root/'approval.json'),decisions=decisions,budget=budget,evidence=evidence,artifacts=artifacts,execution_complete=True,synthetic_fixture=True)
  put(root/'complete.json',report);return report
 def audit(self,permit):return a.audit_probe(self.c,self.supervisor,dict(path=permit['path'],sha256=permit['sha256']))
 def run_chain(self,*,q=4,error=None,stop_after_audit=False,stop_after_permit=False):
  create=a.create_permit;audit=a.audit_probe
  def child(c,root,probe,permit):
   self.children.append(probe)
   if probe:self.make_report(permit,q=q)
   elif error:raise RuntimeError(error)
   return dict(technical_complete=True,result_review='pending',reviewed=False,permit=dict(path=permit['path'],sha256=permit['sha256']),synthetic_fixture=True)
  def audited(*args):
   r=audit(*args)
   if stop_after_audit:(self.supervisor/'STOP').write_text('synthetic STOP')
   return r
  def created(c,probe,*args):
   r=create(c,probe,*args)
   if not probe and stop_after_permit:(self.supervisor/'STOP').write_text('synthetic STOP')
   return r
  with patch.object(h,'old_queue_state',return_value='complete'),patch.object(h,'run_owned',side_effect=child),patch.object(a,'audit_probe',side_effect=audited),patch.object(a,'create_permit',side_effect=created):h.run(self.c,self.supervisor)
 def test_complete_auto_chain_without_manual_permits(self):
  self.run_chain();self.assertEqual(self.children,[True,False]);self.assertFalse((self.package/'probe-review.json').exists());self.assertFalse((self.package/'formal-review.json').exists());done=json.loads((self.supervisor/'complete.json').read_text());self.assertEqual(done['result_review'],'pending');self.assertFalse(done['manual_review']);self.assertEqual([json.loads(x)['state']for x in(self.supervisor/'states.jsonl').read_text().splitlines()],list(h.STATES))
  formal=s.bound(s.ref(a.permit_path(False)));self.assertFalse(formal['reviewed']);self.assertTrue(formal['technical_admission']);self.assertEqual(formal['run_budget'],dict(runs=84,run_epochs=840));self.assertEqual(formal['max_optimizer_steps'],294790);self.assertEqual(len(formal['probe_decisions']),21)
 def test_resource_q4_failed_q2_passed_binds_q2(self):
  self.run_chain(q=2);formal=s.bound(s.ref(a.permit_path(False)));self.assertEqual(formal['probe_decisions'][self.c['groups'][0]['id']]['concurrency'],2)
 def test_resource_q4_q2_failed_valid_serial_binds_q1(self):
  self.run_chain(q=1);formal=s.bound(s.ref(a.permit_path(False)));self.assertEqual(formal['probe_decisions'][self.c['groups'][0]['id']]['concurrency'],1)
 def corrupt_trajectory(self,field,value):
  run=self.c['tasks'][0];p=e.PROBE_ROOT/run['group']/'serial'/run['id']/'trajectory.json';d=json.loads(p.read_text());d[field]=value;put(p,d)
  report=json.loads((e.PROBE_ROOT/'complete.json').read_text());report['artifacts'][str(p)]=s.ref(p);put(e.PROBE_ROOT/'complete.json',report)
 def test_numeric_failure_prevents_formal_permit(self):
  permit=self.probe_permit();self.make_report(permit);self.corrupt_trajectory('numeric_passed',False)
  with self.assertRaises(ValueError):self.audit(permit)
  self.assertFalse(a.permit_path(False).exists());self.assertEqual(self.children,[])
 def test_finite_failure_stops(self):
  permit=self.probe_permit();self.make_report(permit);self.corrupt_trajectory('finite',False)
  with self.assertRaisesRegex(ValueError,'finite'):self.audit(permit)
  self.assertFalse(a.permit_path(False).exists())
 def test_nonresource_failure_cannot_fallback_q2(self):
  permit=self.probe_permit();self.make_report(permit,q=2,kind='numeric')
  with self.assertRaisesRegex(ValueError,'nonresource'):self.audit(permit)
  self.assertFalse(a.permit_path(False).exists())
 def test_artifact_SHA_tamper_rejected(self):
  permit=self.probe_permit();report=self.make_report(permit);payload=next(Path(p)for p in report['artifacts']if p.endswith('.bin'));payload.write_bytes(b'tampered')
  with self.assertRaises(ValueError):self.audit(permit)
 def test_budget_over_cap_rejected(self):
  permit=self.probe_permit();report=self.make_report(permit);report['budget']['reserved']['adam']=s.CAPS['adam']+1;put(e.PROBE_ROOT/'budget.json',report['budget']);put(e.PROBE_ROOT/'complete.json',report)
  with self.assertRaises(ValueError):self.audit(permit)
 def test_machine_commit_protocol_source_data_environment_binding(self):
  permit=self.probe_permit()['value']
  for key in ('commit','protocol_sha','code','environment','hardware','data_bindings','source_states','plan'):
   with self.subTest(field=key):
    wrong=copy.deepcopy(permit);wrong[key]='wrong'
    with self.assertRaises(ValueError):a.validate_permit(self.c,wrong,True)
  wrong=copy.deepcopy(permit);wrong['policy']['sources']['plan']['sha256']='0'*64
  with self.assertRaises(ValueError):a.validate_permit(self.c,wrong,True)
 def test_duplicate_formal_generation_is_exclusive(self):
  permit=self.probe_permit();self.make_report(permit);admission=self.audit(permit);a.create_permit(self.c,False,self.boundary_ref,self.supervisor,admission);before=a.permit_path(False).read_bytes()
  with self.assertRaises(FileExistsError):a.create_permit(self.c,False,self.boundary_ref,self.supervisor,admission)
  self.assertEqual(before,a.permit_path(False).read_bytes())
 def test_STOP_after_audit_before_formal_permit(self):
  with self.assertRaises(InterruptedError):self.run_chain(stop_after_audit=True)
  self.assertEqual(self.children,[True]);self.assertFalse(a.permit_path(False).exists())
 def test_STOP_after_permit_before_formal_dispatch(self):
  with self.assertRaises(InterruptedError):self.run_chain(stop_after_permit=True)
  self.assertEqual(self.children,[True]);self.assertTrue(a.permit_path(False).exists());self.assertFalse((self.supervisor/'complete.json').exists())
 def test_formal_child_failure_stops_suffix(self):
  with self.assertRaisesRegex(RuntimeError,'formal failed'):self.run_chain(error='formal failed')
  self.assertEqual(self.children,[True,False]);self.assertFalse((self.supervisor/'complete.json').exists());self.assertTrue((self.supervisor/'failure.json').exists())
 def test_no_runtime_policy_plan_profile_changes(self):
  before=digest(e.plan(self.c));config=digest(self.c);numeric=digest(self.c['m_experiment']['numeric_policies']);self.run_chain();self.assertEqual(before,digest(e.plan(self.c)));self.assertEqual(config,digest(self.c));self.assertEqual(numeric,digest(self.c['m_experiment']['numeric_policies']))
 def test_machine_and_manual_format_not_interchangeable(self):
  value=self.probe_permit()['value'];value['reviewed']=True
  with self.assertRaises(ValueError):a.validate_permit(self.c,value,True)
  value['reviewed']=False;value['technical_admission']=True
  with self.assertRaises(ValueError):a.validate_permit(self.c,value,True)
 def test_machine_exact_scope_budget_flags_and_unknown_format(self):
  value=self.probe_permit()['value']
  for key,bad in [('purpose','ch3_formal'),('m_scope',s.ID),('execution_permitted',False),('budget_authorized',False),('task','MS'),('metric_scope','target_only'),('authorized_task_ids',[]),('profile_shas',{}),('caps',{}),('already_charged',dict(adam=1,forward=0,backward=0)),('run_budget',dict(runs=85,run_epochs=840)),('max_optimizer_steps',294791),('old_boundary',dict(path='/wrong',sha256='0'*64))]:
   with self.subTest(field=key):
    wrong=copy.deepcopy(value);wrong[key]=bad
    with self.assertRaises(ValueError):a.validate_permit(self.c,wrong,True)
  wrong=copy.deepcopy(value);wrong.update(review_mode='other-machine-mode',reviewed=True)
  self.assertIn('unknown M review mode; exact manual or machine authorization required',e.authorization_reasons(self.c,wrong,True))
 def test_worker_counter_negative_and_noninteger_rejected(self):
  permit=self.probe_permit();report=self.make_report(permit);run=self.c['tasks'][0];p=e.PROBE_ROOT/run['group']/'serial'/run['id']/'budget.json'
  for bad in (-1,1.5,True,7):
   with self.subTest(count=bad):
    changed=dict(counts=dict(adam=bad,forward=8,backward=6),synthetic_fixture=True);report['artifacts'][str(p)]=put(p,changed);put(e.PROBE_ROOT/'complete.json',report)
    with self.assertRaisesRegex(ValueError,'counter'):self.audit(permit)
 def test_metadata_allowlist_includes_actual_auto_sources(self):
  permit=self.probe_permit();self.make_report(permit);admission=self.audit(permit);formal=a.create_permit(self.c,False,self.boundary_ref,self.supervisor,admission);refs=e.metadata_files(self.c,formal['value'])
  self.assertIn(str(e.PROBE_ROOT/'complete.json'),refs);self.assertIn(str(e.PROBE_ROOT/'approval.json'),refs);self.assertIn(permit['path'],refs)
  for row in a.SOURCES.values():self.assertEqual(refs[row['path']],row['sha256'])
