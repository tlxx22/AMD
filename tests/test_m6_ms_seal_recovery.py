"""Synthetic MS-seal recovery and M128 revision; no business-model computation."""
import ast,copy,json,os,subprocess,tempfile,unittest
from pathlib import Path
from contextlib import ExitStack,contextmanager
from unittest.mock import patch
from utils import ch3_type1_chain as q,ch3_type1_tasks as s,ch3_ms_seal_recovery as r,ch3_round2_amendment as a,ch3_type1_upstream as u
from utils.ch3_contract import ROOT,digest,profile,step_arithmetic
from utils.ch3_native_recovery_records import bound,ref,exclusive,sha

def changes(x,y,path=''):
 if isinstance(x,dict)and isinstance(y,dict):
  return sum((changes(x.get(k),y.get(k),path+'.'+k if path else k)for k in sorted(set(x)|set(y))),[])
 return [] if x==y else [path]

class Profiles(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.cs=q.configs()
 def test_counts_84_112_28_35_168(self):self.assertEqual([len(c['tasks'])for c in self.cs.values()],[84,112,28,35,168])
 def test_remaining427_and_third231(self):self.assertEqual(sum(len(c['tasks'])for c in self.cs.values()),427);self.assertEqual(sum(len(self.cs[k]['tasks'])for k in ('URBAN_SUBSET','EPF_ALL','M_ALL')),231)
 def test_order_and_original_three_M_domains(self):
  self.assertEqual(list(dict.fromkeys(t['model']for t in self.cs['M_BASE']['tasks'])),list(s.MODELS));self.assertEqual(list(dict.fromkeys(t['dataset']for t in self.cs['M_BASE']['tasks'])),['ETTh1','Weather','Exchange'])
 def test_57_changed_27_already128(self):
  counts=[r.m_parent()['resolved_profiles'][r.old_task(t)['id']]['training']['batch']!=128 for t in self.cs['M_BASE']['tasks']];self.assertEqual((sum(counts),len(counts)-sum(counts)),(57,27))
 def test_batch_and_eval128_all84(self):
  for t in self.cs['M_BASE']['tasks']:self.assertEqual([profile(self.cs['M_BASE'],t)['training'][k]for k in ('batch','eval_batch')],[128,128])
 def test_M_only_allowed_effective_diffs(self):
  for t in self.cs['M_BASE']['tasks']:
   before=r.m_parent()['resolved_profiles'][r.old_task(t)['id']];after=profile(self.cs['M_BASE'],t);self.assertLessEqual(set(changes(before,after)),{'training.batch','training.eval_batch','training.scheduler.steps_per_epoch'})
 def test_real_scheduler_step_arithmetic(self):
  from utils.ch3_onecycle import configuration
  for stage in ('M_BASE','M_AMEND'):
   for t in self.cs[stage]['tasks']:
    p=profile(self.cs[stage],t);self.assertEqual(p['training']['scheduler'],configuration(p['training']['epochs'],step_arithmetic(self.cs[stage],t)['train_batches']))
 def test_M_all_channels_no_target_slice(self):
  for stage in ('M_BASE','M_AMEND','M_ALL'):
   for t in self.cs[stage]['tasks']:
    p=profile(self.cs[stage],t);self.assertEqual(p['metric_scope'],'all_channels');self.assertEqual(p['supervised_channels'],list(range(p['C'])));self.assertEqual(p['output_order'],p['features'])
 def test_amend_ETT84_zero_science_change(self):
  old=json.loads((ROOT/'configs/ch3_round2_m_amend1.json').read_text());count=0
  for prior,t in zip(old['tasks'],self.cs['M_AMEND']['tasks']):
   if t['dataset']!='Weather':self.assertEqual(old['resolved_profiles'][prior['id']],profile(self.cs['M_AMEND'],t));count+=1
  self.assertEqual(count,84)
 def test_amend_Weather_only_batch_and_derived_steps(self):
  old=json.loads((ROOT/'configs/ch3_round2_m_amend1.json').read_text())
  for prior,t in zip(old['tasks'],self.cs['M_AMEND']['tasks']):
   if t['dataset']=='Weather':
    p=profile(self.cs['M_AMEND'],t);self.assertLessEqual(set(changes(old['resolved_profiles'][prior['id']],p)),{'training.batch','training.eval_batch','training.scheduler.steps_per_epoch'});self.assertEqual((p['training']['epochs'],p['training']['patience'],p['training']['batch']),(20,None,128))
 def test_Weather_inherits_new_base_then_only_length(self):
  for t in self.cs['M_AMEND']['tasks']:
   if t['dataset']=='Weather':self.assertEqual(changes(a.parent()['resolved_profiles'][a.source_task(t)['id']],profile(self.cs['M_AMEND'],t)),['training.epochs','training.scheduler.epochs'])
 def test_third231_zero_science_diff(self):
  for stage in ('URBAN_SUBSET','EPF_ALL','M_ALL'):
   old=json.loads((ROOT/'configs'/('ch3_type1_'+stage.lower()+'_v3.json')).read_text())
   self.assertEqual([old['resolved_profiles'][t['id']]for t in old['tasks']],[profile(self.cs[stage],t)for t in self.cs[stage]['tasks']])
 def test_numeric_registry_unchanged_every_stage(self):
  for stage,c in self.cs.items():
   old=r.m_parent()if stage=='M_BASE'else json.loads((ROOT/'configs'/('ch3_round2_m_amend1.json'if stage=='M_AMEND'else'ch3_type1_'+stage.lower()+'_v3.json')).read_text());self.assertEqual(c['baseline_unified']['numeric_policies'],old['baseline_unified']['numeric_policies'])
 def test_frozen_old_config_SHA(self):self.assertEqual(sha(r.M_PARENT['path']),r.M_PARENT['sha256'])
 def test_no_extra_ETTh1_runs(self):self.assertEqual(sum(t['dataset']=='ETTh1'for t in self.cs['M_BASE']['tasks']),28);self.assertFalse(any(t['dataset']=='ETTh1'for t in self.cs['M_AMEND']['tasks']))
 def test_no_exclusions_or_search(self):
  for c in self.cs.values():self.assertFalse(any(t['model']in ('J','N','S')or t['dataset']=='ECL'or t['seed']!=2024 for t in c['tasks']))
 def test_metadata_scaler_and_marks_unchanged(self):
  for c in self.cs.values():
   data=bound(c['baseline_unified']['data_ref']);self.assertFalse(data['test_observations_accessed'])
   for row in data['mapping']:
    t=next(t for t in c['tasks']if t['id']==row['task_id']);before=bound(row['source_ref'])['metadata'][t['dataset']][row['source_task_id']];self.assertEqual(before,data['metadata'][t['dataset']][t['id']])
 def test_106_groups_427_representatives(self):
  groups=[s.probe_groups(c)for c in self.cs.values()];self.assertEqual([len(g)for g in groups],[21,28,7,8,42]);self.assertEqual(sum(len(g['representatives'])for gs in groups for g in gs),427)
 def test_exact_derived_execution_counts(self):
  expected=[(84,840,108080,128877),(112,1400,325920,410445),(28,560,1028440,1143128),(35,350,399000,467845),(168,1960,354480,446642)]
  for c,want in zip(self.cs.values(),expected):
   total=s.formal_budget(c)['total'];self.assertEqual((total['runs'],total['run_epochs'],total['adam'],total['forward']),want);self.assertEqual(total['backward'],total['adam'])
  caps=[s.probe_budget(c)['caps']for c in self.cs.values()];self.assertEqual([sum(x[k]for x in caps)for k in ('adam','backward','forward')],[7674,7674,10256])
 def test_template_not_executable(self):
  x=q.start_template();self.assertFalse(any(x[k]for k in ('reviewed','execution_permitted','structure_frozen','m6_authorized','budget_authorized')));self.assertIsNone(x['closure_commit'])
 def test_old_permit_commit_namespace_cannot_authorize(self):
  with patch.object(q,'closure',return_value='synthetic-clean-head'):
   with self.assertRaises(PermissionError):q.validate_start(bound(ref(r.PACKAGE.parent/'baseline-type1-followup-v3/start-review.json')))
 def test_effective371_and_planned399(self):self.assertEqual((len(a.expected_cells()),len(set(a.expected_cells()))),(371,371))
 def test_outputs_and_permissions_fresh(self):self.assertFalse(s.RESULT.exists());self.assertFalse((s.PACKAGE/'start-review.json').exists())
 def test_pinned_history_not_current_globals(self):
  raw=subprocess.check_output(['git','show','1134611cd52e4cedb7418d8a9f8a7f6e76512617:utils/ch3_type1_tasks.py'],cwd=ROOT,text=True);tree=ast.parse(raw)
  values={n.targets[0].id:ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id in ('PROTOCOL','STAGES')};self.assertEqual(values['PROTOCOL'],'baseline-type1-followup-v3');self.assertEqual(values['STAGES'],('M_AMEND','URBAN_SUBSET','EPF_ALL','M_ALL'));self.assertNotEqual(values['PROTOCOL'],s.PROTOCOL)

class Seal(unittest.TestCase):
 def test_reproduces_original_duplicate_keyword(self):
  dynamic=dict(protocol_sha='same',commit='training',code={},environment={},hardware={},source_states={})
  with self.assertRaises(TypeError):dict(protocol_sha='same',**dynamic)
 def exercise(self,module,stage,bad=False):
  with tempfile.TemporaryDirectory()as tmp,ExitStack()as stack:
   root=Path(tmp);c={'baseline_unified':{'stage':stage},'tasks':[dict(id='x',model='AMD')]};record=exclusive(root/'group.json',dict(technical_complete=True,model='AMD',task_ids=['x']))
   binding=dict(protocol_sha='wrong'if bad else digest(c),commit='seal-version',code={'runner':'bound'},environment={'python':'fixed'},hardware={'GPU':'fixed'},source_states={'source':'bound'})
   d=stack.enter_context(patch.object(module,'dynamic',return_value=binding));stack.enter_context(patch.object(module,'stop_check'));stack.enter_context(patch.object(module.s,'MODELS',('AMD',)));stack.enter_context(patch.object(module.s,'context',return_value={'control':root,'stage':stage}))
   if bad:
    with self.assertRaises(ValueError):module.seal_boundary(c,{'AMD':record})
    self.assertFalse((root/'technical-boundary.json').exists())
   else:
    value=bound(module.seal_boundary(c,{'AMD':record}));self.assertEqual({k:value[k]for k in binding},binding)
   self.assertEqual(d.call_count,1)
 def test_unified_MS_seal(self):
  from utils import ch3_baseline_unified_chain as original
  self.exercise(original,'MS')
 def test_unified_M_seal(self):
  from utils import ch3_baseline_unified_chain as original
  self.exercise(original,'M')
 def test_all_actual_recovery_stage_seals(self):
  for stage in s.STAGES:self.exercise(q,stage)
 def test_wrong_digest_not_silently_overwritten(self):self.exercise(q,'M_BASE',True)

@contextmanager
def source_fixture():
 with tempfile.TemporaryDirectory()as tmp,ExitStack()as stack:
  root=Path(tmp);old=root/'original';follower=root/'baseline-type1-followup-v3';control=old/'queue/controller';control.mkdir(parents=True);fc=follower/'queue/controller';fc.mkdir(parents=True)
  c=bound(u.START_REF);c=bound(c['config_refs']['MS']);c=copy.deepcopy(c)
  for p in c['resolved_profiles'].values():p['training']['epochs']=1;p['training']['scheduler']['epochs']=1;p['training']['patience']=None
  config=exclusive(root/'MS-config.json',c);data=bound(c['baseline_unified']['data_ref']);data_ref=exclusive(root/'data.json',data);c['baseline_unified']['data_ref']=data_ref
  (root/'MS-config.json').write_text(json.dumps(c));config=ref(root/'MS-config.json')
  code=exclusive(root/'code.json',{'code':{}});env=exclusive(root/'env.json',{'environment':{},'hardware':{}})
  start=exclusive(root/'start.json',dict(scientific_protocol=u.OLD_PROTOCOL,scope=u.OLD_SCOPE,closure_commit=u.BASE,execution_permitted=True,config_refs={'MS':config}))
  owner=dict(pid=123456789,start_ticks='1');controller=exclusive(control/'controller.json',dict(scope=u.OLD_SCOPE,owner=owner,authorization=start))
  exclusive(control/'failure.json',dict(automatic_retry=False,error=r.ERROR,result_review='pending',scope=u.OLD_SCOPE));exclusive(control/'progress.json',dict(state='SEAL_MS_BOUNDARY',scope=u.OLD_SCOPE))
  exclusive(fc/'controller.json',dict(scope='m6-baseline-type1-followup-v3',owner=dict(pid=123456788,start_ticks='2')));exclusive(fc/'failure.json',dict(error="RuntimeError('upstream STOP/failure; successor cannot run')"));exclusive(fc/'progress.json',dict(state='WAIT_V3_COMPLETE_AND_RELEASED'))
  probe=exclusive(root/'probe-permit.json',{'start_authorization_ref':start});report=exclusive(old/'probe/MS/complete.json',dict(scope=u.OLD_SCOPE+'-MS-probe',approval=probe,execution_complete=True,commit=u.BASE,budget={'actual':{'adam':0,'backward':0,'forward':0}},evidence={}))
  summary=exclusive(root/'summary.json',dict(technical_admission=True,owner=owner,protocol_sha=digest(c),complete_ref=report))
  permit=exclusive(old/'queue/MS/formal-permit.json',dict(start_authorization_ref=start,unified_scope=u.OLD_SCOPE,commit=u.BASE,protocol_sha=digest(c),authorized_task_ids=[t['id']for t in c['tasks']],code={},data_binding_ref=data_ref,summary_ref=summary,source_states={}))
  exclusive(old/'queue/MS/runtime-admission.json',dict(owner=owner,permit_ref=permit,protocol_sha=digest(c),integrity_scan_passed=True))
  for model in s.MODELS:
   artifacts={};tasks=[t for t in c['tasks']if t['model']==model]
   for t in tasks:
    out=old/'MS'/('formal-'+model)/t['id'];out.mkdir(parents=True);p=c['resolved_profiles'][t['id']];steps=step_arithmetic(c,t)['train_batches']
    for f in ('best.pt','last.pt'):(out/f).write_bytes(b'synthetic checkpoint; never decoded')
    exclusive(out/'manifest.json',dict(task=t,profile=p,identity={'commit':u.BASE}));exclusive(out/'result.json',dict(id=t['id'],commit=u.BASE,scientific_protocol=u.OLD_PROTOCOL,protocol_sha=digest(c),profile_sha=digest(p),data_sha=data['data_bindings'][t['dataset']][t['id']],best_epoch=1,scheduler_updates=steps,final_test=dict(calls=1,selected='best.pt',sha256=sha(out/'best.pt'),epoch=1)))
    exclusive(out/'budget.json',{'counts':{'adam':steps,'backward':steps,'forward':steps+1}});exclusive(out/'runtime.json',dict(task=t['id'],error=None));(out/'history.jsonl').write_text(json.dumps(dict(epoch=1,steps=steps,validation={'mse':1.},best_epoch=1))+'\n');artifacts[t['id']]={f:ref(out/f)for f in r.FILES}
   exclusive(old/'queue/MS'/('group-'+model)/'complete.json',dict(technical_complete=True,result_review='pending',model=model,task_ids=[t['id']for t in tasks],artifacts=artifacts))
  for k,v in dict(OLD_RESULT=old,START_REF=start,CONTROLLER_REF=controller,CODE_REF=code,ENV_REF=env,OWNER=owner).items():stack.enter_context(patch.object(u,k,v))
  stack.enter_context(patch.object(r.subprocess,'check_output',side_effect=lambda args,**kw:'' if 'status'in args else u.BASE+'\n'))
  pin_fixture_identity(root,old,fc,stack)
  yield old,fc,c

def pin_fixture_identity(root,old,fc,stack,source=None):
 ownership=u.owned_evidence();fref=ref(fc/'controller.json');follower=bound(fref)
 owners=ownership['historical_refs']+[follower['owner']]
 if source is None:
  source=dict(owned_exited=owners,follower_controller_ref=fref,follower_failure_ref=ref(fc/'failure.json'),controller_ref=u.CONTROLLER_REF,failure_ref=ref(old/'queue/controller/failure.json'),progress_ref=ref(old/'queue/controller/progress.json'))
 source_ref=exclusive(root/'identity-source.json',source)
 evidence=dict(purpose='ms203_owned_identity_reconciliation_v1',source_ref=source_ref,follower_ref=fref,historical_refs=owners,upstream=ownership,instances=ownership['instances']+[dict(**follower['owner'],scope=follower['scope'],wave='controller',source=fref)])
 stack.enter_context(patch.object(r,'SOURCE_REF',source_ref));stack.enter_context(patch.object(r,'IDENTITY_REF',exclusive(root/'identity-reconciliation.json',evidence)))
 return evidence

class ProcessIdentity(unittest.TestCase):
 PID=123456787
 def sample(self,ticks='10',namespace=None,host_pid=99):
  value=dict(pid=self.PID,host_pid=host_pid,start_ticks=ticks,namespace=namespace or os.readlink('/proc/self/ns/pid'))if ticks is not None else dict(pid=self.PID,host_pid=None,state='metadata_unavailable')
  return dict(owned_pid_metadata={str(self.PID):value})
 def wave(self,rows,wave='wave-0',scope='fixed-scope'):
  return u.reconcile_wave(scope,wave,{self.PID},rows,dict(path='/fixture/'+wave+'/process.json',sha256='process'),dict(path='/fixture/'+wave+'/memory.jsonl',sha256='memory'))
 def test_same_old_instance_live_rejected_with_bounded_diagnostic(self):
  old=self.wave([self.sample()])[0]
  with patch.object(u,'observe_instance',return_value=dict(kind='present',identity=dict(pid=self.PID,start_ticks='10',state='S',namespace=old['namespace']))),patch.object(os,'kill')as signal:
   with self.assertRaisesRegex(ValueError,'same old instance still live')as error:u.assert_owned_exited([old])
   self.assertIn(str(self.PID),str(error.exception));self.assertIn('memory.jsonl',str(error.exception));self.assertLess(len(str(error.exception)),1600);signal.assert_not_called()
 def test_reused_pid_different_ticks_passes_without_signal(self):
  old=self.wave([self.sample()])[0]
  with patch.object(u,'observe_instance',return_value=dict(kind='present',identity=dict(pid=self.PID,start_ticks='999',state='R',namespace=old['namespace']))),patch.object(os,'kill')as signal:
   self.assertIn(self.PID,u.assert_owned_exited([old]));signal.assert_not_called()
 def test_same_wave_missing_and_duplicate_missing_resolved(self):
  result=self.wave([self.sample(),self.sample(None),self.sample(None)])
  self.assertEqual(len(result),1);self.assertEqual(result[0]['start_ticks'],'10');self.assertEqual(result[0]['missing_sample_count'],2);self.assertEqual(result[0]['missing_first_line'],2)
 def test_different_waves_and_instances_not_merged_by_pid(self):
  one=self.wave([self.sample(),self.sample(None)],'wave-0');two=self.wave([self.sample('20'),self.sample(None)],'wave-1')
  self.assertEqual([(v['wave'],v['start_ticks'])for v in one+two],[('wave-0','10'),('wave-1','20')])
  other=self.wave([self.sample('30'),self.sample(None)],'wave-0','other-scope');self.assertEqual(other[0]['scope'],'other-scope')
 def test_missing_only_wave_cannot_borrow_other_wave_ticks(self):
  self.wave([self.sample()],'wave-0')
  with self.assertRaisesRegex(ValueError,'insufficient'):self.wave([self.sample(None)],'wave-1')
 def test_two_candidate_instances_in_same_wave_rejected(self):
  with self.assertRaisesRegex(ValueError,'conflicting'):self.wave([self.sample(),self.sample('20'),self.sample(None)])
 def test_conflicting_identity_diagnostic_is_bounded(self):
  with self.assertRaisesRegex(ValueError,'conflicting')as error:self.wave([self.sample(str(n))for n in range(20)]+[self.sample(None)])
  detail=json.loads(str(error.exception).split(': ',1)[1]);self.assertEqual(detail['candidate_count'],20);self.assertEqual(len(detail['candidates']),4);self.assertLess(len(str(error.exception)),1600)
 def test_missing_namespace_conflict_rejected(self):
  row=self.sample(None);row['owned_pid_metadata'][str(self.PID)]['namespace']='pid:[wrong]'
  with self.assertRaisesRegex(ValueError,'conflicts'):self.wave([self.sample(),row])
 def test_unregistered_pid_and_incomplete_known_identity_rejected(self):
  with self.assertRaisesRegex(ValueError,'unregistered'):u.reconcile_wave('scope','wave',set(),[self.sample()],{}, {})
  with self.assertRaisesRegex(ValueError,'incomplete'):self.wave([self.sample(namespace='unknown')])
 def test_observer_namespace_conflict_rejected(self):
  old=self.wave([self.sample(namespace='pid:[other]')])[0]
  with patch.object(u,'observe_instance',return_value=dict(kind='absent')):
   with self.assertRaisesRegex(ValueError,'namespace conflict'):u.assert_owned_exited([old])
 def test_two_absent_reads_are_released(self):
  with patch.object(u,'_read_instance',side_effect=[None,None])as read:
   self.assertEqual(u.observe_instance(self.PID),dict(kind='absent'));self.assertEqual(read.call_count,2)
 def test_disappearance_during_read_is_uncertain_and_rejected(self):
  old=self.wave([self.sample()])[0];current=dict(pid=self.PID,start_ticks='99',state='S',namespace=old['namespace'])
  with patch.object(u,'_read_instance',side_effect=[current,None]):
   with self.assertRaisesRegex(ValueError,'current process evidence insufficient'):u.assert_owned_exited([old])
 def test_pid_identity_changes_during_read_are_uncertain(self):
  one=dict(pid=self.PID,start_ticks='99',state='S',namespace=os.readlink('/proc/self/ns/pid'));two=dict(one,start_ticks='100')
  with patch.object(u,'_read_instance',side_effect=[one,two]):self.assertEqual(u.observe_instance(self.PID)['kind'],'uncertain')
 def test_permission_denied_not_treated_as_absence(self):
  with patch.object(u,'_read_instance',side_effect=PermissionError('synthetic denial')):
   with self.assertRaisesRegex(ValueError,'evidence insufficient'):u.assert_owned_exited(self.wave([self.sample()]))
 def test_missing_proc_namespace_after_stat_is_uncertain(self):
  with patch.object(u,'_read_instance',side_effect=FileNotFoundError('namespace vanished')):self.assertEqual(u.observe_instance(self.PID)['kind'],'uncertain')
 def test_run_state_change_does_not_change_instance(self):
  one=dict(pid=self.PID,start_ticks='99',state='R',namespace=os.readlink('/proc/self/ns/pid'));two=dict(one,state='S')
  with patch.object(u,'_read_instance',side_effect=[one,two]):self.assertEqual(u.observe_instance(self.PID),dict(kind='present',identity=two))
 def test_actual_public_preflight_and_import_agree_for_reused_number(self):
  import io,m6_type1_followup_entry as entry
  from contextlib import redirect_stdout
  with source_fixture()as(old,fc,c),tempfile.TemporaryDirectory()as tmp,ExitStack()as stack:
   root=Path(tmp);wave=old/'queue/MS/group-AMD/wave-identity';wave.mkdir()
   (wave/'memory.jsonl').write_text('\n'.join(json.dumps(x)for x in [self.sample(),self.sample(None),self.sample(None)])+'\n');exclusive(wave/'process.json',dict(process_peaks={str(self.PID):1.}))
   pin_fixture_identity(root/'first',old,fc,stack)
   observe=stack.enter_context(patch.object(u,'observe_instance',side_effect=lambda pid:dict(kind='present',identity=dict(pid=pid,start_ticks='99999',state='S',namespace=os.readlink('/proc/self/ns/pid')))))
   self.assertTrue(any(v['start_ticks']is None for v in u.owned_refs()))
   full=r.snapshot();self.assertEqual(len(full['task_ids']),203);pin_fixture_identity(root/'full',old,fc,stack,full)
   self.assertEqual(r.verify_source(),full);self.assertEqual(r.status()['state'],'MS_SEAL_RECOVERY_AWAITING_FULL_START_CHECK')
   stack.enter_context(patch.object(q,'closure',return_value='synthetic-reviewed-closure'));stack.enter_context(patch.object(q,'dynamic',return_value={}));remote=stack.enter_context(patch.object(q,'verify_live_remote'));stack.enter_context(patch.object(q.subprocess,'run',return_value=subprocess.CompletedProcess([],1)))
   approval=q.start_template()
   for key in ('reviewed','execution_permitted','structure_frozen','m6_authorized','budget_authorized'):approval[key]=True
   approval.update(closure_commit='synthetic-reviewed-closure',authorization_basis='synthetic user-approved fixed recovery scope')
   auth=exclusive(root/'synthetic-start.json',approval);output=io.StringIO()
   with patch('sys.argv',['entry','preflight','--approval',auth['path'],'--approval-sha',auth['sha256']]),redirect_stdout(output):exit_code=entry.cli()
   result=json.loads(output.getvalue());self.assertEqual(exit_code,0);self.assertEqual(result['blocked'],[]);self.assertTrue(result['READY_TO_ARM_HANDOFF']);self.assertFalse(result['READY_FOR_GPU_EXECUTION']);remote.assert_called_once();self.assertGreater(observe.call_count,0)
 def test_original_evidence_retained_and_exact_raw_projection_checked(self):
  evidence=r.identity_evidence();source=bound(r.SOURCE_REF)
  self.assertEqual(len(source['owned_exited']),661);self.assertEqual(sum(v['start_ticks']is None for v in source['owned_exited']),329);self.assertEqual(evidence['historical_refs'],source['owned_exited']);self.assertEqual(len({(v['pid'],v['start_ticks'])for v in evidence['instances']}),332)
  with tempfile.TemporaryDirectory()as tmp:
   changed=copy.deepcopy(evidence);changed['historical_refs']=changed['historical_refs'][:-1]
   with patch.object(r,'IDENTITY_REF',exclusive(Path(tmp)/'changed.json',changed)):
    with self.assertRaisesRegex(ValueError,'source mismatch'):r.identity_evidence()

class SourceImport(unittest.TestCase):
 def test_203_read_only_full_integrity(self):
  with source_fixture()as(old,fc,c):
   v=r.snapshot();self.assertEqual((len(v['task_ids']),len(v['artifact_refs'])),(203,1421));self.assertEqual((v['checkpoints_deserialized'],v['numeric_replays'],v['new_test_calls']),(0,0,0))
 def test_different_failure_refused(self):
  with source_fixture()as(old,fc,c):
   p=old/'queue/controller/failure.json';x=json.loads(p.read_text());x['error']='numeric failure';p.write_text(json.dumps(x))
   with self.assertRaises(ValueError):r.snapshot()
 def test_missing_result_refused(self):
  with source_fixture()as(old,fc,c):
   next((old/'MS').glob('formal-*/*/result.json')).unlink()
   with self.assertRaises(ValueError):r.snapshot()
 def test_tampered_checkpoint_checksum_refused_without_decode(self):
  with source_fixture()as(old,fc,c):
   next((old/'MS').glob('formal-*/*/best.pt')).write_bytes(b'changed')
   with self.assertRaises(ValueError):r.snapshot()
 def test_wrong_group_tasks_refused(self):
  with source_fixture()as(old,fc,c):
   p=old/'queue/MS/group-AMD/complete.json';x=json.loads(p.read_text());x['task_ids'][0]=x['task_ids'][1];p.write_text(json.dumps(x))
   with self.assertRaises(ValueError):r.snapshot()
 def test_alive_original_worker_refused(self):
  with source_fixture()as(old,fc,c),patch.object(u,'observe_instance',side_effect=lambda pid:dict(kind='present',identity=dict(pid=pid,start_ticks='1',state='S',namespace=os.readlink('/proc/self/ns/pid')))):
   with self.assertRaises(ValueError):r.snapshot()
 def test_M_already_attempted_refused(self):
  with source_fixture()as(old,fc,c):
   (old/'probe/M').mkdir()
   with self.assertRaises(ValueError):r.snapshot()
 def test_unknown_test_state_refused(self):
  with source_fixture()as(old,fc,c):
   p=next((old/'MS').glob('formal-*/*/result.json'));x=json.loads(p.read_text());x['final_test']={};p.write_text(json.dumps(x))
   receipt=old/'queue/MS'/('group-'+next(t['model']for t in c['tasks']if t['id']==p.parent.name))/'complete.json';g=json.loads(receipt.read_text());g['artifacts'][p.parent.name]['result.json']=ref(p);receipt.write_text(json.dumps(g))
   with self.assertRaises(ValueError):r.snapshot()
 def test_wrong_training_commit_refused(self):
  with source_fixture()as(old,fc,c):
   p=next((old/'MS').glob('formal-*/*/result.json'));x=json.loads(p.read_text());x['commit']='wrong';p.write_text(json.dumps(x))
   receipt=old/'queue/MS'/('group-'+next(t['model']for t in c['tasks']if t['id']==p.parent.name))/'complete.json';g=json.loads(receipt.read_text());g['artifacts'][p.parent.name]['result.json']=ref(p);receipt.write_text(json.dumps(g))
   with self.assertRaises(ValueError):r.snapshot()
 def test_actual_import_and_mixed_base287_seals_distinguish_training_and_sealing(self):
  with source_fixture()as(old,fc,c),tempfile.TemporaryDirectory()as tmp,ExitStack()as stack:
   root=Path(tmp);control=root/'controller';control.mkdir();source=r.snapshot();record=exclusive(root/'source.json',source);stack.enter_context(patch.object(r,'SOURCE_REF',record));stack.enter_context(patch.object(q,'CONTROL',control));stack.enter_context(patch.object(q,'closure',return_value='synthetic-recovery-closure'));stack.enter_context(patch('ch3_runner.code_binding',return_value={'recovery':'synthetic-new-code'}));stack.enter_context(patch.dict(os.environ,{q.SECRET:'synthetic-import-secret'}));exclusive(control/'controller.json',dict(scope=s.ID,owner=q.owner()))
   evidence=bound(r.IDENTITY_REF);evidence['source_ref']=record;stack.enter_context(patch.object(r,'IDENTITY_REF',exclusive(root/'rebound-identity.json',evidence)))
   upstream=bound(q.wait_upstream());ms=bound(upstream['boundaries']['MS']);self.assertEqual(len(ms['task_ids']),203);self.assertEqual(ms['training_commit'],u.BASE);self.assertEqual(ms['seal_execution_commit'],'synthetic-recovery-closure');self.assertEqual(ms['training_code'],source['code']);self.assertEqual(ms['seal_execution_code'],{'recovery':'synthetic-new-code'})
   mc=q.configs()['M_BASE'];ctx=dict(s.context(mc),control=root/'M_BASE');stack.enter_context(patch.object(s,'context',return_value=ctx))
   receipts={model:exclusive(root/(model+'.json'),dict(model=model,technical_complete=True,task_ids=[t['id']for t in mc['tasks']if t['model']==model]))for model in s.MODELS};binding=dict(commit='synthetic-recovery-closure',protocol_sha=digest(mc),code={},environment={},hardware={},source_states={});stack.enter_context(patch.object(q,'dynamic',return_value=binding))
   mr=q.seal_boundary(mc,receipts);base=q.seal_base287_boundary(mr);self.assertEqual(bound(base)['counts'],dict(MS=203,M=84,total=287));self.assertEqual(bound(base)['boundaries'],dict(MS=upstream['boundaries']['MS'],M=mr));q.validate_base287_boundary_light(base)

class Dispatch(unittest.TestCase):
 def test_actual_run_order_no_old_MS_dispatch_or_self_wait(self):
  cs=q.configs();calls=[]
  with tempfile.TemporaryDirectory()as tmp,ExitStack()as stack:
   control=Path(tmp)/'controller';control.mkdir();stack.enter_context(patch.object(q,'CONTROL',control));stack.enter_context(patch.object(q,'stop_check'));stack.enter_context(patch.object(q,'validate_start'));stack.enter_context(patch.object(q,'bound',side_effect=lambda value:{'boundaries':{'MS':{'path':'old-source','sha256':'MS'}}} if value.get('path')=='synthetic-import' else {}))
   stack.enter_context(patch.object(q,'wait_upstream',side_effect=lambda:calls.append('importMS203')or {'path':'synthetic-import','sha256':'source'}));stack.enter_context(patch.object(q,'create_permit',side_effect=lambda c,*args,**kw:calls.append(('permit',c['baseline_unified']['stage']))or {}));stack.enter_context(patch.object(q,'wait_owned',side_effect=lambda stage,*args:calls.append(('owned',stage))));stack.enter_context(patch.object(q,'audit_probe',side_effect=lambda c:calls.append(('audit',c['baseline_unified']['stage']))or {}));stack.enter_context(patch.object(q,'seal_runtime',return_value={}));stack.enter_context(patch.object(q,'ref',side_effect=lambda p:{'path':str(p),'sha256':'synthetic'}));stack.enter_context(patch.object(q,'seal_boundary',side_effect=lambda c,*args:calls.append(('seal',c['baseline_unified']['stage']))or {}));stack.enter_context(patch.object(q,'seal_base287_boundary',side_effect=lambda v:calls.append('base287')or {}));stack.enter_context(patch.object(q,'seal_round2_boundary',side_effect=lambda v:calls.append('round2-371')or {}));stack.enter_context(patch.object(s,'context',side_effect=lambda c:{'control':Path(tmp)/c['baseline_unified']['stage']}))
   q.run({'path':'synthetic-start','sha256':'start'})
   self.assertEqual(calls[0],'importMS203');self.assertLess(calls.index(('seal','M_BASE')),calls.index('base287'));self.assertLess(calls.index('base287'),calls.index(('permit','M_AMEND')));self.assertLess(calls.index('round2-371'),calls.index(('permit','URBAN_SUBSET')))
   self.assertFalse(any(isinstance(v,tuple)and v[1]=='MS'for v in calls));self.assertEqual([v[1]for v in calls if isinstance(v,tuple)and v[0]=='audit'],list(s.STAGES));top=json.loads((control/'complete.json').read_text());self.assertEqual((top['total_runs'],top['imported_ms_runs'],top['base_round2_runs'],top['third_round_runs'],top['round2_effective_runs']),(427,203,287,231,371))
 def test_failure_stops_later_stages(self):
  calls=[];actions={state:(lambda receipts,state=state:calls.append(state))for state in q.STATES if state!='COMPLETE'}
  def failure(receipts):raise RuntimeError('synthetic numeric gate')
  actions['M128_AUTO_AUDIT']=failure
  with self.assertRaises(RuntimeError):q.drive(actions,lambda state:None,lambda:None)
  self.assertNotIn('M128_FORMAL_ALL_BASELINES',calls);self.assertNotIn('AMEND_RESOURCE_NUMERIC_PROBE',calls)
 def test_no_remote_in_runtime_closure_or_worker_source(self):
  for name in ('closure','validate_start','validate_permit'):
   tree=ast.parse((ROOT/'utils/ch3_type1_chain.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name);self.assertNotIn('verify_live_remote',ast.unparse(node));self.assertNotIn('ls-remote',ast.unparse(node))
 def test_public_preflight_retains_remote_check(self):
  with patch.object(q,'readiness',return_value=[])as call,patch.object(q,'upstream_status',return_value={}):q.readiness_report({})
  self.assertEqual(call.call_args.kwargs,{'live_remote':True})
 def test_launcher_namespace_fixed_python_and_no_old_log(self):
  text=(ROOT/'scripts/ch3/start_type1_followup.sh').read_text();self.assertIn(q.SESSION,text);self.assertIn(str(q.LOG),text);self.assertIn(q.PYTHON,text);self.assertNotIn('/baseline-type1-followup-v3/followup-launcher.log',text)
 def test_runner_math_AST_unchanged(self):
  before=ast.parse(subprocess.check_output(['git','show','1134611cd52e4cedb7418d8a9f8a7f6e76512617:ch3_runner.py'],cwd=ROOT,text=True));after=ast.parse((ROOT/'ch3_runner.py').read_text());old={n.name:ast.dump(n,include_attributes=False)for n in before.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};new={n.name:ast.dump(n,include_attributes=False)for n in after.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};self.assertEqual([k for k in old if old[k]!=new[k]],['code_binding'])

class SourceIndex(unittest.TestCase):
 @contextmanager
 def fixture(self):
  with tempfile.TemporaryDirectory()as tmp,ExitStack()as stack:
   root=Path(tmp);oldroot=root/'old';newroot=root/'new';amendroot=newroot/'round2-amendment'
   stack.enter_context(patch.object(u,'OLD_RESULT',oldroot));stack.enter_context(patch.object(r,'RESULT',newroot));stack.enter_context(patch.object(a,'RESULT',amendroot))
   start,_=u.records();configs={'MS':bound(start['config_refs']['MS']),'M':a.parent(),'AMEND':q.configs()['M_AMEND']};boundaries={}
   for stage,c in configs.items():
    directory=oldroot/'MS'if stage=='MS'else newroot/'M_BASE'if stage=='M'else amendroot/'M_AMEND';receipts={};commit=u.BASE if stage=='MS'else'synthetic-new-worker-commit'
    for model in s.MODELS:
     artifacts={};tasks=[t for t in c['tasks']if t['model']==model]
     for t in tasks:
      out=directory/('formal-'+model)/t['id'];p=c['resolved_profiles'][t['id']]
      result=exclusive(out/'result.json',dict(id=t['id'],commit=commit,profile_sha=digest(p),protocol_sha=digest(c),scientific_protocol=c['baseline_unified']['id'],final_test={'calls':1},mse=1e100 if stage=='AMEND'else 1.,mae=1.))
      manifest=exclusive(out/'manifest.json',dict(task=t,profile=p,identity={'commit':commit}));runtime=exclusive(out/'runtime.json',dict(task=t['id'],error=None));artifacts[t['id']]={'result.json':result,'manifest.json':manifest,'runtime.json':runtime}
     receipts[model]=exclusive(root/(stage+'-'+model+'.json'),dict(artifacts=artifacts,technical_complete=True))
    boundaries[stage]=exclusive(root/(stage+'-boundary.json'),dict(task_ids=[t['id']for t in c['tasks']],receipts=receipts,commit=commit))
   yield configs,boundaries
 def test_371_Weather_worse_still_selected_and_old10_retained(self):
  with self.fixture()as(cs,bs):
   value=a.build_index({'MS':bs['MS'],'M':bs['M']},bs['AMEND'],cs['AMEND']);self.assertEqual(len(value['cells']),371);self.assertEqual(value['planned_formal_executions'],399)
   weather=[row for row in value['cells']if row['dataset']=='Weather'];self.assertEqual(len(weather),28);self.assertTrue(all(row['mse']==1e100 and row['supersedes']and row['origin']=='amend1'for row in weather))
   self.assertTrue(all(row['origin']=='base_m128' and row['execution_commit']=='synthetic-new-worker-commit'for row in value['cells']if row['dataset']in ('ETTh1','Exchange')));self.assertTrue(all(row['execution_commit']==u.BASE for row in value['cells']if row['task']=='MS'))
 def test_partial_Weather_or_duplicate_or_wrong_source_rejected(self):
  with self.fixture()as(cs,bs):
   value=a.build_index({'MS':bs['MS'],'M':bs['M']},bs['AMEND'],cs['AMEND'])
   for kind in ('missing','duplicate','old-M','Weather10'):
    v=copy.deepcopy(value)
    if kind=='missing':v['cells'].pop()
    elif kind=='duplicate':v['cells'][0]=v['cells'][1]
    elif kind=='old-M':next(x for x in v['cells']if x['dataset']=='ETTh1')['origin']='original_v3'
    else:next(x for x in v['cells']if x['dataset']=='Weather')['origin']='base_m128'
    with self.assertRaises(ValueError):a.validate_index(v)

class ProbeOwnership(unittest.TestCase):
 observations=[]
 @contextmanager
 def fixture(self,stage,failure=None):
  import m5_formal_entry as tool,ch3_runner
  from utils import ch3_native_execution as native
  with tempfile.TemporaryDirectory()as tmp,ExitStack()as stack:
   c=q.configs()[stage];root=Path(tmp);ctx=dict(s.context(c),control=root/'control',probe_root=root/'probe',fixture=root/'fixture');ctx['probe_root'].mkdir();ctx['control'].mkdir();ctx['fixture'].mkdir();control=root/'controller';control.mkdir();own=q.owner();exclusive(control/'controller.json',dict(owner=own,scope=s.ID))
   groups=s.probe_groups(c)[:2];events=[];binding=dict(commit='synthetic',protocol_sha=digest(c),code={},environment={},hardware={},source_states={});approval=dict(binding,type1_scope=s.ID);exclusive(ctx['probe_root']/'approval.json',approval)
   stack.enter_context(patch.object(s,'context',return_value=ctx));stack.enter_context(patch.object(s,'probe_groups',return_value=groups));stack.enter_context(patch.object(q,'CONTROL',control));stack.enter_context(patch.dict(os.environ,{q.SECRET:'synthetic-audit-secret'}))
   def make(cc,purpose,out,*,task,approval):
    t=next(t for t in c['tasks']if t['id']==task);p=profile(c,t);out=Path(out);out.mkdir(parents=True);cfg=p['training']['scheduler'];trace=[]
    if cfg['name']=='OneCycleLR':trace=[dict(config=cfg,config_sha=digest(cfg),updates=i,scheduler={'last_epoch':i,'_step_count':i+1,'total_steps':cfg['epochs']*cfg['steps_per_epoch']},lr=[.0004],beta1=[.95])for i in range(1,7)]
    else:
     from utils.ch3_type1_scaled import Schedule
     class Optimizer:
      param_groups=[{'lr':1e-4,'betas':tuple(p['training']['betas'])}]
     scheduler=Schedule(Optimizer(),cfg)
     for i in range(6):scheduler.after_successful_update();trace.append(scheduler.state_dict())
    points=[dict(step=i,schema_file=str(out/(str(i)+'.json')),data_file=str(out/(str(i)+'.bin')))for i in range(1,7)]
    for point in points:Path(point['schema_file']).write_text('{}');Path(point['data_file']).write_bytes(b'synthetic')
    tr=dict(id=t['id'],profile_sha=digest(p),initial='i',initial_rng='r',batch_ids=list(range(6)),validation_tail=1,steps=6,final_rng='r',finite=True,trajectory=[dict(step=i,loss=1.,state='s',optimizer='o')for i in range(1,7)],validation=dict(mse=1.,mae=1.,sse=1.,sae=1.,elements=1),final='s',scheduler_trace=trace,M_full_state_trace=points)
    rule=s.numeric_policy(c,t)
    if rule:tr.update(numeric_policy_sha=digest(rule),full_numeric_trace=points)
    if failure and task==groups[0]['representatives'][0]and out.parent.name=='q4':
     if failure=='finite':tr['finite']=False
     elif failure=='identity':tr['initial_rng']='wrong'
     else:tr['trajectory'][0]['loss']=2.
    exclusive(out/'trajectory.json',tr);counts=s.worker_counts(c,t);exclusive(out/'budget.json',{'counts':counts});exclusive(out/'runtime.json',dict(pid=10000000,task=task,error=None));exclusive(out/'config.json',dict(synthetic=True));return dict(purpose=purpose,task=task,output=str(out),budget_file=str(out/'budget.json'))
   def launch(configs,out,monitor):
    out=Path(out);out.mkdir(parents=True);events.append((out.parent.parent.name,out.parent.name));result=dict(failure=None,returncodes=[0]*len(configs),resource_admission=True,elapsed=2. if out.parent.name=='serial'else 1.)
    exclusive(out/'process.json',result);(out/'memory.jsonl').write_text('{}\n');return result
   stack.enter_context(patch.object(tool,'make_config',side_effect=make));stack.enter_context(patch.object(tool,'run_configs',side_effect=launch));stack.enter_context(patch.object(ch3_runner,'_compare_full_numeric_files',return_value=dict(passed=True,state_max_abs=0.,compared_elements=1,exact=True,failures=[])));stack.enter_context(patch('utils.ch3_urban_capture.compare_traces',return_value={}));stack.enter_context(patch('utils.ch3_urban_confirmation.compare_measured',return_value={'passed':True}));full=stack.enter_context(patch.object(native,'validate_probe_completion',side_effect=lambda cc,report:report));immediate=stack.enter_context(patch.object(native,'compare',wraps=native.compare))
   yield c,approval,native,full,immediate,events,ctx
 def test_every_stage_tail0_auto_audit1_actual_entrypoints(self):
  for stage in s.STAGES:
   with self.fixture(stage)as(c,approval,native,full,immediate,events,ctx):
    report=native.run_probe(c,approval);self.assertEqual(full.call_count,0);self.assertGreater(immediate.call_count,0);q.audit_probe(c);self.assertEqual(full.call_count,1);self.observations.append(dict(stage=stage,probe_tail=0,AUTO_AUDIT=1,immediate_gates=immediate.call_count,synthetic=True))
 def test_immediate_numeric_finite_identity_failure_stops_other_groups(self):
  for kind in ('numeric','finite','identity'):
   with self.fixture('M_BASE',kind)as(c,approval,native,full,immediate,events,ctx):
    with self.assertRaises((ValueError,RuntimeError)):native.run_probe(c,approval)
    self.assertEqual(len({g for g,phase in events}),1);self.assertFalse(any(phase=='q2'for g,phase in events));self.assertEqual(full.call_count,0);self.assertTrue((ctx['probe_root']/'failure.json').exists())

from tests import test_m6_round2_amendment as historical_git
class Runtime(unittest.TestCase):
 BRANCH=historical_git.ClosureLineage.BRANCH
 command=historical_git.ClosureLineage.command;commit_file=historical_git.ClosureLineage.commit_file;setUp=historical_git.ClosureLineage.setUp
 authorization=historical_git.ClosureLineage.authorization;remote_calls=historical_git.ClosureLineage.remote_calls;preflight_fixture=historical_git.ClosureLineage.preflight_fixture
 observations=[]
 @contextmanager
 def fixture(self):
  import ch3_runner
  from utils import ch3_type1_execution as execution
  from utils.ch3_native_recovery_records import manifest_projection
  c=q.configs()['M_BASE'];original=s.context;root=self.root/'execution';root.mkdir();control=root/'controller';control.mkdir()
  ctx=dict(original(c),control=root/'stage',probe_root=root/'probe',fixture=root/'fixture',result_root=root/'results')
  for name in ('control','probe_root','fixture'):ctx[name].mkdir()
  secret='synthetic-local-runtime';actual_git=ch3_runner.git
  def local_git(*args):
   with patch.object(ch3_runner,'ROOT',self.repo):return actual_git(*args)
  def binding(config,worker=False):return dict(commit=q.closure(),protocol_sha=digest(config),code={'fixture':sha(self.repo/'implementation.txt')},environment={},hardware={},source_states={})
  def signed(path,body):
   body=dict(body);body['mac']=q.hmac.new(secret.encode(),digest(body).encode(),q.hashlib.sha256).hexdigest();return exclusive(path,body)
  with ExitStack()as stack:
   stack.enter_context(patch.object(ch3_runner,'ROOT',ROOT));stack.enter_context(patch.object(ch3_runner,'git',side_effect=local_git));stack.enter_context(patch.object(q,'CONTROL',control));stack.enter_context(patch.dict(os.environ,{q.SECRET:secret,'TMPDIR':str(ctx['fixture'])}));stack.enter_context(patch.object(s,'context',side_effect=lambda config:ctx if config['baseline_unified']['stage']=='M_BASE'else original(config)));stack.enter_context(patch.object(q,'dynamic',side_effect=binding));stack.enter_context(patch('utils.ch3_m_execution.gpu_environment_reasons',return_value=[]));stack.enter_context(patch('utils.ch3_native_recovery.resource_check'))
   own=q.owner();exclusive(control/'controller.json',dict(scope=s.ID,owner=own));signed(control/'upstream-technical-boundary.json',dict(handoff_scope=s.ID,anchors=q.upstream_anchors(),READY_FOR_GPU_EXECUTION=True,successor_owner=own))
   payload=exclusive(ctx['probe_root']/'synthetic-payload.json',dict(synthetic=True));report=dict(artifacts={payload['path']:payload});complete=exclusive(ctx['probe_root']/'complete.json',report);manifest=exclusive(ctx['control']/'probe-artifact-manifest.json',manifest_projection(report,complete,ctx['probe_root']))
   summary=signed(ctx['control']/'admission-summary.json',dict(purpose='baseline_type1_technical_admission_v1',scope=ctx['probe_scope'],technical_admission=True,manual_review=False,reviewed=False,result_review='pending',owner=own,complete_ref=complete,manifest_ref=manifest,protocol_sha=digest(c),policy_sha=digest(c['baseline_unified']['numeric_policies']),profile_shas={t['id']:digest(profile(c,t))for t in c['tasks']},decisions={g['id']:dict(status='Passed',concurrency=g['planned_q'],coverage=g['coverage'])for g in s.probe_groups(c)},budget=dict(caps=ctx['caps'],refund=False,actual={k:0 for k in ctx['caps']},reserved={k:0 for k in ctx['caps']})))
   start=exclusive(root/'synthetic-start.json',self.authorization());permit=q.create_permit(c,start,False,summary_ref=summary)
   scan=stack.enter_context(patch.object(q,'scan_manifest',wraps=q.scan_manifest));runtime=q.seal_runtime(c,permit);self.assertEqual(scan.call_count,1)
   yield c,ctx,bound(permit),runtime,execution,scan
 def test_actual_permit_config_worker_group_no_remote_or_numeric_replay(self):
  from contextlib import nullcontext
  import ch3_runner
  from utils import ch3_native_execution as native
  with self.remote_calls(forbid=True)as network,self.fixture()as(c,ctx,permit,runtime,execution,scan),ExitStack()as stack:
   audit=stack.enter_context(patch.object(native,'validate_probe_completion',side_effect=AssertionError('formal cannot replay probe')));t=c['tasks'][0];q.validate_permit(c,permit)
   cfg=execution.make_config(c,'ch3_formal',ctx['result_root']/('formal-'+t['model'])/t['id'],task=t['id'],approval=permit,runtime_ref=runtime);execution.validate_worker(c,cfg)
   def compute(configs,out,monitor):
    execution.validate_wave(c,configs,out)
    for value in configs:execution.validate_worker(c,value)
    return dict(failure=None,returncodes=[0]*len(configs),resource_admission=True)
   stack.enter_context(patch.object(ch3_runner,'GPULock',side_effect=lambda _:nullcontext()));stack.enter_context(patch('m5_formal_entry.run_configs',side_effect=compute));stack.enter_context(patch.object(execution,'technical_group',return_value=dict(synthetic=True,technical_complete=True)))
   execution.run_group(c,permit,'DLinear',runtime);self.assertEqual(network,[]);self.assertEqual(scan.call_count,1);audit.assert_not_called();self.observations.append(dict(stage='M_BASE',remote=0,full_probe_audit=0,runtime_manifest_scans=1,real_entrypoints=True))
 def test_wrong_commit_scope_profile_and_local_bytes_rejected(self):
  with self.remote_calls(forbid=True)as network,self.fixture()as(c,ctx,permit,runtime,execution,scan):
   for key,value in (('commit',self.implementation),('type1_scope',u.OLD_SCOPE),('profile_shas',{})):
    bad=copy.deepcopy(permit);bad[key]=value
    with self.assertRaises((ValueError,PermissionError)):q.validate_permit(c,bad)
   changed=copy.deepcopy(c);changed['resolved_profiles'][changed['tasks'][0]['id']]['training']['batch']=32
   with self.assertRaises(ValueError):s.validate(changed)
   (self.repo/'implementation.txt').write_text('dirty fixture\n')
   with self.assertRaises(ValueError):q.validate_permit(c,permit)
   self.assertEqual(network,[])
 def test_public_preflight_once_then_offline_local_authorization(self):
  import io,m6_type1_followup_entry as entry
  from contextlib import redirect_stdout
  approval=exclusive(self.root/'synthetic-preflight.json',self.authorization());output=io.StringIO()
  with self.preflight_fixture(),self.remote_calls()as network,patch('sys.argv',['entry','preflight','--approval',approval['path']]),redirect_stdout(output):result=entry.cli()
  self.assertEqual(result,0);self.assertEqual(len(network),1);self.assertEqual(json.loads(output.getvalue())['blocked'],[])
  self.command('remote','set-url','origin',str(self.root/'absent.git'))
  with self.remote_calls(forbid=True)as network:self.assertEqual(q.closure(),self.head);q.validate_start(self.authorization())
  self.assertEqual(network,[])

class Lifecycle(unittest.TestCase):
 def test_synthetic_tmux_owned_stop_preserves_old_v3(self):
  import tests.test_m6_type1_followup_v2 as historical
  with patch.dict(historical.__dict__,s=s,q=q):historical.CPUPlumbingTests().test_synthetic_tmux_owned_waiter_stop()
 def test_foreign_scope_safe_stop_refused(self):
  with tempfile.TemporaryDirectory()as tmp,patch.object(q,'CONTROL',Path(tmp)),patch.object(q,'signal_owned')as signal:
   exclusive(Path(tmp)/'controller.json',dict(scope=u.OLD_SCOPE,owner=u.OWNER))
   with self.assertRaises(PermissionError):q.safe_stop()
   signal.assert_not_called()

if __name__=='__main__':unittest.main()
