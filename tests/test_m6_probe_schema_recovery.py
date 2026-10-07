"""No-model regression for strict M schemas and one fixed partial-probe adoption."""
import ast,copy,json,math,os,struct,subprocess,sys,tempfile,unittest
from contextlib import contextmanager,ExitStack
from pathlib import Path
from unittest.mock import patch
import ch3_runner as runner
from utils import ch3_type1_tasks as s,ch3_type1_chain as q,ch3_native_execution as native,ch3_probe_schema_recovery as recovery
from utils.ch3_contract import ROOT,digest,profile,task_by_id,numeric_probe_policy
from utils.ch3_native_recovery_records import bound,ref,exclusive,manifest_projection,scan_manifest,process_refs

def validation(p,tail=1):
    n=(p['training']['eval_batch']+(tail or p['training']['eval_batch']))*p['pred_len']
    if p.get('task')!='M':return dict(mse=1.,mae=1.,sse=float(n),sae=float(n),elements=n)
    rows=[dict(index=i,name=name,mse=1.,mae=1.,sse=float(n),sae=float(n),elements=n)for i,name in enumerate(p['features'])]
    return dict(mse=1.,mae=1.,sse=float(n*p['C']),sae=float(n*p['C']),elements=n*p['C'],channels=rows,MS_target_diagnostic=rows[p['target_idx']])

def trajectory(c,t,out,rule=None):
    p=profile(c,t);out=Path(out);out.mkdir(parents=True,exist_ok=True);points=[]
    entries=[dict(path=name,dtype='torch.float64',shape=[1],mode=mode,offset=i*8,nbytes=8,numel=1)for i,(name,mode)in enumerate([('model/parameter/w','bounded'),('model/buffer/b','bounded'),('gradient/w','bounded'),('optimizer/w/exp_avg','bounded'),('optimizer/w/exp_avg_sq','bounded'),('optimizer/w/step','exact')])]
    schema=exclusive(out/'schema.json',dict(policy_id=rule['id']if rule else t['group']+'-exact-full-state',entries=entries,optimizer_groups=[{'params':['w']}],optimizer_non_tensor_state={},total_bytes=48))
    for i in range(1,7):
        path=out/(str(i)+'.bin');path.write_bytes(struct.pack('<6d',1.,1.,1.,1.,1.,float(i)));points.append(dict(step=i,schema_file=schema['path'],schema_sha=schema['sha256'],data_file=str(path),data_sha=ref(path)['sha256'],bytes=48))
    cfg=p['training']['scheduler'];trace=[]
    if cfg['name']=='OneCycleLR':trace=[dict(config=cfg,config_sha=digest(cfg),updates=i,scheduler=dict(last_epoch=i,_step_count=i+1,total_steps=cfg['epochs']*cfg['steps_per_epoch']),lr=[.0004],beta1=[.95])for i in range(1,7)]
    else:
        from utils.ch3_type1_scaled import Schedule
        class Opt:
            def __init__(self):self.param_groups=[dict(lr=1e-4,betas=tuple(p['training']['betas']))]
        sch=Schedule(Opt(),cfg)
        for i in range(6):sch.after_successful_update();trace.append(sch.state_dict())
    memory=[dict(step=i,allocated=0,reserved=0,rss_before_hash=100,rss_after_hash=100)for i in range(1,7)]
    tr=dict(id=t['id'],profile_sha=digest(p),initial='initial',initial_rng='rng',batch_ids=list(range(6)),validation_tail=1,steps=6,final_rng='rng',finite=True,trajectory=[dict(step=i,loss=1.,state='state',optimizer='optimizer')for i in range(1,7)],validation=validation(p),final='state',scheduler_trace=trace,M_full_state_trace=points,time_mark=p.get('time_mark'),threads=4,affinity=[],memory=memory,memory_review=runner.memory_growth_review(memory))
    if rule:
        tr['numeric_policy_sha']=digest(rule)
        if rule['kind']=='full_float_state':tr['full_numeric_trace']=points
        else:
            tr['numeric_trace']=[dict(step=i,parameter_name=rule['parameter'],optimizer_parameter_id=0,exact_model='exact',exact_optimizer='exact',exact_gradients='exact',tensors={k:dict(dtype=rule['dtype'],shape=rule['shape'],values=[1.]*math.prod(rule['shape']))for k in rule['tensor_fields']})for i in range(1,7)]
    return tr

@contextmanager
def attempt():
    from utils import ch3_ms_seal_recovery as ms,ch3_round2_amendment as amendment
    with ExitStack()as stack:
        for module,names in ((s,('RESULT','context')),(ms,('RESULT',)),(amendment,('RESULT',)),(q,('CONTROL','LOG','SESSION','ENTRY','WRAPPER','PROBE_RECOVERY','QUEUE_LOCK')),(recovery,('ACTIVE',))):
            for name in names:stack.enter_context(patch.object(module,name,getattr(module,name)))
        recovery.ACTIVE=False;recovery.activate();yield

class Schema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.configs=q.configs()
    def pair(self,stage='M_BASE',model='TimeMixer',dataset='ETTh1',rule=None):
        c=self.configs[stage];t=next(t for t in c['tasks']if t['model']==model and t['dataset']==dataset)
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name)
        x=trajectory(c,t,root/'a',rule if rule is not None else numeric_probe_policy(c,t));y=trajectory(c,t,root/'b',rule if rule is not None else numeric_probe_policy(c,t))
        return c,t,x,y,root
    def test_actual_saved_failure_and_bounded_offline_self_review(self):
        old=bound(recovery.SOURCE_REF);failure=bound(old['refs']['probe_failure']);self.assertEqual(failure['error'],"ValueError('validation schema changed')")
        proof=bound(recovery.REUSE_REF);self.assertEqual((proof['retained_groups'],proof['retained_trajectories'],proof['offline_numeric_comparisons']),(12,97,49));self.assertFalse(proof['whole_M_probe_admission'])
        self.assertEqual(sum('self_comparison'in v for v in proof['trajectory_reviews'].values()),1)
        self.assertTrue(all(v.get('self_comparison',{'passed':True})['passed']for v in proof['trajectory_reviews'].values()))
    def test_M_without_legacy_marker_and_historical_marker(self):
        c,t,x,y,root=self.pair()
        self.assertNotIn('m_experiment',c)
        for marker in (False,True):
            cc=dict(c)
            if marker:cc['m_experiment']={'legacy':True}
            with patch.object(native.scope,'context',return_value={'probe_root':root}):self.assertTrue(native.compare(cc,t,x,y)['passed'])
    def test_exact_named_full_branches(self):
        full=dict(kind='full_float_state',id='synthetic-full',state_atol=1e-6,metric_atol=1e-6,loss_atol=1e-6,rtol=0,equal_nan=False)
        named=dict(kind='named_tensor',id='synthetic-named',parameter='w',dtype='torch.float64',shape=[1],tensor_fields=['parameter','gradient','exp_avg','exp_avg_sq'],atol=1e-6,metric_atol=1e-6)
        for rule in (None,named,full):
            with self.subTest(kind=rule and rule['kind']),patch('utils.ch3_contract.numeric_probe_policy',return_value=rule):
                c,t,x,y,root=self.pair(model='AMD',rule=rule)
                with patch.object(native.scope,'context',return_value={'probe_root':root}):
                    # named_tensor is a historical shared-comparator policy;
                    # current successor registries contain only exact/full.
                    compare=runner.compare_probe_trajectories if rule and rule['kind']=='named_tensor'else native.compare
                    self.assertTrue(compare(c,t,x,y)['passed'])
                    bad=copy.deepcopy(y);bad['trajectory'][0]['loss']+=1
                    self.assertFalse(compare(c,t,x,bad)['passed'])
                    if rule and rule['kind']=='named_tensor':
                        bad=copy.deepcopy(y);bad['numeric_trace'][0]['tensors']['gradient']['values'][0]+=1
                        self.assertFalse(compare(c,t,x,bad)['passed'])
    def test_M_all_current_domains_and_all_channel_profiles(self):
        for stage in ('M_BASE','M_AMEND','M_ALL'):
            c=self.configs[stage]
            for t in c['tasks']:
                p=profile(c,t);v=validation(p);before=copy.deepcopy(v);self.assertEqual(runner._probe_validation_metrics(c,t,v,1),{k:v[k]for k in ('mse','mae','sse','sae','elements')});self.assertEqual(v,before)
    def test_MS_does_not_accept_M_projection(self):
        for stage in ('URBAN_SUBSET','EPF_ALL'):
            c=self.configs[stage];t=c['tasks'][0];v=validation(profile(c,t));self.assertEqual(runner._probe_validation_metrics(c,t,v),v)
            with self.assertRaises(ValueError):runner._probe_validation_metrics(c,t,dict(v,channels=[],MS_target_diagnostic={}))
    def test_task_profile_conflict_foreign_task_and_metric_contract_refused(self):
        c,t,x,y,root=self.pair()
        for kind in ('task','foreign','scope','supervised','output','C'):
            cc=copy.deepcopy(c);tt=task_by_id(cc,t['id']);p=cc['resolved_profiles'][t['id']]
            if kind=='task':tt['task']='MS'
            elif kind=='foreign':tt=copy.deepcopy(tt);tt['id']='foreign'
            elif kind=='scope':p['metric_scope']='target_only'
            elif kind=='supervised':p['supervised_channels']=[p['target_idx']]
            elif kind=='output':p['output_order']=list(reversed(p['output_order']))
            else:p['C']+=1
            with self.assertRaises(ValueError):runner._probe_validation_metrics(cc,tt,x['validation'],1)
    def test_strict_schema_elements_channel_order_and_aggregation(self):
        c,t,x,y,root=self.pair()
        for kind in ('missing','extra','order','name','count','elements','mse','channel_sse','target','tail'):
            v=copy.deepcopy(x['validation']);tail=1
            if kind=='missing':v.pop('channels')
            elif kind=='extra':v['unspecified']=True
            elif kind=='order':v['channels'].reverse()
            elif kind=='name':v['channels'][0]['name']='wrong'
            elif kind=='count':v['channels'].pop()
            elif kind=='elements':v['elements']+=1
            elif kind=='mse':v['mse']+=1
            elif kind=='channel_sse':v['channels'][0]['sse']+=100;v['channels'][0]['mse']=v['channels'][0]['sse']/v['channels'][0]['elements']
            elif kind=='target':v['MS_target_diagnostic']=v['channels'][0]
            else:tail=128
            with self.assertRaises(ValueError,msg=kind):runner._probe_validation_metrics(c,t,v,tail)
    def test_NaN_Inf_and_excessive_loss_state_metric_still_refused(self):
        c,t,x,y,root=self.pair()
        for value in (float('nan'),float('inf')):
            bad=copy.deepcopy(x);bad['validation']['channels'][0]['mse']=value
            with self.assertRaises(ValueError):runner.compare_probe_trajectories(c,t,x,bad)
        bad=copy.deepcopy(y);bad['trajectory'][0]['loss']+=1
        self.assertFalse(runner.compare_probe_trajectories(c,t,x,bad)['passed'])
        bad=copy.deepcopy(y);v=bad['validation']
        for row in v['channels']:row.update(mse=2.,sse=2.*row['elements'])
        v.update(mse=2.,sse=2.*v['elements'])
        self.assertFalse(runner.compare_probe_trajectories(c,t,x,bad)['passed'])
        point=bad['full_numeric_trace'][0];Path(point['data_file']).write_bytes(struct.pack('<6d',3.,1.,1.,1.,1.,1.));point['data_sha']=ref(point['data_file'])['sha256']
        self.assertFalse(runner.compare_probe_trajectories(c,t,x,bad)['passed'])
    def test_OneCycle_and_type1_LR_beta_step_remain_exact(self):
        for stage in ('M_BASE','M_ALL'):
            c,t,x,y,root=self.pair(stage)
            for kind in ('lr','beta','updates'):
                bad=copy.deepcopy(y)
                if kind=='updates':bad['scheduler_trace'][0]['updates']+=1
                elif kind=='lr':
                    if stage=='M_BASE':bad['scheduler_trace'][0]['lr'][0]*=2
                    else:bad['scheduler_trace'][0]['groups'][0]['lr']*=2
                elif stage=='M_BASE':bad['scheduler_trace'][0]['beta1'][0]=.1
                else:bad['scheduler_trace'][0]['groups'][0]['betas'][0]=.1
                with patch.object(native.scope,'context',return_value={'probe_root':root}),self.assertRaises(ValueError):native.compare(c,t,x,bad)
    def test_bounded_policy_compares_aggregate_not_new_channel_effect_gate(self):
        c,t,x,y,root=self.pair();v=y['validation'];n=v['channels'][0]['elements']
        for i,delta in ((0,1.),(1,-1.)):
            row=v['channels'][i];row['sse']+=delta;row['mse']=row['sse']/n
        v['MS_target_diagnostic']=v['channels'][profile(c,t)['target_idx']]
        self.assertTrue(runner.compare_probe_trajectories(c,t,x,y)['passed'])

class PartialRecovery(unittest.TestCase):
    observations=[]
    @contextmanager
    def fixture(self):
        import m5_formal_entry as tool
        c=q.configs()['M_BASE'];groups=s.probe_groups(c)
        with tempfile.TemporaryDirectory()as temp,ExitStack()as stack:
            root=Path(temp);old=root/'old';new=root/'new';ctx=dict(s.context(c),control=new/'queue',probe_root=old/'probe/M_BASE',fixture=root/'fixture',result_root=new/'results');ctx['probe_root'].mkdir(parents=True);ctx['control'].mkdir(parents=True);ctx['fixture'].mkdir();control=new/'controller';control.mkdir();events=[];counter=[10000000]
            binding=dict(commit=recovery.OLD_COMMIT,protocol_sha=digest(c),code={},environment={},hardware={'cpu_affinity':[]},source_states={});permit=dict(binding,type1_scope=s.ID,caps=s.probe_budget(c)['caps']);exclusive(ctx['probe_root']/'approval.json',permit)
            stack.enter_context(patch.object(s,'context',return_value=ctx));stack.enter_context(patch.object(q,'CONTROL',control));stack.enter_context(patch.object(q,'PROBE_RECOVERY',recovery));stack.enter_context(patch.object(q,'dynamic',return_value=binding));stack.enter_context(patch.object(recovery,'OLD_RESULT',old));stack.enter_context(patch.object(recovery,'ACTIVE',True));stack.enter_context(patch.dict(os.environ,{q.SECRET:'synthetic-schema-secret'}));exclusive(control/'controller.json',dict(owner=q.owner(),scope=s.ID))
            def make(cc,purpose,out,*,task,approval):
                t=task_by_id(c,task);out=Path(out);phase=out.parent.name;cost=s.worker_counts(c,t);counter[0]+=1;pid=counter[0]
                cfg=dict(task=task,output=str(out),approval=approval,protocol_sha=digest(c),successor_scope=ctx['probe_scope'],successor_phase=phase,prefix_files={},limits=dict(**cost,seconds=1800),purpose=purpose,unified_stage='M_BASE',type1_scope=s.ID,protocol_file=str(s.file('M_BASE')),session_root=str(ctx['probe_root']),artifact_root=str(out),resume=False,kernel_probe=False,device='cuda:0',ids=[task],bound_files={},author_files=c['sources'].get(t['model'],{}).get('files',{}),metadata_files={})
                tr=trajectory(c,t,out,numeric_probe_policy(c,t));exclusive(out/'trajectory.json',tr);exclusive(out/'config.json',cfg);exclusive(out/'budget.json',dict(counts=cost,by_pid={str(pid):cost}));exclusive(out/'runtime.json',dict(pid=pid,task=task,error=None));(out/'audit.jsonl').write_text('{"event":"synthetic-no-model"}\n')
                return dict(cfg,budget_file=str(out/'budget.json'),pid=pid)
            def launch(configs,out,monitor):
                out=Path(out);out.mkdir(parents=True);events.extend((v['task'],out.parent.name)for v in configs);peaks={str(v['pid']):1 for v in configs};v=dict(failure=None,returncodes=[0]*len(configs),resource_admission=True,elapsed=2. if out.parent.name=='serial'else 1.,exit_transitions_resolved=True,process_attribution='Measured',process_peaks=peaks,cpu_peaks=peaks)
                exclusive(out/'process.json',v);(out/'memory.jsonl').write_text(json.dumps({'owned_pid_metadata':{pid:dict(pid=int(pid),start_ticks='1',namespace='pid:[1]',host_pid=int(pid))for pid in peaks}})+'\n');return v
            stack.enter_context(patch.object(tool,'make_config',side_effect=make));stack.enter_context(patch.object(tool,'run_configs',side_effect=launch))
            # Drive the actual probe with the frozen old comparator, stopping
            # at the same 97th synthetic worker. No run_probe/compare mock.
            old_source=subprocess.check_output(['git','show',recovery.OLD_COMMIT+':ch3_runner.py'],cwd=ROOT,text=True);node=next(n for n in ast.parse(old_source).body if isinstance(n,ast.FunctionDef)and n.name=='compare_probe_trajectories');namespace=dict(digest=digest,profile=profile,_compare_full_numeric_files=runner._compare_full_numeric_files);exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen-old-comparator','exec'),namespace)
            with patch.object(runner,'compare_probe_trajectories',namespace['compare_probe_trajectories']),self.assertRaisesRegex(ValueError,'validation schema changed'):native.run_probe(c,permit)
            self.assertEqual(len(events),97);failure=ref(ctx['probe_root']/'failure.json');previous=bound(failure);self.assertEqual(len(previous['decisions']),12)
            refs={'config':exclusive(root/'config.json',c),'producer_permit':exclusive(root/'producer.json',permit),'probe_failure':failure,'authorization':exclusive(root/'authorization.json',dict(closure_commit=recovery.OLD_COMMIT))}
            refs['controller']=exclusive(root/'old-controller.json',dict(authorization=refs['authorization']));refs['failure']=exclusive(root/'old-failure.json',dict(error="RuntimeError('owned child technical failure: M_BASE-probe')"));refs['child']=exclusive(root/'old-child.json',dict(pid=10000000,start_ticks='1'))
            msids=['MS-'+str(i)for i in range(203)];refs['source_verification']=exclusive(root/'source-verification.json',dict(task_ids=msids,receipts={},training_protocol_sha='original-MS'))
            refs['MS_boundary']=exclusive(root/'MS.json',dict(technical_complete=True,training_commit='df6a16403e10d51097db8c88829909c533d15652',seal_execution_commit=recovery.OLD_COMMIT,task_ids=msids,receipts={},protocol_sha='original-MS',source_verification_ref=refs['source_verification']))
            refs['upstream_boundary']=exclusive(root/'upstream.json',dict(boundaries={'MS':refs['MS_boundary']},owned_exited=[]));refs['exit_verification']=exclusive(root/'exit.json',dict(owned_exited=[{'pid':10000000,'start_ticks':'1'}]));anchor=exclusive(root/'source-anchors.json',dict(producer_commit=recovery.OLD_COMMIT,refs=refs));stack.enter_context(patch.object(recovery,'SOURCE_REF',anchor))
            proof=recovery.compile_reuse(c);proof_ref=exclusive(root/'reuse.json',proof);stack.enter_context(patch.object(recovery,'REUSE_REF',proof_ref));events.clear();ctx['probe_root']=new/'probe/M_BASE';ctx['probe_root'].mkdir(parents=True)
            fresh=dict(permit,commit='synthetic-new-reviewed-head',probe_recovery_ref=proof_ref,execution_attempt='M_BASE-probe-schema-r1');exclusive(ctx['probe_root']/'approval.json',fresh)
            yield c,fresh,ctx,events,proof,root,stack
    def test_actual_resume_dispatches_only71_and_final21_group_audit_manifest(self):
        with self.fixture()as(c,permit,ctx,events,proof,root,stack):
            report=native.run_probe(c,permit);self.assertEqual(len(events),71);self.assertEqual(len(report['decisions']),21)
            oldids={t for g in s.probe_groups(c)[:12]for t in g['representatives']};self.assertFalse(any(t in oldids for t,phase in events));self.assertNotIn((s.probe_groups(c)[12]['representatives'][0],'serial'),events)
            self.assertEqual(report['budget']['historical_actual'],{'adam':582,'backward':582,'forward':776});self.assertEqual(report['budget']['new_actual'],{'adam':426,'backward':426,'forward':568});self.assertEqual(report['budget']['actual'],{'adam':1008,'backward':1008,'forward':1344})
            audit=stack.enter_context(patch.object(native,'validate_probe_completion',wraps=native.validate_probe_completion));compare=stack.enter_context(patch.object(native,'compare',wraps=native.compare));summary=q.audit_probe(c);self.assertEqual(audit.call_count,1);self.assertEqual(compare.call_count,107);self.assertTrue(bound(summary)['technical_admission']);self.assertEqual(len(bound(summary)['decisions']),21)
            manifest=bound(bound(summary)['manifest_ref']);scan_manifest(manifest,bound(summary)['complete_ref'],ctx['probe_root']);self.assertTrue(any(not Path(x['path']).is_relative_to(ctx['probe_root'])for x in manifest['rows']))
            self.observations.append(dict(synthetic=True,retained_workers=97,new_workers=71,probe_tail_audits=0,AUTO_AUDIT=1,groups=21,retained_numeric_replays_at_final_audit=0,final_new_or_mixed_comparisons=compare.call_count))
    def test_tampered_retained_bytes_rejected_before_dispatch(self):
        with self.fixture()as(c,permit,ctx,events,proof,root,stack):
            path=next(Path(p)for p in proof['artifacts']if p.endswith('trajectory.json'));path.write_text('{}')
            with self.assertRaises(ValueError):native.run_probe(c,permit)
            self.assertEqual(events,[])
    def test_wrong_config_and_unregistered_old_permit_refused(self):
        with self.fixture()as(c,permit,ctx,events,proof,root,stack):
            changed=copy.deepcopy(c);changed['resolved_profiles'][changed['tasks'][0]['id']]['training']['batch']=32
            with self.assertRaises(ValueError):recovery.load_seed(changed,permit)
            for key,value in [('probe_recovery_ref',{'path':'wrong','sha256':'wrong'}),('execution_attempt','old')]:
                bad=dict(permit);bad[key]=value
                with self.assertRaises(PermissionError):recovery.load_seed(c,bad)
            self.assertEqual(events,[])
    def test_unknown_failure_missing_MS_or_live_instance_refused(self):
        with self.fixture()as(c,permit,ctx,events,proof,root,stack):
            source=bound(recovery.SOURCE_REF)
            for kind in ('failure','MS','live'):
                refs=copy.deepcopy(source['refs'])
                if kind=='failure':refs['failure']=exclusive(root/'bad-failure.json',{'error':'another error'})
                elif kind=='MS':
                    v=bound(refs['MS_boundary']);v['task_ids'].pop();refs['MS_boundary']=exclusive(root/'bad-MS.json',v)
                else:refs['exit_verification']=exclusive(root/'live.json',dict(owned_exited=[q.owner()]))
                new=exclusive(root/(kind+'-anchors.json'),dict(source,refs=refs))
                with patch.object(recovery,'SOURCE_REF',new),self.assertRaises((ValueError,PermissionError)):recovery.verify_registered_source()
    def test_MS_seal_adopted_without_snapshot_checksum_or_model(self):
        from utils import ch3_ms_seal_recovery as ms
        with self.fixture()as(c,permit,ctx,events,proof,root,stack),patch.object(ms,'snapshot',side_effect=AssertionError('no repeat MS scan')),patch.object(ms,'verify_source',side_effect=AssertionError('no repeat MS import')),patch.object(q,'closure',return_value='synthetic-reviewed-head'):
            v=bound(recovery.import_ms());self.assertEqual(v['boundaries']['MS'],bound(recovery.SOURCE_REF)['refs']['MS_boundary']);self.assertEqual(v['new_MS_test_calls'],0);self.assertEqual(v['new_MS_training_calls'],0);q.validate_upstream_boundary_light(ref(q.CONTROL/'upstream-technical-boundary.json'))
    def test_partial12_cannot_authorize_whole_M_stage(self):
        with self.fixture()as(c,permit,ctx,events,proof,root,stack):
            bad=dict(purpose='native_successor_probe_complete_v1',scope=ctx['probe_scope'],plan=s.plan(c),execution_complete=True,decisions=proof['decisions'])
            with self.assertRaises(ValueError):native.validate_probe_completion(c,bad)
    def test_new_numeric_finite_identity_failure_stops_without_resource_fallback(self):
        import m5_formal_entry as tool
        for kind in ('numeric','finite','identity'):
            with self.subTest(kind=kind),self.fixture()as(c,permit,ctx,events,proof,root,stack):
                make=tool.make_config.side_effect
                def inject(*args,**kwargs):
                    cfg=make(*args,**kwargs);path=Path(cfg['output'])/'trajectory.json';tr=json.loads(path.read_text())
                    if kind=='numeric'and Path(cfg['output']).parent.name=='q4':tr['trajectory'][0]['loss']+=1
                    elif kind=='finite':tr['finite']=False
                    elif kind=='identity':tr['profile_sha']='wrong'
                    path.write_text(json.dumps(tr));return cfg
                stack.enter_context(patch.object(tool,'make_config',side_effect=inject))
                with self.assertRaises(ValueError):native.run_probe(c,permit)
                self.assertTrue(events);self.assertTrue(all(task_by_id(c,t)['model']=='TimeMixer'and task_by_id(c,t)['dataset']=='ETTh1'for t,phase in events));self.assertFalse(any(phase=='q2'for t,phase in events));self.assertTrue((ctx['probe_root']/'failure.json').exists())

class Integration(unittest.TestCase):
    def test_review_only_template_old_start_refused_and_fresh_attempt(self):
        with attempt():
            template=q.start_template();self.assertIsNone(template['closure_commit']);self.assertFalse(template['execution_permitted']);self.assertEqual(template['upstream_anchors']['probe_reuse_ref'],recovery.REUSE_REF);self.assertFalse(s.RESULT.exists());self.assertFalse(q.LOG.exists())
            old=bound(bound(recovery.SOURCE_REF)['refs']['authorization'])
            with patch.object(q,'closure',return_value='new-reviewed-head'),self.assertRaises(PermissionError):q.validate_start(old)
            self.assertTrue(all(s.context(c)['fixture'].is_relative_to(recovery.PACKAGE)for c in q.configs().values()))
    def test_new_entry_is_local_and_all_child_commands_use_it(self):
        with attempt():self.assertEqual(q.ENTRY,recovery.ENTRY);self.assertEqual(q.WRAPPER,recovery.WRAPPER);self.assertEqual(q.SESSION,recovery.SESSION)
        self.assertNotIn('AMD-m-baselines-v1',recovery.ENTRY.read_text());self.assertIn(str(q.PYTHON),recovery.WRAPPER.read_text())
    def test_production_functions_and_science_unchanged(self):
        old=bound(recovery.SOURCE_REF);deltas=recovery.verify_production_inheritance(bound(old['refs']['producer_permit']))
        self.assertNotIn('update',deltas.get('ch3_runner.py',[]));self.assertNotIn('evaluate',deltas.get('ch3_runner.py',[]));self.assertNotIn('probe_worker',deltas.get('ch3_runner.py',[]));self.assertEqual([len(c['tasks'])for c in q.configs().values()],[84,112,28,35,168])
    def test_formal_permit_config_worker_runtime_no_probe_or_remote_replay(self):
        from tests.test_m6_ms_seal_recovery import Runtime
        for method in ('test_actual_permit_config_worker_group_no_remote_or_numeric_replay','test_wrong_commit_scope_profile_and_local_bytes_rejected'):
            case=Runtime();case.setUp()
            try:
                with attempt():getattr(case,method)()
            finally:case.doCleanups()
    def test_existing_exact_stage_seals_and_chain_order(self):
        from tests.test_m6_ms_seal_recovery import Seal,Dispatch
        case=Seal();case.test_reproduces_original_duplicate_keyword();case.test_unified_MS_seal();case.test_unified_M_seal()
        with attempt():Dispatch().test_actual_run_order_no_old_MS_dispatch_or_self_wait();Dispatch().test_failure_stops_later_stages()
    def test_synthetic_owned_lifecycle_new_entry_only(self):
        from tests.test_m6_ms_seal_recovery import Lifecycle
        with attempt():Lifecycle().test_foreign_scope_safe_stop_refused()
        import shlex,time
        with tempfile.TemporaryDirectory()as tmp,attempt():
            root=Path(tmp);pidfile=root/'pid';session='fixture-schema-'+str(os.getpid())+'-'+str(time.time_ns())
            code="import os,time;from pathlib import Path;Path("+repr(str(pidfile))+").write_text(str(os.getpid()));time.sleep(30)"
            command=shlex.join([q.PYTHON,'-B','-c',code,str(q.ENTRY),'start'])
            subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),command],check=True)
            for _ in range(100):
                if pidfile.exists():break
                time.sleep(.02)
            self.assertTrue(pidfile.exists());owner=q.owner(int(pidfile.read_text()));exclusive(root/'controller.json',dict(scope=s.ID,owner=owner))
            with patch.object(q,'CONTROL',root):v=q.safe_stop()
            self.assertTrue(v['supervisor_signal_sent']);self.assertFalse(v['old_chain_signal_sent'])
            for _ in range(100):
                if not q.same(owner):break
                time.sleep(.02)
            self.assertFalse(q.same(owner))
            for _ in range(100):
                if subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
                time.sleep(.02)
            self.assertNotEqual(subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode,0)
    def test_runtime_has_no_network_or_full_probe_audit(self):
        from tests.test_m6_ms_seal_recovery import Dispatch
        Dispatch().test_no_remote_in_runtime_closure_or_worker_source();Dispatch().test_public_preflight_retains_remote_check()
    def test_no_torch_import_or_real_compute(self):self.assertNotIn('torch',sys.modules)

from tests import test_m6_ms_seal_recovery as historical_recovery
class RetainedIdentityRegression(historical_recovery.ProcessIdentity):
    def test_actual_public_preflight_and_import_agree_for_reused_number(self):
        # The historical startup fixture must not consume today's real failed
        # output/log. Keep every inherited PID/source/CLI assertion unchanged.
        from utils import ch3_ms_seal_recovery as ms,ch3_round2_amendment as amendment
        with tempfile.TemporaryDirectory()as tmp,ExitStack()as stack:
            root=Path(tmp)
            for module,name,value in ((s,'RESULT',root/'fresh'),(ms,'RESULT',root/'fresh'),(q,'CONTROL',root/'fresh/controller'),(q,'LOG',root/'fresh.log'),(amendment,'RESULT',root/'fresh/amendment')):
                stack.enter_context(patch.object(module,name,value))
            super().test_actual_public_preflight_and_import_agree_for_reused_number()

if __name__=='__main__':unittest.main()
