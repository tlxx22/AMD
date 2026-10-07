"""Real CPU model admission plus no-model completed-prefix recovery fixtures."""
import ast
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stdout
from unittest.mock import patch

from utils import ch3_amend_ett_identity_recovery as r, ch3_moderntcn_etth1_recovery as numeric
from utils import ch3_type1_chain as q, ch3_type1_tasks as s
from utils.ch3_contract import ROOT, digest, profile, validate_amd_declaration
from utils.ch3_native_recovery_records import bound, exclusive, ref, sha


@contextmanager
def attempt():
    from utils import ch3_ms_seal_recovery as ms, ch3_round2_amendment as amend
    with ExitStack() as stack:
        for module, names in ((s, ('RESULT','context','package','file')), (ms, ('RESULT','M_FILE')),
            (amend, ('RESULT',)), (q, ('CONTROL','LOG','SESSION','ENTRY','WRAPPER','PROBE_RECOVERY','QUEUE_LOCK')),
            (numeric, ('ACTIVE',)), (r, ('ACTIVE',))):
            for name in names:
                stack.enter_context(patch.object(module, name, getattr(module, name)))
        numeric.ACTIVE = r.ACTIVE = False
        r.activate()
        yield


def args(p):
    return dict(input_shape=(p['T'],p['C']), pred_len=p['pred_len'], patch=p['structure']['patch'],
        layernorm=p['structure']['layernorm'], target_idx=p['target_idx'], aux_idx=p['aux_idx'],
        norm=True, task_mode='parallel_multivariate', s2=False, thls=False)


class Declaration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with attempt(): cls.cs = q.configs()

    def test_new_ETT_24_real_declarations(self):
        count = 0
        for stage in ('M_AMEND','M_ALL'):
            for t in self.cs[stage]['tasks']:
                if t['model']=='AMD' and t['dataset'] in ('ETTh2','ETTm1','ETTm2'):
                    p=profile(self.cs[stage],t); validate_amd_declaration(p, **args(p)); count+=1
                    self.assertEqual(p['supervised_channels'], list(range(7)))
                    self.assertEqual(p['output_order'], p['features'])
        self.assertEqual(count,24)

    def test_existing_M_28_declarations_remain_valid(self):
        count=0
        for stage in ('M_BASE','M_AMEND','M_ALL'):
            for t in self.cs[stage]['tasks']:
                if t['model']=='AMD' and t['dataset'] in ('ETTh1','Weather','Exchange'):
                    p=profile(self.cs[stage],t);validate_amd_declaration(p,**args(p));count+=1
        self.assertEqual(count,28)

    def test_guard_shape_pred_patch_norm_mode_aux_target_modules_reject(self):
        c=self.cs['M_AMEND'];t=next(t for t in c['tasks']if t['model']=='AMD');p=profile(c,t)
        bad=dict(input_shape=(95,7),pred_len=97,patch=8,layernorm=False,target_idx=0,
                 aux_idx=[0],norm=False,task_mode='target_exogenous',s2=True,thls=True)
        for key,value in bad.items():
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_amd_declaration(p,**dict(args(p),**{key:value}))

    def test_unauthorized_domain_and_model_reject(self):
        c=self.cs['M_AMEND'];p=profile(c,next(t for t in c['tasks']if t['model']=='AMD'))
        for key,value in (('dataset','ECL'),('dataset','unregistered'),('model','J'),('model','DLinear')):
            bad=copy.deepcopy(p);bad[key]=value
            with self.assertRaises(ValueError):validate_amd_declaration(bad,**args(bad))

    def test_historical_MS_branch_AST_is_unchanged(self):
        old=subprocess.check_output(['git','show',r.PRODUCER+':utils/ch3_contract.py'],cwd=ROOT,text=True)
        r.amd_guard_delta(old,(ROOT/'utils/ch3_contract.py').read_text())
        functions={n.name:n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef)}
        ns={};exec(compile(ast.Module(body=[functions['validate_amd_declaration']],type_ignores=[]),'<historical MS guard>','exec'),{'CONTRACT':'ch3-target-ms-formal-v2'},ns)
        for stage in ('URBAN_SUBSET','EPF_ALL'):
            for t in self.cs[stage]['tasks']:
                if t['model']!='AMD':continue
                p=profile(self.cs[stage],t);a=dict(args(p),task_mode='target_exogenous')
                ns['validate_amd_declaration'](p,**a);validate_amd_declaration(p,**a)
                with self.assertRaises(ValueError):validate_amd_declaration(p,**dict(a,s2=True))

    def test_preflight_checks_declarations_once_without_models(self):
        with attempt(),patch('utils.ch3_contract.validate_amd_declaration',wraps=validate_amd_declaration)as guard:
            q.validate_amd_configs(q.configs());self.assertEqual(guard.call_count,52)
            q.configs();self.assertEqual(guard.call_count,52)
        self.assertNotIn('torch',sys.modules)

    def test_only_the_dataset_tuple_is_an_allowed_contract_delta(self):
        old=subprocess.check_output(['git','show',r.PRODUCER+':utils/ch3_contract.py'],cwd=ROOT,text=True)
        new=(ROOT/'utils/ch3_contract.py').read_text()
        for change in (new.replace("or s2 or thls","or s2"),new.replace("not norm or task_mode", "task_mode")):
            with self.assertRaises(ValueError):r.amd_guard_delta(old,change)


SMOKE = r'''
import json,sys,os
from pathlib import Path
from utils.ch3_native_recovery_records import bound
from utils.ch3_contract import profile,digest
from utils import ch3_moderntcn_etth1_recovery as numeric
c=bound(numeric.CONFIG_REFS['M_AMEND'])
model,dataset,horizons=json.loads(sys.argv[1])
tasks=[t for t in c['tasks']if t['model']==model and t['dataset']==dataset and t['h']in horizons]
def audit(event,values):
    if event=='open' and isinstance(values[0],(str,bytes)):
        p=os.fsdecode(values[0])
        if '/AMD/data/'in p or p.endswith(('.pt','.pth','.safetensors')):
            raise PermissionError('CPU smoke cannot read data/checkpoint: '+p)
sys.addaudithook(audit)
import torch
torch.set_num_threads(4)
def denied(*a,**k):raise AssertionError('CPU shape smoke cannot update/backward/GPU')
torch.cuda.init=denied
torch.Tensor.backward=denied
torch.optim.Adam=denied
torch.load=denied
from utils.ch3_time_marks import synthetic
from models.ch3_adapter import build,target_prediction
rows=[]
for t in tasks:
    p=profile(c,t);mark=synthetic(p,2)if p.get('time_mark')else None
    m=build(c,t);m.eval()
    x=torch.linspace(-1,1,2*p['T']*p['C']).reshape(2,p['T'],p['C'])
    with torch.no_grad():y,aux=target_prediction(m,x,p,mark)
    assert y.shape==(2,t['h'],7)and torch.isfinite(y).all()
    assert not torch.cuda.is_initialized()
    rows.append(dict(task=t['id'],profile_sha=digest(p),shape=list(y.shape),marks=list(mark.shape)if mark is not None else None))
print(json.dumps(dict(model=model,dataset=dataset,rows=rows,model_construction=len(rows),forward=len(rows),backward=0,adam=0,GPU=0,real_validation=0,test=0,small_batch_is_engineering_only=True)))
'''


class RealCPUInterface(unittest.TestCase):
    def smoke(self,model,dataset,horizons):
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT))
        result=subprocess.run([q.PYTHON,'-B','-c',SMOKE,json.dumps([model,dataset,horizons])],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
        record=dict(model=model,dataset=dataset,requested_horizons=horizons,exit=result.returncode,stdout=result.stdout,stderr=result.stderr)
        destination=os.environ.get('M6_ETT_SMOKE_EVIDENCE')
        if destination:
            with (Path(destination)/(model+'-'+dataset+'.jsonl')).open('a')as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        self.assertEqual(result.returncode,0,result.stderr)
        value=json.loads(result.stdout);self.assertEqual(len(value['rows']),len(horizons))
        return value

    def test_AMD_three_ETT_four_H_real_build_and_forward(self):
        values=[self.smoke('AMD',d,[96,192,336,720])for d in ('ETTh2','ETTm1','ETTm2')]
        self.assertEqual(sum(v['model_construction']for v in values),12)

    def test_other_six_baselines_three_ETT_real_interfaces(self):
        for model in ('DLinear','PatchTST','iTransformer','TimeMixer','ModernTCN','TimeXer'):
            for dataset in ('ETTh2','ETTm1','ETTm2'):
                with self.subTest(model=model,dataset=dataset):self.smoke(model,dataset,[96])

    def test_M_AMEND_M_ALL_build_profiles_are_identical(self):
        with attempt():cs=q.configs()
        for a in cs['M_AMEND']['tasks']:
            if a['dataset']not in ('ETTh2','ETTm1','ETTm2'):continue
            b=next(t for t in cs['M_ALL']['tasks']if all(t[k]==a[k]for k in ('model','dataset','h')))
            p,p2=profile(cs['M_AMEND'],a),profile(cs['M_ALL'],b)
            for value in (p,p2):value.pop('training')
            self.assertEqual(p,p2)


@contextmanager
def prefix_fixture():
    """Actual 84-task contract with synthetic receipts; no real observations."""
    import math
    from utils.ch3_contract import step_arithmetic
    with tempfile.TemporaryDirectory(prefix='m6-base287-ett-')as temp, attempt(), ExitStack()as stack:
        root=Path(temp);old=root/'old';new=root/'new';control=new/'queue/controller'
        cs=q.configs();c=cs['M_BASE'];data=bound(c['baseline_unified']['data_ref']);groups={};records={}
        for model in s.MODELS:
            tasks=[t for t in c['tasks']if t['model']==model];files_by_task={}
            for t in tasks:
                p=profile(c,t);a=step_arithmetic(c,t);out=old/'M_BASE'/('formal-'+model)/t['id'];out.mkdir(parents=True)
                checkpoint={}
                for name in ('best.pt','last.pt'):
                    (out/name).write_bytes(b'synthetic non-model checkpoint bytes');checkpoint[name]=ref(out/name)
                elements=a['validation_windows']*p['pred_len']*p['C'];history=[]
                for epoch in range(1,11):history.append(dict(epoch=epoch,steps=epoch*a['train_batches'],best_epoch=1,validation=dict(mse=1.,mae=1.,sse=float(elements),sae=float(elements),elements=elements)))
                (out/'history.jsonl').write_text(''.join(json.dumps(x)+'\n'for x in history))
                counts=dict(adam=a['max_optimizer_steps'],backward=a['max_optimizer_steps'],forward=10*(a['train_batches']+math.ceil(a['validation_windows']/128))+math.ceil(a['test_windows_arithmetic_only']/128))
                files=dict(checkpoint)
                files['history.jsonl']=ref(out/'history.jsonl')
                files['budget.json']=exclusive(out/'budget.json',dict(counts=counts,by_pid={'123':counts}))
                files['runtime.json']=exclusive(out/'runtime.json',dict(task=t['id'],pid=123,error=None))
                files['manifest.json']=exclusive(out/'manifest.json',dict(task=t,profile=p,identity=dict(commit=r.PRODUCER,profile_sha=digest(p),protocol_sha=digest(c),data_sha=data['data_bindings'][t['dataset']][t['id']])))
                files['result.json']=exclusive(out/'result.json',dict(id=t['id'],commit=r.PRODUCER,scientific_protocol=c['baseline_unified']['id'],profile_sha=digest(p),protocol_sha=digest(c),scheduler_updates=counts['adam'],scheduler_sha=digest(p['training']['scheduler']),metric_scope='all_channels',elements=a['test_windows_arithmetic_only']*p['pred_len']*p['C'],best_epoch=1,final_test=dict(calls=1,selected='best.pt',sha256=checkpoint['best.pt']['sha256'],epoch=1),mse=1.,mae=1.))
                files_by_task[t['id']]=files
                records[t['id']]=dict(artifacts=files,checkpoints={name:dict(size=(out/name).stat().st_size,mtime_ns=(out/name).stat().st_mtime_ns)for name in checkpoint})
            groups[model]=exclusive(old/'queue/M_BASE'/('group-'+model)/'complete.json',dict(model=model,task_ids=[t['id']for t in tasks],artifacts=files_by_task,technical_complete=True))
        from utils import ch3_type1_upstream as u
        start,_=u.records();msconfig=bound(start['config_refs']['MS']);msids=[t['id']for t in msconfig['tasks']];msgroups={}
        stack.enter_context(patch.object(u,'OLD_RESULT',old/'original'))
        for model in s.MODELS:
            artifacts={};tasks=[t for t in msconfig['tasks']if t['model']==model]
            for t in tasks:
                out=old/'original/MS'/('formal-'+model)/t['id'];p=profile(msconfig,t)
                files=dict(result=exclusive(out/'result.json',dict(id=t['id'],commit=u.BASE,profile_sha=digest(p),protocol_sha=digest(msconfig),scientific_protocol=msconfig['baseline_unified']['id'],final_test={'calls':1},mse=1.,mae=1.)),manifest=exclusive(out/'manifest.json',dict(task=t,profile=p,identity=dict(commit=u.BASE))),runtime=exclusive(out/'runtime.json',dict(task=t['id'],error=None)))
                artifacts[t['id']]={k+'.json':v for k,v in files.items()}
            msgroups[model]=exclusive(old/'MS'/('group-'+model)/'complete.json',dict(artifacts=artifacts,technical_complete=True,task_ids=[t['id']for t in tasks],model=model))
        refs={}
        refs['MS_verification']=exclusive(old/'verification.json',dict(training_commit=u.BASE,task_ids=msids,receipts=msgroups))
        refs['MS_boundary']=exclusive(old/'MS-boundary.json',dict(training_commit=u.BASE,task_ids=msids,receipts=msgroups,source_verification_ref=refs['MS_verification']))
        binding=dict(code={},environment={},hardware={},source_states={})
        refs['M_boundary']=exclusive(old/'queue/M_BASE/technical-boundary.json',dict(purpose='baseline_type1_M_BASE_boundary_v1',scope=s.ID,technical_complete=True,result_review='pending',commit=r.PRODUCER,protocol_sha=digest(c),task_ids=[t['id']for t in c['tasks']],receipts=groups,**binding))
        refs['upstream']=exclusive(old/'queue/controller/upstream-technical-boundary.json',dict(boundaries=dict(MS=refs['MS_boundary'])))
        refs['base']=exclusive(old/'queue/controller/base287-boundary.json',dict(commit=r.PRODUCER,counts=dict(MS=203,M=84,total=287),technical_complete=True,boundaries=dict(MS=refs['MS_boundary'],M=refs['M_boundary']),upstream_ref=refs['upstream']))
        refs['authorization']=exclusive(old/'authorization.json',dict(closure_commit=r.PRODUCER))
        refs['controller']=exclusive(old/'controller.json',dict(authorization=refs['authorization']))
        refs['failure']=exclusive(old/'failure.json',dict(error="RuntimeError('owned child technical failure: M_AMEND-probe')"))
        refs['probe_failure']=exclusive(old/'probe-failure.json',dict(error="RuntimeError('serial technical gate failed; no fallback')"))
        refs['failed_budget']=exclusive(old/'failed-budget.json',dict(counts=dict(adam=0,backward=0,forward=0)))
        refs['failed_process']=exclusive(old/'failed-process.json',dict(returncodes=[1],failure_kind='business'))
        failed='AMD-ETTh2-M-oc01-v3-amend1-m128-recovery1-f1-h96-s2024'
        refs['failed_runtime']=exclusive(old/'failed-runtime.json',dict(task=failed))
        refs['failed_worker_config']=exclusive(old/'failed-worker.json',dict(task=failed))
        refs['probe_permit']=exclusive(old/'probe-permit.json',dict(commit=r.PRODUCER,protocol_sha=digest(bound(numeric.CONFIG_REFS['M_AMEND']))))
        refs['M_permit']=exclusive(old/'M-permit.json',dict(commit=r.PRODUCER,protocol_sha=digest(c),**binding))
        refs['M_probe_complete']=exclusive(old/'M-complete.json',dict(execution_complete=True,protocol_sha=digest(c),plan=s.plan(c),decisions={g['id']:dict(status='Passed',coverage=g['coverage'])for g in s.probe_groups(c)}))
        (old/'worker.log').write_text('ValueError: M AMD native all-channel identity\n')
        source=dict(producer_commit=r.PRODUCER,old_result=str(old),config_refs=numeric.CONFIG_REFS,refs=refs,exit_instances=[],failed_task=failed,failed_worker_log=ref(old/'worker.log'))
        sr=exclusive(root/'source.json',source)
        evidence=dict(source_ref=sr,producer_commit=r.PRODUCER,counts=dict(MS=203,M=84,total=287),M_receipts=groups,M_tasks=records)
        er=exclusive(root/'prefix.json',evidence)
        for mod,key,value in ((r,'SOURCE_REF',sr),(r,'REUSE_REF',er),(r,'OLD_RESULT',old),(r,'RESULT',new),
            (s,'RESULT',new),(q,'CONTROL',control),(q,'LOG',root/'launcher.log'),(q,'SESSION','synthetic-ett-absent-'+str(os.getpid()))):stack.enter_context(patch.object(mod,key,value))
        from utils import ch3_ms_seal_recovery as ms,ch3_round2_amendment as amend
        stack.enter_context(patch.object(ms,'RESULT',new));stack.enter_context(patch.object(amend,'RESULT',new/'round2-amendment'))
        original_context=s.context
        stack.enter_context(patch.object(s,'context',side_effect=lambda c:dict(original_context(c),fixture=root/'fixtures'/c['baseline_unified']['stage'])))
        stack.enter_context(patch.object(ms,'SOURCE_REF',refs['MS_verification']))
        stack.enter_context(patch.object(q,'closure',return_value='f'*40))
        stack.enter_context(patch.dict(os.environ,{q.SECRET:'synthetic-secret'}))
        exclusive(control/'controller.json',dict(owner=q.owner(),scope=s.ID))
        delta=exclusive(root/'delta.json',dict(purpose='exact_base287_producer_delta_v1',producer_commit=r.PRODUCER,changes={},new_code=r.code_binding(),weather20_patience_ref=r.WEATHER_REF))
        stack.enter_context(patch.object(r,'delta_ref',return_value=delta));stack.enter_context(patch.object(r,'DELTA_REF',delta))
        yield root,old,new,source,evidence,stack


class Prefix(unittest.TestCase):
    def test_exact_metadata_prefix_without_checkpoint_reads(self):
        with prefix_fixture()as(_,_,_,_,_,_):
            real_open=Path.open
            def guarded(path,*args,**kwargs):
                if path.suffix=='.pt':raise AssertionError('prefix adoption cannot read checkpoint bytes')
                return real_open(path,*args,**kwargs)
            with patch.object(Path,'open',guarded):rows=r.verify_prefix()
            self.assertEqual(len(rows['M_boundary']['task_ids']),84)

    def mutate_source(self,source,root,stack):
        value=exclusive(root/'mutated-source.json',source);stack.enter_context(patch.object(r,'SOURCE_REF',value))

    def test_wrong_failure_or_producer_refused(self):
        for kind in ('failure','producer','permit','task'):
            with self.subTest(kind=kind),prefix_fixture()as(root,_,_,source,_,stack):
                value=copy.deepcopy(source)
                if kind=='failure':value['refs']['failure']=exclusive(root/'bad-failure.json',dict(error='different failure'))
                elif kind=='producer':value['producer_commit']='0'*40
                elif kind=='task':value['failed_task']='different-task'
                else:value['refs']['probe_permit']=exclusive(root/'bad-permit.json',dict(commit='0'*40,protocol_sha='bad'))
                self.mutate_source(value,root,stack)
                with self.assertRaises(ValueError):r.verify_source_light()

    def test_live_old_instance_refused_no_signal(self):
        with prefix_fixture()as(root,_,_,source,_,stack):
            value=copy.deepcopy(source);value['exit_instances']=[q.owner()];self.mutate_source(value,root,stack)
            with self.assertRaises(ValueError):r.verify_source_light()

    def test_missing_duplicate_wrong_version_and_tamper_refused(self):
        for kind in ('missing','duplicate','version','tamper','test'):
            with self.subTest(kind=kind),prefix_fixture()as(root,_,_,source,evidence,stack):
                value=copy.deepcopy(evidence);run=next(iter(value['M_tasks']));files=value['M_tasks'][run]['artifacts']
                if kind=='missing':value['M_tasks'].pop(run)
                elif kind=='duplicate':value['M_tasks']['duplicate']=value['M_tasks'][run]
                elif kind=='tamper':Path(files['result.json']['path']).write_text('{}\n')
                else:
                    result=bound(files['result.json']);result['commit']='0'*40 if kind=='version'else result['commit']
                    if kind=='test':result['final_test']['calls']=2
                    files['result.json']=exclusive(root/'bad-result.json',result)
                stack.enter_context(patch.object(r,'REUSE_REF',exclusive(root/'bad-prefix.json',value)))
                with self.assertRaises(ValueError):r.verify_prefix()

    def test_real_adoption_new_MAC_preserves_old_training_sources(self):
        with prefix_fixture()as(root,old,new,source,_,stack):
            result=r.adopt_prefix();self.assertEqual(set(result),{'VERIFY_IMPORT_MS203_AND_SEAL','SEAL_M128_BOUNDARY','SEAL_BASE_287_BOUNDARY'})
            m=q.validate_boundary_light(result['SEAL_M128_BOUNDARY'],'M_BASE');self.assertEqual(m['commit'],r.PRODUCER)
            self.assertEqual(m['adoption_commit'],'f'*40);self.assertEqual(m['adopted_source_ref'],source['refs']['M_boundary'])
            base=q.validate_base287_boundary_light(result['SEAL_BASE_287_BOUNDARY']);self.assertEqual(base['counts'],dict(MS=203,M=84,total=287))
            self.assertFalse((new/'probe/M_BASE').exists());self.assertFalse((new/'M_BASE').exists())
            with self.assertRaises((PermissionError,ValueError)):q.validate_boundary_light(source['refs']['M_boundary'],'M_BASE')
            with patch.dict(os.environ,{q.SECRET:'different'}),self.assertRaises(PermissionError):q.validate_boundary_light(result['SEAL_M128_BOUNDARY'],'M_BASE')

    def test_completed_base_permit_and_old_attempt_cannot_dispatch(self):
        with attempt():cs=q.configs()
        for stage in cs:
            with self.assertRaises(PermissionError):r.validate_permit_link(cs[stage],dict(execution_attempt=numeric.ATTEMPT,probe_recovery_ref=numeric.REUSE_REF),True)
        with self.assertRaises(PermissionError):r.load_seed(cs['M_BASE'],{})

    def test_unchanged_133_policies_427_tasks_and_budget(self):
        with attempt():
            cs=q.configs()
            self.assertEqual(sum(len(c['tasks'])for c in cs.values()),427)
            self.assertEqual(sum(len(c['baseline_unified']['numeric_policies'])for c in cs.values()),133)
            self.assertEqual(sum(len(cs[k]['tasks'])for k in ('M_AMEND','URBAN_SUBSET','EPF_ALL','M_ALL')),343)
            for stage,c in cs.items():
                old=bound(numeric.CONFIG_REFS[stage])
                self.assertEqual(ref(s.file(stage)),r.config_refs()[stage])
                self.assertEqual(r.weather_base(c),old)
                self.assertEqual(s.formal_budget(c),s.formal_budget(old))
                self.assertEqual(c['baseline_unified']['numeric_policies'],old['baseline_unified']['numeric_policies'])


@contextmanager
def flow_fixture(failure_stage=None):
    import ch3_runner as runner
    from utils import ch3_native_execution as native, ch3_type1_execution as execution
    with prefix_fixture()as(root,old,new,source,evidence,stack):
        calls=[];cs=q.configs();head='f'*40
        def dynamic(c,worker=False):return dict(commit=head,protocol_sha=digest(c),code=r.code_binding(),environment={},hardware={},source_states={})
        stack.enter_context(patch.object(q,'dynamic',side_effect=dynamic))
        stack.enter_context(patch('utils.ch3_native_recovery.resource_check'))
        stack.enter_context(patch('utils.ch3_m_execution.gpu_environment_reasons',return_value=[]))
        remote=stack.enter_context(patch.object(q,'verify_live_remote',return_value=head))
        template=q.start_template();a=copy.deepcopy(template)
        a.update(reviewed=True,execution_permitted=True,structure_frozen=True,m6_authorized=True,budget_authorized=True,closure_commit=head,authorization_basis='synthetic contract fixture, not a real start authorization')
        start=exclusive(root/'synthetic-start.json',a)
        audit=stack.enter_context(patch.object(native,'validate_probe_completion',return_value=None))
        # Numeric tensor work is replaced, but AUTO_AUDIT/manifest projection,
        # permit creation, runtime sealing, all boundary seals and index stay real.
        def wait(stage,permit_ref,probe,model=None,runtime_ref=None):
            calls.append(('probe'if probe else'formal',stage,model))
            if stage=='M_BASE':raise AssertionError('completed M_BASE redispatched')
            if stage==failure_stage:raise RuntimeError('synthetic identity failure; no resource fallback')
            c=cs[stage];ctx=s.context(c);a=bound(permit_ref);q.validate_permit(c,a,probe)
            if probe:
                rootp=ctx['probe_root'];approval=exclusive(rootp/'approval.json',a)
                payload=exclusive(rootp/'synthetic-payload.json',dict(stage=stage,synthetic=True))
                report=dict(approval=approval,artifacts={payload['path']:payload},evidence={},decisions={g['id']:dict(status='Passed',concurrency=g['planned_q'],coverage=g['coverage'],representatives=g['representatives'])for g in s.probe_groups(c)},budget=dict(caps=ctx['caps'],reserved=dict(adam=0,forward=0,backward=0),actual=dict(adam=0,forward=0,backward=0),refund=False),probe_recovery_ref=r.REUSE_REF,**dynamic(c))
                exclusive(rootp/'complete.json',report)
            else:
                q.validate_runtime(c,runtime_ref,permit_ref)
                tasks=[t for t in c['tasks']if t['model']==model];artifacts={}
                # Real make_config/worker validation on the new M_AMEND path.
                if stage=='M_AMEND'and model=='AMD':
                    t=tasks[0];out=ctx['result_root']/('formal-'+model)/t['id'];ctx['fixture'].mkdir(parents=True,exist_ok=True)
                    # The actual group-child branch injects this worker-only
                    # tool directory before make_config; parent preflight does not.
                    with patch.object(sys,'path',[str(ROOT/'tools/restricted_regression'),*sys.path]),patch.dict(os.environ,{'TMPDIR':str(ctx['fixture'])}):
                        cfg=execution.make_config(c,'ch3_formal',out,task=t['id'],approval=a,runtime_ref=runtime_ref)
                        execution.validate_worker(c,cfg)
                for t in tasks:
                    out=ctx['result_root']/('formal-'+model)/t['id'];p=profile(c,t)
                    result=exclusive(out/'result.json',dict(id=t['id'],commit=head,profile_sha=digest(p),protocol_sha=digest(c),scientific_protocol=c['baseline_unified']['id'],final_test={'calls':1},mse=9.,mae=9.))
                    manifest=exclusive(out/'manifest.json',dict(task=t,profile=p,identity=dict(commit=head)))
                    runtime=exclusive(out/'runtime.json',dict(task=t['id'],error=None))
                    artifacts[t['id']]={'result.json':result,'manifest.json':manifest,'runtime.json':runtime}
                exclusive(ctx['control']/('group-'+model)/'complete.json',dict(technical_complete=True,model=model,task_ids=[t['id']for t in tasks],artifacts=artifacts))
            return dict(exit_code=0)
        stack.enter_context(patch.object(q,'wait_owned',side_effect=wait))
        for name in ('init_training','update','evaluate'):
            stack.enter_context(patch.object(runner,name,side_effect=AssertionError('synthetic flow cannot train/evaluate')))
        yield root,new,start,calls,audit,remote,stack


class RecoveryFlow(unittest.TestCase):
    def test_actual_run_skips_completed_prefix_and_seals_mixed_371(self):
        from utils import ch3_round2_amendment as amend
        with flow_fixture()as(root,new,start,calls,audit,remote,stack):
            q.run(start)
            self.assertEqual([stage for kind,stage,_ in calls if kind=='probe'],['M_AMEND','URBAN_SUBSET','EPF_ALL','M_ALL'])
            self.assertEqual(sum(len(q.configs()[stage]['tasks'])for kind,stage,model in calls if kind=='formal'and model=='AMD'),343)
            self.assertEqual(audit.call_count,4);remote.assert_not_called()
            self.assertFalse((new/'probe/M_BASE').exists());self.assertFalse((new/'M_BASE').exists())
            v=amend.summary();self.assertEqual((len(v['cells']),v['planned_formal_executions']),(371,399))
            self.assertTrue(all(x['execution_commit']==r.PRODUCER for x in v['cells']if x['origin']=='base_m128'))
            weather=[x for x in v['cells']if x['dataset']=='Weather'];self.assertEqual(len(weather),28)
            self.assertTrue(all(x['mse']==9. and x['origin']=='amend1'and x['supersedes']for x in weather))
            complete=bound(ref(q.CONTROL/'complete.json'))
            self.assertEqual((complete['total_runs'],complete['adopted_new_formal_runs'],complete['executed_new_formal_runs']),(427,84,343))
            # Final complete verification runs after the owner is gone, without
            # using the old lifecycle MAC as a new runtime authorization.
            import m6_type1_followup_entry as entry
            with patch.object(q,'status',return_value=dict(running=False,STOP=False,failure=False,complete=True)),patch.dict(os.environ,{q.SECRET:''}),patch.object(sys,'argv',['entry','complete']),redirect_stdout(io.StringIO()):
                self.assertEqual(entry.cli(),0)
            audit_count=audit.call_count;q.validate_start(bound(start));self.assertEqual(audit.call_count,audit_count)

    def test_stage_failure_stops_every_later_dispatch(self):
        with flow_fixture('M_AMEND')as(_,new,start,calls,audit,_,_):
            with self.assertRaisesRegex(RuntimeError,'identity failure'):q.run(start)
            self.assertEqual(calls,[('probe','M_AMEND',None)]);audit.assert_not_called()
            self.assertFalse((q.CONTROL/'complete.json').exists())

    def test_public_preflight_reaches_real_status_read_only(self):
        import m6_type1_followup_entry as entry
        with flow_fixture()as(root,new,start,_,audit,remote,stack):
            # preflight uses an absent new root; prefix_fixture's controller is
            # only a synthetic adoption fixture, not a real attempt.
            import shutil
            shutil.rmtree(new)
            guards=[stack.enter_context(patch.object(r,name,side_effect=AssertionError('preflight cannot '+name)))for name in ('adopt_prefix','verify_prefix')]
            declaration=stack.enter_context(patch('utils.ch3_contract.validate_amd_declaration',wraps=validate_amd_declaration))
            output=io.StringIO()
            with patch.object(sys,'argv',['entry','preflight','--approval',start['path'],'--approval-sha',start['sha256']]),redirect_stdout(output):code=entry.cli()
            v=json.loads(output.getvalue());self.assertEqual((code,v['blocked'],v['remaining_formal_runs']),(0,[],343))
            self.assertTrue(v['READY_TO_ARM_HANDOFF']);self.assertFalse(v['READY_FOR_GPU_EXECUTION'])
            self.assertEqual(remote.call_count,1);self.assertEqual(declaration.call_count,52)
            self.assertFalse(new.exists());audit.assert_not_called()
            for guard in guards:guard.assert_not_called()

    def test_old_commit_or_attempt_authorization_refused(self):
        with flow_fixture()as(root,_,start,_,_,_,_):
            current=bound(start)
            for key in ('commit','attempt','source'):
                bad=copy.deepcopy(current)
                if key=='commit':bad['closure_commit']=r.PRODUCER
                elif key=='attempt':bad['upstream_anchors']['execution_attempt']=numeric.ATTEMPT
                else:bad['upstream_anchors']['source_anchors_ref']['sha256']='0'*64
                with self.assertRaises(PermissionError):q.validate_start(bad)

    def test_public_preflight_blocks_bad_bundle_without_materialization(self):
        import shutil
        import m6_type1_followup_entry as entry
        with flow_fixture()as(_,new,start,_,audit,remote,stack):
            shutil.rmtree(new)
            stack.enter_context(patch('tools.restricted_regression.run_restricted.verify_bundle',side_effect=RuntimeError('synthetic bundle seal mismatch')))
            output=io.StringIO()
            with patch.object(sys,'argv',['entry','preflight','--approval',start['path'],'--approval-sha',start['sha256']]),redirect_stdout(output):code=entry.cli()
            v=json.loads(output.getvalue());self.assertEqual(code,2)
            self.assertTrue(any('bundle seal mismatch'in x for x in v['blocked']))
            self.assertFalse(v['READY_TO_ARM_HANDOFF']);self.assertFalse(new.exists())
            audit.assert_not_called();remote.assert_not_called()

    def test_old_output_conflict_is_not_overwritten(self):
        with flow_fixture()as(_,new,start,_,_,_,_):
            reasons=q.readiness(bound(start))
            self.assertTrue(any('fresh repeat forbidden'in x for x in reasons))
            self.assertTrue(new.exists())


class OwnedLifecycle(unittest.TestCase):
    def test_new_entry_synthetic_tmux_owned_stop(self):
        import shlex
        import time
        with tempfile.TemporaryDirectory(prefix='m6-ett-owned-')as tmp,attempt():
            root=Path(tmp);pidfile=root/'pid';log=root/'log'
            session='fixture-amend-ett-'+str(os.getpid())+'-'+str(time.time_ns())
            code='import os,time;from pathlib import Path;Path('+repr(str(pidfile))+').write_text(str(os.getpid()));time.sleep(30)'
            command=shlex.join([q.PYTHON,'-B','-c',code,str(r.ENTRY),'start'])+' >'+shlex.quote(str(log))+' 2>&1'
            subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),command],check=True)
            owner=None
            try:
                for _ in range(100):
                    if pidfile.exists():break
                    time.sleep(.02)
                self.assertTrue(pidfile.exists());owner=q.owner(int(pidfile.read_text()))
                self.assertTrue(q.same(owner));exclusive(root/'controller.json',dict(scope=s.ID,owner=owner))
                with patch.object(q,'CONTROL',root):value=q.safe_stop()
                self.assertTrue(value['supervisor_signal_sent']);self.assertFalse(value['old_chain_signal_sent'])
                for _ in range(100):
                    if not q.same(owner):break
                    time.sleep(.02)
                self.assertFalse(q.same(owner))
                for _ in range(100):
                    if subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
                    time.sleep(.02)
                self.assertNotEqual(subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode,0)
            finally:
                if owner and q.same(owner):q.signal_owned(owner)

    def test_foreign_scope_is_not_signalled(self):
        with tempfile.TemporaryDirectory(prefix='m6-ett-foreign-')as tmp,attempt(),patch.object(q,'CONTROL',Path(tmp)),patch.object(q,'signal_owned')as signal:
            exclusive(Path(tmp)/'controller.json',dict(scope='unrelated-frozen-v3',owner=dict(pid=1,start_ticks=1)))
            with self.assertRaises(PermissionError):q.safe_stop()
            signal.assert_not_called()


if __name__=='__main__':
    unittest.main()
