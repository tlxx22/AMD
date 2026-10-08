"""Real guard installation and worker entry up to the no-model CPU boundary."""
import contextlib
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
TOOL=ROOT/'tools/restricted_regression'


class CPUProbeBoundaryReached(Exception):
    pass


def worker_fixture(case,root):
    """Only machine/lifecycle inputs and the absent compute producer are fixtures."""
    class NoModelImports(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in ('torch','models','tensorflow','cupy'):raise AssertionError('real model import forbidden: '+fullname)
    sys.meta_path.insert(0,NoModelImports())
    sys.path.insert(0,str(TOOL))
    from tools.restricted_regression import m5_formal_entry as tool
    from tools.restricted_regression import run_restricted,resource_budget
    sys.modules.update(m5_formal_entry=tool,run_restricted=run_restricted,resource_budget=resource_budget)
    import restricted_io_guard as guard
    from utils import ch3_patchtst_depth_urban6_recovery as r
    r.activate()
    from utils import ch3_event_resources as event
    from utils import ch3_type1_chain as q,ch3_type1_tasks as scope,ch3_type1_execution as execution,ch3_round2_amendment as amend
    from utils.ch3_native_recovery_records import exclusive,ref,bound
    from utils.ch3_contract import digest,task_by_id
    import ch3_runner as runner
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    c=q.configs()['PATCH_ENC1'];task=c['tasks'][0];previous=scope.context
    context=dict(previous(c),probe_root=root/'probe/PATCH_ENC1',control=root/'queue/PATCH_ENC1',fixture=root/'fixtures',result_root=root/'formal')
    context['fixture'].mkdir()
    calls=dict(validate_config=0,stage_validation=0,all_stage_after_install=0,init_training=0,binding=0,synthetic_card_samples=0)
    real_lock=runner.GPULock;real_query=subprocess.check_output
    def fixture_lock(cc):
        cc=copy.deepcopy(cc);cc['baseline_unified']['project_lock']=str(root/'CPU-fixture.lock')
        return real_lock(cc)
    def card_sample(pids,query_timeout=10.,resource_mode=None):
        assert pids==[] and resource_mode=='exclusive_gpu_whole_card_v1' and 0<query_timeout<=10
        calls['synthetic_card_samples']+=1
        return dict(time=1.,uuid='GPU-CPU-fixture',device='cuda:0',total=24*1024**3,used=1024**3,free=23*1024**3,driver_reserved=0,resource_mode=resource_mode,query_timeout=query_timeout,query_elapsed=0.)
    def no_GPU_query(args,*a,**kw):
        if args and args[0]=='nvidia-smi':raise AssertionError('CPU worker binding fixture cannot query real GPU')
        return real_query(args,*a,**kw)
    with contextlib.ExitStack()as stack:
        stack.enter_context(patch.object(runner,'GPULock',side_effect=fixture_lock))
        stack.enter_context(patch.object(tool,'gpu_sample',side_effect=card_sample))
        stack.enter_context(patch.object(subprocess,'check_output',side_effect=no_GPU_query))
        stack.enter_context(patch.object(scope,'RESULT',root))
        stack.enter_context(patch.object(amend,'RESULT',root/'round2-amendment'))
        stack.enter_context(patch.object(scope,'context',side_effect=lambda cc:context if cc==c else previous(cc)))
        stack.enter_context(patch.object(q,'CONTROL',root/'queue/controller'))
        stack.enter_context(patch.object(q,'closure',return_value='f'*40))
        original_git=runner.git
        stack.enter_context(patch.object(runner,'git',side_effect=lambda *args:'f'*40 if args==('rev-parse','HEAD')else original_git(*args)))
        stack.enter_context(patch.dict(os.environ,{q.SECRET:'CPU-fixture-not-a-launch-grant','TMPDIR':str(context['fixture'])}))
        stack.enter_context(patch('utils.ch3_m_execution.gpu_environment_reasons',return_value=[]))
        from tests.test_m6_event_driven_resources import startup_query
        start=q.start_template();start.update(reviewed=True,execution_permitted=True,structure_frozen=True,m6_authorized=True,budget_authorized=True,closure_commit='f'*40,authorization_basis='CPU-only synthetic fixture; actual HEAD differs and cannot authorize real computation')
        start_ref=exclusive(root/'synthetic-auth.json',start)
        with patch.object(scope,'RESULT',root/'not-started'),patch.object(subprocess,'check_output',side_effect=startup_query),event.prestart(start):pass
        exclusive(q.CONTROL/'controller.json',dict(scope=scope.ID,owner=q.owner(),authorization=start_ref))
        event.seal_startup(start_ref)
        adopted=r.adopt_prefix()
        permit_ref=q.create_permit(c,start_ref,True,boundary_ref={k:adopted[q.STAGE_STATES[k][3]]for k in r.COMPLETED_STAGES},round2_ref=adopted['SEAL_ROUND2_REVISED_BOUNDARY'])
        out=context['probe_root']/r.groups(c)[0]['id']/'serial'/task['id']
        cfg=execution.make_config(c,'ch3_probe',out,task=task['id'],approval=bound(permit_ref))
        denied_path=bound(r.WORKER_FAILURE_REF)['old_evidence_denied_path']
        assert denied_path not in cfg['metadata_files']
        assert cfg['metadata_files'][r.RESOURCE_CONTRACT_REF['path']]==r.RESOURCE_CONTRACT_REF['sha256']
        config_path=out/'config.json';config_sha=ref(config_path)['sha256']
        os.environ.update(AMD_RR_CONFIG=str(config_path),AMD_RR_CONFIG_SHA256=config_sha)
        real_validate=tool.validate_config;real_stage=scope.validate
        def validate(s):
            calls['validate_config']+=1
            return real_validate(s)
        def stage(s):
            calls['stage_validation']+=1
            return real_stage(s)
        stack.enter_context(patch.object(tool,'validate_config',side_effect=validate))
        stack.enter_context(patch.object(scope,'validate',side_effect=stage))
        guard.install(str(config_path),config_sha)
        state=guard.require_installed();before_stage=calls['stage_validation']
        assert calls['validate_config']==1 and before_stage>0
        def all_stages():
            calls['all_stage_after_install']+=1
            raise AssertionError('restricted worker must not revalidate all stages')
        real_binding=runner.probe_resource_binding
        def binding(cc,tt=None):
            calls['binding']+=1
            return real_binding(cc,tt)
        def no_model(*args,**kwargs):
            calls['init_training']+=1
            raise CPUProbeBoundaryReached('binding reached; real model and six-step production deliberately not executed')
        stack.enter_context(patch.object(runner,'probe_resource_binding',side_effect=binding))
        stack.enter_context(patch.object(runner,'init_training',side_effect=no_model))
        torch_stub=types.ModuleType('torch');torch_stub.CPU_binding_fixture=True
        stack.enter_context(patch.dict(sys.modules,{'torch':torch_stub}))
        if case=='good':
            # Execute the real bootstrap and budget sampler with no compute API.
            def no_compute(*args,**kwargs):raise AssertionError('fixture must not compute or query CUDA')
            class Module:
                __call__=no_compute
            class Adam:
                step=no_compute
            fractions=[]
            torch_stub.set_num_threads=lambda n:None
            torch_stub.nn=types.SimpleNamespace(Module=Module)
            torch_stub.optim=types.SimpleNamespace(Adam=Adam)
            torch_stub.autograd=types.SimpleNamespace(backward=no_compute)
            torch_stub.cuda=types.SimpleNamespace(mem_get_info=no_compute,is_initialized=no_compute,max_memory_reserved=no_compute,set_per_process_memory_fraction=lambda f,d:fractions.append((f,d)))
            tool.bootstrap(state)
            resource_budget.INSTANCE.sample(torch_stub)
            startup=event.validate_startup(state['startup_hardware_ref'],state['approval'],live=True)['sample']
            assert fractions==[(min(.95,(startup['free']-max(8*1024**3,.1*startup['total']))/startup['total']),0)]
            assert json.loads(Path(state['budget_file']).read_text())['cuda_reserved_peak']is None
        if case=='original-bug':
            # Execute precisely the reviewed failing helper under the real hook.
            raw=subprocess.check_output(['git','-C',str(ROOT),'show','805656e9da7d780330cddbf79995f234deac90d6:ch3_runner.py'])
            import ast
            node=next(x for x in ast.parse(raw).body if isinstance(x,ast.FunctionDef)and x.name=='probe_resource_binding')
            namespace=dict(runner.__dict__)
            exec(compile(ast.Module(body=[node],type_ignores=[]),'original-r4-helper','exec'),namespace)
            try:namespace['probe_resource_binding'](c)
            except guard.ForbiddenAccess as exc:assert denied_path in str(exc)
            else:raise AssertionError('original r4 helper must reproduce real guard rejection')
            assert calls['stage_validation']>before_stage
            before_stage=calls['stage_validation']
        stack.enter_context(patch.object(q,'configs',side_effect=all_stages))
        target=copy.deepcopy(c)
        if case.startswith('worker-'):
            field,value={'worker-stage':('unified_stage','M_ALL'),'worker-protocol':('protocol_sha','0'*64),'worker-type1':('type1_scope','other'),'worker-scope':('successor_scope','other'),'worker-task':('task','foreign'),'worker-mode':('resource_mode','arbitrary'),'worker-contract':('resource_contract_ref',dict(path='foreign',sha256='0'*64))}[case]
            state[field]=value
        elif case.startswith('permit-'):
            field,value={'permit-protocol':('protocol_sha','0'*64),'permit-type1':('type1_scope','other'),'permit-scope':('successor_scope','other'),'permit-task':('authorized_task_ids',[]),'permit-mode':('resource_mode','arbitrary'),'permit-contract':('resource_contract_ref',dict(path='foreign',sha256='0'*64))}[case]
            state['approval'][field]=value
        elif case=='purpose':state['purpose']='ch3_formal'
        elif case=='config-argument':target['tasks']=target['tasks'][1:]
        elif case=='unbound-mode':q.PROBE_RECOVERY=None
        elif case=='handshake':os.environ['AMD_RR_CONFIG_SHA256']='0'*64
        elif case=='handshake-missing':os.environ.pop('AMD_RR_CONFIG',None);os.environ.pop('AMD_RR_CONFIG_SHA256',None)
        if case=='unvalidated-copy':
            try:execution.read_config(copy.deepcopy(state))
            except PermissionError:pass
            else:raise AssertionError('a config copy must not inherit the installed grant')
        elif case in ('good','original-bug'):
            try:tool.worker()
            except CPUProbeBoundaryReached:pass
            else:raise AssertionError('real worker entry did not reach the binding boundary')
            assert calls['binding']==1 and calls['init_training']==1
            assert real_binding(c,task)==r.resource_binding()
            assert json.loads((out/'runtime.json').read_text())['task']==task['id']
        else:
            try:real_binding(target,task)
            except (PermissionError,ValueError,RuntimeError):pass
            else:raise AssertionError('bad installed binding was accepted: '+case)
            assert calls['init_training']==0
        assert calls['all_stage_after_install']==0 and calls['stage_validation']==before_stage and calls['synthetic_card_samples']==0
        if case=='good':
            try:Path(denied_path).read_bytes()
            except guard.ForbiddenAccess:pass
            else:raise AssertionError('the original old-evidence access guard must still reject')
        audit=[json.loads(l)for l in Path(cfg['audit_log']).read_text().splitlines()]
        denied=[x for x in audit if x['event']=='denied']
        if case in ('good','original-bug'):assert len(denied)==1 and denied[0]['path']==denied_path
        else:assert not denied
        assert getattr(sys.modules['torch'],'CPU_binding_fixture',False)
        assert not any(x=='models' or x.startswith('models.')for x in sys.modules)
        report=dict(case=case,passed=True,calls=calls,real_guard_installed=True,config=ref(config_path),audit=ref(Path(cfg['audit_log'])),unauthorized_old_path=denied_path,old_path_in_metadata=False,real_model=0,GPU=0,forward=0,backward=0,Adam=0,validation_test=0,checkpoint_load=0,stopped_before_real_init=True)
        exclusive(out/'binding-fixture-result.json',report)
        print(json.dumps(report,ensure_ascii=False))


class WorkerGuardBinding(unittest.TestCase):
    def run_fixture(self,case):
        base=Path(os.environ['M6_GUARD_FIXTURE_ROOT'])
        out=base/case
        result=subprocess.run([sys.executable,'-B','-m','tests.test_m6_worker_guard_binding','--worker-fixture',case,str(out)],cwd=ROOT,capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''},timeout=120)
        base.mkdir(parents=True,exist_ok=True)
        (base/(case+'.stdout.txt')).write_text(result.stdout)
        (base/(case+'.stderr.txt')).write_text(result.stderr)
        (base/(case+'.exit.json')).write_text(json.dumps(dict(exit_code=result.returncode))+'\n')
        self.assertEqual(result.returncode,0,result.stdout[-1000:]+result.stderr[-4000:])

    def run_fixtures(self,cases):
        # Each installed audit hook lives in its own CPU-only process/root.
        with ThreadPoolExecutor(max_workers=4)as pool:
            runs=[(case,pool.submit(self.run_fixture,case))for case in cases]
            for case,run in runs:
                with self.subTest(case=case):run.result()

    def test_real_worker_and_install_no_revalidation_and_guard_still_denies(self):
        self.run_fixture('good')

    def test_original_r4_failure_reproduced_by_real_guard_then_fixed_path_reached(self):
        self.run_fixture('original-bug')

    def test_installed_task_stage_protocol_permit_and_resource_mismatches(self):
        self.run_fixtures(('worker-stage','worker-protocol','worker-type1','worker-scope','worker-task','worker-mode','worker-contract','permit-protocol','permit-type1','permit-scope','permit-task','permit-mode','permit-contract','purpose','config-argument','unbound-mode'))

    def test_environment_handshake_and_unvalidated_copy_cannot_reuse_grant(self):
        self.run_fixtures(('handshake','handshake-missing','unvalidated-copy'))


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker-fixture':worker_fixture(sys.argv[2],sys.argv[3])
    else:unittest.main()
