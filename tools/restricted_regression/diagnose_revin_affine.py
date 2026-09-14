"""Bounded paired RevIN derivative observation; no production or policy changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

REVISION = 'revin-shared-affine-independent-reference-v1'
CARRIED = dict(forward=114, backward=30, adam=0)
PRIOR_SHA = '46f9afd6e186852e8b4e708f2eabca67f6478430d747d1b0c362944bae423d8c'


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    from diagnose_closed_equivalence import json_compatible
    with Path(path).open('x') as f:
        json.dump(json_compatible(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def worker(config):
    from restricted_io_guard import require_installed
    from current_policy import require_scope
    from acceptance_driver import verify_seal
    from resource_budget import INSTANCE, attach_analysis_apis
    import torch
    import numpy as np
    import main as runner
    from models.tsAMD import AMD
    from models.tsAMD_enhanced import AMDEnhanced
    from test_tsAMD_enhanced import AMDEnhancedTHLSTests, _closed_gradient_numerics

    state = require_installed(); require_scope(state, 'new_cuda')
    assert config['revision'] == REVISION and config['access_policy'] == 'synthetic_regression'
    verify_seal(read(config['seal_file'])); attach_analysis_apis()
    torch.set_num_threads(1)
    session = Path(config['session_root'])
    handles = []; traces = {}; grads = {}; layouts = {}
    def layout(t):
        return dict(shape=list(t.shape), stride=list(t.stride()), storage_offset=t.storage_offset(),
                    contiguous=t.is_contiguous(), dtype=str(t.dtype), device=str(t.device),
                    requires_grad=t.requires_grad, storage_ptr=t.untyped_storage().data_ptr())
    def capture(key, t, gradient=False):
        # Read layout before cloning; persist the immediate immutable observation.
        layouts[key] = layout(t); snap=t.detach().clone()
        (grads if gradient else traces)[key] = snap
        write_new(session/(key+'.json'), dict(layout=layouts[key], values=snap.tolist(),
                  finite=bool(torch.isfinite(snap).all()), nan=int(torch.isnan(snap).sum()),
                  inf=int(torch.isinf(snap).sum())))
        if not torch.isfinite(snap).all():
            raise RuntimeError('nonfinite snapshot: '+key)
    def same(a,b):
        if torch.is_tensor(a): return torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
        if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        return a==b
    def vector(a,b):
        assert a.shape==b.shape and a.dtype==b.dtype
        x,y=a.detach().double(),b.detach().double();d=(x-y).abs();scale=torch.maximum(x.abs(),y.abs())
        rel=[float(v/s) if s!=0 else None for v,s in zip(d.reshape(-1),scale.reshape(-1))]
        def ordered(t):
            bits=t.detach().cpu().contiguous().view(torch.int32).to(torch.int64) & 0xffffffff
            return torch.where((bits & 0x80000000) != 0, 0x80000000-(bits & 0x7fffffff), 0x80000000+bits)
        ulp=(ordered(a)-ordered(b)).abs()
        return dict(left=x.tolist(),right=y.tolist(),abs_diff=d.tolist(),relative_diff=rel,
                    max_abs=float(d.max()),max_rel=max((v for v in rel if v is not None),default=None),
                    ulp=ulp.tolist(),max_ulp=int(ulp.max()),different_indices=(a!=b).nonzero().tolist(),
                    bitwise=torch.equal(a,b),finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))
    def observe(name,model):
        def node_record(kind):
            def hook(grad_inputs,grad_outputs):
                for i,t in enumerate(grad_inputs):
                    if t is not None:capture(name+'_'+kind+'_gin'+str(i),t,True)
                for i,t in enumerate(grad_outputs):
                    if t is not None:capture(name+'_'+kind+'_gout'+str(i),t,True)
                return None
            return hook
        def revin(module,args,out):
            mode=args[1];capture(name+'_'+mode+'_input',args[0]);capture(name+'_'+mode+'_output',out)
            def gout(g):capture(name+'_'+mode+'_cotangent',g,True);return None
            handles.append(out.register_hook(gout))
            if mode=='norm':
                capture(name+'_mean',module.mean);capture(name+'_stdev',module.stdev)
                node=out.grad_fn.next_functions[0][0]
                assert node.name().startswith('MulBackward')
                capture(name+'_norm_pre_affine',node._saved_self)
                capture(name+'_norm_weight_saved',node._saved_other)
                handles.append(node.register_hook(node_record('norm_mul')))
            else:
                mul=out.grad_fn.next_functions[0][0];node=mul.next_functions[0][0]
                assert mul.name().startswith('MulBackward') and node.name().startswith('DivBackward')
                capture(name+'_denorm_numerator',node._saved_self)
                capture(name+'_denorm_divisor',node._saved_other)
                handles.append(node.register_hook(node_record('denorm_div')))
                write_new(session/(name+'-revin-contract.json'), dict(target_slice=repr(args[2]),eps=module.eps,
                    affine=module.affine,training=module.training,mean_detached=not module.mean.requires_grad,
                    stdev_detached=not module.stdev.requires_grad,weight_ptr=module.affine_weight.data_ptr(),
                    norm_saved_weight_alias=layouts[name+'_norm_weight_saved']['storage_ptr']==module.affine_weight.untyped_storage().data_ptr()))
            return None
        handles.append(model.rev_norm.register_forward_hook(revin))
    stopped=threading.Event(); monitor_errors=[]
    def alarm(signum,frame):raise TimeoutError('RevIN diagnostic time/resource limit')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(180)
    def monitor():
        while not stopped.wait(.5):
            try:
                INSTANCE.sample(torch)
                if sum(p.stat().st_size for p in session.rglob('*') if p.is_file())>config['limits']['output']:
                    raise RuntimeError('diagnostic output limit')
            except BaseException as exc:
                monitor_errors.append(str(exc));INSTANCE.stop(exc);os.kill(os.getpid(),signal.SIGALRM);return
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    result=dict(revision=REVISION,status='running',acceptance_tests_executed=0)
    try:
        INSTANCE.current('diagnostic/revin_cpu_T512_pair')
        common,options=AMDEnhancedTHLSTests._kwargs(512)
        runner.set_seed(2024);a=AMDEnhanced(**common,**options).eval()
        runner.set_seed(2024);b=AMD(**common,target_slice=slice(6,7)).eval()
        assert same(a.state_dict(),b.state_dict())
        initial={k:v.detach().clone() for k,v in a.state_dict().items()}
        assert all(p.grad is None for m in (a,b) for p in m.parameters())
        x=torch.randn(2,512,7,requires_grad=True);xb=x.detach().clone().requires_grad_()
        capture('input_A',x);capture('input_AMD',xb)
        observe('A',a);observe('AMD',b)
        rng=runner.capture_rng_state();before=dict(read(config['budget_file'])['counts'])
        pa,aa=a(x);pb,ab=b(xb)
        la=pa.square().mean()+aa;lb=pb.square().mean()+ab
        write_new(session/'forward-alignment.json',dict(prediction=torch.equal(pa,pb),aux=torch.equal(aa,ab),loss=torch.equal(la,lb),
            prediction_values=pa.detach().tolist(),aux_values=[float(aa),float(ab)],loss_values=[float(la),float(lb)],
            rng_equal=same(rng,runner.capture_rng_state()),same_state=same(a.state_dict(),b.state_dict())))
        assert torch.equal(pa,pb) and torch.equal(aa,ab) and torch.equal(la,lb)
        assert same(rng,runner.capture_rng_state())
        la.backward();lb.backward()
        params={k:_closed_gradient_numerics(p.grad,dict(b.named_parameters())[k].grad) for k,p in a.named_parameters()}
        write_new(session/'raw-gradients-before-comparison.json',dict(
            left_affine_weight=a.rev_norm.affine_weight.grad.detach().clone().tolist(),
            right_affine_weight=b.rev_norm.affine_weight.grad.detach().clone().tolist(),
            parameters=params,input_gradient=_closed_gradient_numerics(x.grad,xb.grad)))
        comparison=vector(a.rev_norm.affine_weight.grad,b.rev_norm.affine_weight.grad)
        pair=dict(weight_gradients=comparison,parameter_gradients=params,input_gradient=_closed_gradient_numerics(x.grad,xb.grad),
                  state_unchanged=same(initial,a.state_dict()) and same(initial,b.state_dict()),rng_unchanged=same(rng,runner.capture_rng_state()),
                  mode_eval=all(not m.training for model in (a,b) for m in model.modules()),counter_difference={k:read(config['budget_file'])['counts'][k]-before[k] for k in CARRIED})
        write_new(session/'paired-gradients.json',pair)
        assert comparison['finite'] and all(v['finite'] for v in params.values()) and pair['input_gradient']['finite']
        assert pair['state_unchanged'] and pair['rng_unchanged'] and pair['mode_eval']
        assert pair['input_gradient']['bitwise_equal']
        assert all(v['bitwise_equal'] for k,v in params.items() if k!='rev_norm.affine_weight')
        expected_equal=['norm_input','norm_output','norm_pre_affine','mean','stdev','denorm_input']
        assert all(torch.equal(traces['A_'+k],traces['AMD_'+k]) for k in expected_equal)
        assert torch.equal(grads['A_norm_cotangent'],grads['AMD_norm_cotangent'])
        assert torch.equal(grads['A_denorm_cotangent'],grads['AMD_denorm_cotangent'])
        assert int(torch.count_nonzero(grads['A_denorm_cotangent'][:,:,:6]))==0
        references={}
        for name,model,mapping in [('A',a,list(range(7))),('AMD',b,[6]*7)]:
            INSTANCE.charge('forward','independent_RevIN_shared_affine_reference')
            q=traces[name+'_norm_pre_affine'].double();gz=grads[name+'_norm_cotangent'].double()
            norm=(q*gz).sum(dim=(0,1))
            p=traces[name+'_denorm_input'].double();go=grads[name+'_denorm_cotangent'].double()
            std=traces[name+'_stdev'].double();w=model.rev_norm.affine_weight.detach().double();bias=model.rev_norm.affine_bias.detach().double()
            eps=model.rev_norm.eps
            den=torch.zeros(7,dtype=torch.float64)
            for c,j in enumerate(mapping):
                den[j]+=(-go[:,:,c]*std[:,:,j]*(p[:,:,c]-bias[j])/(w[j]+eps*eps).square()).sum()
            reference=norm+den;actual=model.rev_norm.affine_weight.grad.detach().double()
            direct=grads[name+'_denorm_div_gin1'];expanded=torch.zeros(7)
            if name=='A':expanded.copy_(direct)
            else:expanded[6]=direct[0]
            norm_actual=grads[name+'_norm_mul_gin1']
            sum32=norm_actual+expanded
            refs=dict(norm_reference64=norm.tolist(),denorm_reference64=den.tolist(),total_reference64=reference.tolist(),
                      norm_actual32=norm_actual.tolist(),denorm_actual32=expanded.tolist(),actual_total32=actual.tolist(),
                      norm_plus_denorm32=sum32.tolist(),actual_equals_component_sum=torch.equal(sum32,model.rev_norm.affine_weight.grad),
                      max_abs_to_reference=float((actual-reference).abs().max()),
                      symmetric_reference_pass=bool(((actual-reference).abs()<=1e-7+1e-6*torch.maximum(actual.abs(),reference.abs())).all()),
                      norm_max_abs_to_reference=float((norm_actual.double()-norm).abs().max()),
                      denorm_max_abs_to_reference=float((expanded.double()-den).abs().max()),
                      eps=eps,statistics_detached=True,denorm_mapping=mapping)
            write_new(session/(name+'-independent-reference.json'),refs);references[name]=refs
            assert refs['symmetric_reference_pass'] and refs['actual_equals_component_sum']
        norm_cmp=vector(grads['A_norm_mul_gin1'],grads['AMD_norm_mul_gin1'])
        denA=grads['A_denorm_div_gin1'];denB=torch.zeros_like(denA);denB[6]=grads['AMD_denorm_div_gin1'][0]
        den_cmp=vector(denA,denB)
        totalA=torch.tensor(references['A']['total_reference64'],dtype=torch.float64)
        totalB=torch.tensor(references['AMD']['total_reference64'],dtype=torch.float64)
        result.update(status='reference_conditions_satisfied',pair=pair,references=references,norm_contribution_comparison=norm_cmp,
            denorm_contribution_comparison=den_cmp,independent_references_max_abs=float((totalA-totalB).abs().max()),
            shared_norm_input_and_cotangent_equal=True,denorm_model_input_and_cotangent_equal=True,
            target_only_denorm_cotangent=True,
            cause_scope='Denorm DivBackward0 broadcast-divisor reduction: A returns7 channel gradients; AMD returns1 target-scalar gradient. Norm contribution matches; exact kernel accumulation instruction order not traced.',
            formula='sum_BT(q*g_norm) - sum_BHC_mapped(g_denorm*std*(p-bias)/(weight+eps^2)^2); mean/std detached, g_norm includes full loss+aux upstream contribution')
        assert norm_cmp['bitwise'] and result['independent_references_max_abs']==0
        ga=a.rev_norm.affine_weight.grad.detach().double();gb=b.rev_norm.affine_weight.grad.detach().double()
        assert bool(((ga-gb).abs()<=1e-7+1e-6*torch.maximum(ga.abs(),gb.abs())).all())
    except BaseException:
        result.update(status='diagnostic_failed',traceback=traceback.format_exc())
    finally:
        for h in handles:h.remove()
        stopped.set();thread.join(timeout=2);signal.alarm(0)
        INSTANCE.sample(torch)
        result['budget']=read(config['budget_file'])
        result['actual']={k:result['budget']['counts'][k]-CARRIED[k] for k in CARRIED}
        result['actual_this_process']={k:result['budget']['counts'][k]-config['process_carried'][k] for k in CARRIED}
        result['monitor_errors']=monitor_errors
        result['denials']=[v for v in (json.loads(l) for l in Path(config['audit_log']).read_text().splitlines()) if v['event']=='denied']
        verify_seal(read(config['seal_file']));write_new(config['report_file'],result)
        print(json.dumps({k:result[k] for k in ['status','actual','monitor_errors']},ensure_ascii=False),flush=True)
    return 0 if result['status']=='reference_conditions_satisfied' and not result['denials'] and not monitor_errors else 1


def launch(repo,evidence,attempt=1):
    from acceptance_driver import validate_inputs,file_seal,verify_seal
    from run_restricted import VERSION,TOOL_ROOT
    from resource_budget import initialize
    import current_policy
    repo=Path(repo).resolve();evidence=Path(evidence).resolve();validate_inputs(repo)
    prior=evidence.parent/'m4610-gelu-compat-4t0vksun/acceptance-conclusion.json'
    assert hashlib.sha256(prior.read_bytes()).hexdigest()==PRIOR_SHA
    assert read(prior)['cumulative']==dict(forward_API=114,backward=30,Adam_step=0)
    approval=read(evidence/'authorization.json');assert approval['cumulative_limits']==dict(forward=512,backward=80,adam=16)
    if attempt not in (1,2):raise ValueError('No automatic retry; only the documented ULP-recording repair continuation is supported')
    carried=dict(CARRIED); prior_seconds=0.
    if attempt==2:
        previous=read(evidence/'diagnostic-01/report.json');exit_record=read(evidence/'diagnostic-01/exit.json')
        assert previous['status']=='diagnostic_failed' and 'where expected condition to be a boolean tensor' in previous['traceback']
        assert previous['actual']==dict(forward=2,backward=2,adam=0) and not previous['denials'] and not previous['monitor_errors']
        carried=previous['budget']['counts'];prior_seconds=exit_record['elapsed_seconds']
    session=evidence/('diagnostic-%02d'%attempt);session.mkdir(exist_ok=False);(session/'fixtures').mkdir()
    limits=dict(forward=146,backward=38,adam=0,seconds=180-prior_seconds,method_seconds=180,rss=8*1024**3,reserved=4*1024**3,output=1024**3)
    budget=initialize(session/'budget.json','new_cuda/revin_diagnostic',limits)
    with budget.state() as s:s['counts']=dict(carried);s['carried_counts']=dict(carried);s['carried_source']=dict(path=str(prior),sha256=PRIOR_SHA)
    seal=file_seal(repo);seal[str(evidence/'authorization.json')]=hashlib.sha256((evidence/'authorization.json').read_bytes()).hexdigest()
    write_new(session/'sealed-inputs.json',seal)
    config=dict(version=VERSION,repo=str(repo),tool_root=str(TOOL_ROOT),session_root=str(session),revision=REVISION,process_carried=carried,
        audit_log=str(session/'audit.jsonl'),forbidden_roots=[str(repo/'data'),str(repo/'artifacts')],stage='new_cuda',
        access_policy='synthetic_regression',restriction_policy_id=current_policy.POLICY_ID,approval_sha256=current_policy.APPROVAL_SHA,
        budget_file=str(session/'budget.json'),limits=limits,seal_file=str(session/'sealed-inputs.json'),
        report_file=str(session/'report.json'),owner_pid=os.getpid(),business_bootstrap=True)
    path=session/'guard-config.json';write_new(path,config)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker-config',str(path)];write_new(session/'command.json',command)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GIT_OPTIONAL_LOCKS='0',PYTHONPATH=os.pathsep.join([str(TOOL_ROOT),str(repo/'tests'),str(repo)]),
        AMD_RR_CONFIG=str(path),AMD_RR_CONFIG_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),TMPDIR=str(session/'fixtures'),CUDA_VISIBLE_DEVICES='0')
    start=time.monotonic()
    with (session/'execution.log').open('x') as log:
        child=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_new(session/'process.json',dict(pid=child.pid,command=command,started_monotonic=start))
        try:code=child.wait(timeout=limits['seconds'])
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=124
    verify_seal(seal)
    exit_record=dict(exit_code=code,elapsed_seconds=time.monotonic()-start,budget=read(session/'budget.json'))
    write_new(session/'exit.json',exit_record);print(json.dumps({k:v for k,v in exit_record.items() if k!='budget'}))
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repo');parser.add_argument('--evidence');parser.add_argument('--worker-config');parser.add_argument('--attempt',type=int,default=1)
    args=parser.parse_args()
    raise SystemExit(worker(read(args.worker_config)) if args.worker_config else launch(args.repo,args.evidence,args.attempt))
