"""One guarded CUDA pair and complete bias derivative analysis from immutable snapshots."""
import argparse
import csv
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

REVISION = 'revin-cuda-affine-group-reference-v1'
CARRIED = dict(forward=141, backward=44, adam=0)
PRIOR_SHA = '195a15a7fbdbb83752bc7addbe1205aab51b1b4f0ffa67f9c73089a0e10175a7'
WEIGHT_REFERENCE_SHA = '0437d040f57af38ee886bda85654ccd07779b686ecf70dd075badbdf3a81ede7'


def read(path):
    return json.loads(Path(path).read_text())


def safe(value):
    """Serialize nonfinite observations explicitly, without treating them as accepted values."""
    if isinstance(value, dict): return {k:safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [safe(v) for v in value]
    if isinstance(value, F): return {'numerator':str(value.numerator),'denominator':str(value.denominator)}
    if hasattr(value,'item') and not isinstance(value,(str,bytes)): return safe(value.item())
    if isinstance(value,float) and not math.isfinite(value):
        return {'nonfinite':'NaN' if math.isnan(value) else ('+Infinity' if value>0 else '-Infinity')}
    return value


def write_new(path,value):
    payload=json.dumps(safe(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with Path(path).open('x') as f:f.write(payload)


def worker(config):
    from restricted_io_guard import require_installed
    from current_policy import require_scope
    from acceptance_driver import verify_seal
    from resource_budget import INSTANCE, attach_analysis_apis
    from analyze_revin_roundoff import frac,rn32,decomposition,assumption_record,U,gamma,exact_record,MIN_NORMAL
    import torch
    import numpy as np
    import main as runner
    from models.tsAMD import AMD
    from models.tsAMD_enhanced import AMDEnhanced
    from test_tsAMD_enhanced import AMDEnhancedTHLSTests, _closed_gradient_numerics
    state=require_installed();require_scope(state,'new_cuda')
    assert config['access_policy']=='synthetic_regression' and config['revision']==REVISION
    verify_seal(read(config['seal_file']));attach_analysis_apis();torch.set_num_threads(1)
    session=Path(config['session_root']);handles=[];snapshots={};nonfinite=[];monitor_errors=[]
    stop=threading.Event();result=dict(status='running',revision=REVISION,permanent_IDs_executed=0)
    def alarm(signum,frame):raise TimeoutError('bias diagnostic time/resource bound')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(180)
    def monitor():
        while not stop.wait(.5):
            try:
                INSTANCE.sample(torch)
                if sum(p.stat().st_size for p in session.rglob('*') if p.is_file())>config['limits']['output']:
                    raise RuntimeError('diagnostic output bound')
            except BaseException as exc:
                monitor_errors.append(str(exc));INSTANCE.stop(exc);os.kill(os.getpid(),signal.SIGALRM);return
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    def layout(t):
        return dict(shape=list(t.shape),stride=list(t.stride()),storage_offset=t.storage_offset(),contiguous=t.is_contiguous(),
                    dtype=str(t.dtype),device=str(t.device),storage_ptr=t.untyped_storage().data_ptr(),requires_grad=t.requires_grad)
    def capture(key,t):
        assert key not in snapshots,'duplicate snapshot '+key
        meta=layout(t);v=t.detach().clone().cpu();snapshots[key]=v
        row=dict(layout=meta,values=v.tolist(),finite=bool(torch.isfinite(v).all()),
                 nan=int(torch.isnan(v).sum()),positive_inf=int(torch.isposinf(v).sum()),negative_inf=int(torch.isneginf(v).sum()))
        write_new(session/(key+'.json'),row)
        if not row['finite']:nonfinite.append(key)
    def same(a,b):
        if torch.is_tensor(a):return torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
        if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        return a==b
    def observe(side,model):
        def node_hook(kind):
            def hook(gin,gout):
                for direction,values in [('gin',gin),('gout',gout)]:
                    for i,t in enumerate(values):
                        if t is not None:capture(side+'_'+kind+'_'+direction+str(i),t)
                return None
            return hook
        def hook(module,args,out):
            kind=args[1];capture(side+'_'+kind+'_input',args[0]);capture(side+'_'+kind+'_output',out)
            def grad(g):capture(side+'_'+kind+'_cotangent',g);return None
            handles.append(out.register_hook(grad))
            if kind=='norm':
                add=out.grad_fn;mul=add.next_functions[0][0]
                assert add.name().startswith('AddBackward') and mul.name().startswith('MulBackward')
                capture(side+'_norm_q',mul._saved_self);capture(side+'_norm_saved_weight',mul._saved_other)
                capture(side+'_mean',module.mean);capture(side+'_stdev',module.stdev)
                handles.append(add.register_hook(node_hook('norm_add')))
                handles.append(mul.register_hook(node_hook('norm_mul')))
            else:
                mul=out.grad_fn.next_functions[0][0];div=mul.next_functions[0][0];sub=div.next_functions[0][0]
                assert mul.name().startswith('MulBackward') and div.name().startswith('DivBackward') and sub.name().startswith('SubBackward')
                capture(side+'_den_saved_std',mul._saved_other);capture(side+'_den_divisor',div._saved_other)
                capture(side+'_den_numerator',div._saved_self)
                for label,node in [('den_mul',mul),('den_div',div),('den_sub',sub)]:handles.append(node.register_hook(node_hook(label)))
                write_new(session/(side+'-contract.json'),dict(target_slice=repr(args[2]),eps=module.eps,
                    affine=module.affine,training=module.training,statistics_detached=not module.mean.requires_grad and not module.stdev.requires_grad,
                    bias_uses=['norm AddBackward input1','denorm SubBackward input1'],weight_uses=['norm MulBackward input1','denorm DivBackward divisor'],
                    denorm_nodes=[mul.name(),div.name(),sub.name()]))
            return None
        handles.append(model.rev_norm.register_forward_hook(hook))
    def persist_gradients(side,model,x):
        arrays={};metadata={}
        for name,p in [('input',x),*model.named_parameters()]:
            g=p.grad
            if g is None:metadata[name]=dict(none=True);continue
            meta=layout(g);v=g.detach().clone().cpu();arrays[name]=v.numpy()
            finite=bool(torch.isfinite(v).all());metadata[name]=dict(none=False,layout=meta,finite=finite,
                nan=int(torch.isnan(v).sum()),positive_inf=int(torch.isposinf(v).sum()),negative_inf=int(torch.isneginf(v).sum()))
            if not finite:nonfinite.append(side+'/'+name)
        with (session/(side+'-all-gradients.npz')).open('xb') as f:np.savez(f,**arrays)
        write_new(session/(side+'-all-gradient-layouts.json'),metadata)
        for name in ['rev_norm.affine_weight','rev_norm.affine_bias']:
            capture(side+'_'+name.rsplit('.',1)[1]+'_grad',dict(model.named_parameters())[name].grad)
    def vector(a,b):
        row=_closed_gradient_numerics(a,b)
        row.update(left=a.detach().clone().cpu().tolist(),right=b.detach().clone().cpu().tolist(),
                   different_indices=(a!=b).nonzero().cpu().tolist(),absolute_differences=(a.double()-b.double()).abs().cpu().tolist())
        return row
    try:
        INSTANCE.current('diagnostic/revin_cuda_T512_affine_pair')
        if not torch.cuda.is_available():raise RuntimeError('Pending: CUDA required and unavailable')
        result.update(device=torch.cuda.get_device_name(0),torch_version=torch.__version__)
        common,options=AMDEnhancedTHLSTests._kwargs(512)
        runner.set_seed(2024);a=AMDEnhanced(**common,**options).to('cuda').eval()
        runner.set_seed(2024);b=AMD(**common,target_slice=slice(6,7)).to('cuda').eval()
        assert same(a.state_dict(),b.state_dict()) and all(p.grad is None for m in (a,b) for p in m.parameters())
        initial={k:v.detach().clone() for k,v in a.state_dict().items()}
        x=torch.randn(2,512,7,device='cuda',requires_grad=True);xb=x.detach().clone().requires_grad_()
        capture('input_A',x);capture('input_AMD',xb);observe('A',a);observe('AMD',b);rng=runner.capture_rng_state()
        pa,aa=a(x)
        if nonfinite or not torch.isfinite(pa).all() or not torch.isfinite(aa).all():raise RuntimeError('nonfinite A forward')
        pb,ab=b(xb);la=pa.square().mean()+aa;lb=pb.square().mean()+ab
        forward=dict(prediction_bitwise=torch.equal(pa,pb),moe_bitwise=torch.equal(aa,ab),loss_bitwise=torch.equal(la,lb),
            prediction_values=[pa.detach().cpu().tolist(),pb.detach().cpu().tolist()],aux_values=[float(aa),float(ab)],loss_values=[float(la),float(lb)],
            state_equal=same(a.state_dict(),b.state_dict()),rng_equal=same(rng,runner.capture_rng_state()))
        write_new(session/'forward-alignment.json',forward)
        assert all(forward[k] for k in ['prediction_bitwise','moe_bitwise','loss_bitwise','state_equal','rng_equal']) and not nonfinite
        la.backward();persist_gradients('A',a,x)
        if nonfinite:raise RuntimeError('nonfinite after A backward; B backward not executed')
        lb.backward();persist_gradients('AMD',b,xb)
        params={k:_closed_gradient_numerics(p.grad,dict(b.named_parameters())[k].grad) for k,p in a.named_parameters()}
        pair=dict(weight=vector(a.rev_norm.affine_weight.grad,b.rev_norm.affine_weight.grad),bias=vector(a.rev_norm.affine_bias.grad,b.rev_norm.affine_bias.grad),
                  parameters=params,input_gradient=_closed_gradient_numerics(x.grad,xb.grad),
                  state_unchanged=same(initial,a.state_dict()) and same(initial,b.state_dict()),rng_unchanged=same(rng,runner.capture_rng_state()),
                  mode_eval=all(not m.training for model in (a,b) for m in model.modules()))
        write_new(session/'paired-gradients.json',pair)
        result['pair']=pair
        assert not nonfinite and all(v['finite'] and v['none_equal'] for v in params.values()) and pair['input_gradient']['finite']
        assert pair['state_unchanged'] and pair['rng_unchanged'] and pair['mode_eval'] and pair['input_gradient']['bitwise_equal']
        assert all(v['bitwise_equal'] for k,v in params.items() if k not in ['rev_norm.affine_weight','rev_norm.affine_bias'])
        assert pair['weight']['approved_symmetric_bound_satisfied']
        # Exactly one independent-reference API, with no autograd, CPU reduction replay or extra model calls.
        INSTANCE.charge('forward','independent_complete_CUDA_bias_saved_snapshot_reference')
        def values(side,key):return snapshots[side+'_'+key].tolist()
        def flat(side,key):return snapshots[side+'_'+key].reshape(-1).tolist()
        for key in ['norm_q','norm_cotangent','denorm_input','denorm_cotangent','mean','stdev']:
            assert torch.equal(snapshots['A_'+key],snapshots['AMD_'+key]),key
        assert torch.count_nonzero(snapshots['A_denorm_cotangent'][:,:,:6])==0
        reports={}
        for side in ('A','AMD'):
            g=values(side,'norm_add_gout0');n_actual=flat(side,'norm_add_gin1');norm=[];nh=[];nb=[]
            assert g==values(side,'norm_cotangent')
            with (session/(side+'-norm-bias-terms.csv')).open('x',newline='') as f:
                writer=csv.writer(f);writer.writerow(['channel','batch','time','cotangent'])
                for c in range(7):
                    terms=[]
                    for i in range(2):
                        for t in range(512):v=frac(g[i][t][c]);terms.append(v);writer.writerow([c,i,t,float(v)])
                    row=decomposition(terms,terms,n_actual[c],F(0));row['channel']=c;norm.append(row)
                    nh.append(sum(terms,F(0)));nb.append(gamma(1023)*sum(map(abs,terms),F(0)))
            write_new(session/(side+'-norm-bias-analysis.json'),norm)
            go=values(side,'denorm_cotangent');std=values(side,'den_saved_std');divisors=flat(side,'den_divisor');sub_go=values(side,'den_sub_gout0')
            div_go=values(side,'den_div_gout0');d_actual=flat(side,'den_sub_gin1')
            assert all(v==1. for v in divisors)
            mapping=list(range(7)) if side=='A' else [6]*7
            terms={};stage1=[];nonzero_go=[];std_values=[]
            with (session/(side+'-denorm-bias-terms.csv')).open('x',newline='') as f:
                writer=csv.writer(f);writer.writerow(['channel','mapped_bias','batch','time','outer_cotangent','saved_stdev','exact_product_num','exact_product_den','rounded_mul','saved_sub_cotangent'])
                for c,j in enumerate(mapping):
                    for i in range(2):
                        for t in range(96):
                            gv=frac(go[i][t][c]);sv=frac(std[i][0][c if side=='A' else 0]);exact=gv*sv;rounded=rn32(exact)
                            assert abs(rounded-exact)<=U*abs(exact)
                            assert float(rounded)==div_go[i][t][c]==sub_go[i][t][c]
                            if gv:nonzero_go.append(gv)
                            std_values.append(sv);stage1.append(rounded)
                            terms.setdefault(j,[]).append((-exact,-rounded))
                            writer.writerow([c,j,i,t,float(gv),float(sv),exact.numerator,exact.denominator,float(rounded),sub_go[i][t][c]])
            for sequence in [nonzero_go,std_values,stage1]:assumption_record(sequence)
            den=[];dh=[F(0)]*7;db=[F(0)]*7;denmapped=[0.]*7
            eps=read(session/(side+'-contract.json'))['eps'];ideal_divisor=F(1)+F.from_float(eps*eps)
            for j,terms_j in terms.items():
                high=[z[0] for z in terms_j];rounded=[z[1] for z in terms_j];actual=d_actual[j if side=='A' else 0]
                bound=U*sum(map(abs,high),F(0));row=decomposition(high,rounded,actual,bound)
                ideal=sum(high,F(0))/ideal_divisor;den_input_error=sum(high,F(0))-ideal
                row.update(channel=j,saved_divisor=1.,ideal_divisor=exact_record(ideal_divisor),denominator_input_rounding_error=float(den_input_error),
                    multiplication_terms_correctly_rounded_and_match_saved=True,division_by_saved_one_exact=True,sign_negation_exact=True,
                    observed_reduction_axes=[0,1] if side=='A' else [0,1,2],
                    observed_reduced_shape=read(session/(side+'_den_sub_gin1.json'))['layout']['shape'])
                den.append(row);dh[j]=ideal;denmapped[j]=actual;db[j]=bound+gamma(len(terms_j)-1)*sum(map(abs,rounded),F(0))+abs(den_input_error)
            write_new(session/(side+'-denorm-bias-analysis.json'),den)
            actual=flat(side,'affine_bias_grad');total=[]
            for c in range(7):
                partial=frac(n_actual[c])+frac(denmapped[c]);rounded=rn32(partial);assert float(rounded)==actual[c]
                ref=nh[c]+dh[c];error=frac(actual[c])-ref;bound=nb[c]+db[c]+U*abs(partial);assert abs(error)<=bound
                total.append(dict(channel=c,norm=n_actual[c],denorm=denmapped[c],actual=actual[c],reference=exact_record(ref),
                    error=float(error),bound=float(bound),final_add_roundoff=float(rounded-partial),
                    exact_sum_reconstructs_gradient=True,within_derived_bound=True,
                    bound_to_gradient=float(bound/abs(frac(actual[c]))) if actual[c] else None))
            write_new(session/(side+'-complete-bias-reference.json'),total)
            reports[side]=dict(norm=norm,denorm=den,total=total)
        assert pair['bias']['approved_symmetric_bound_satisfied']
        reference_diff=[reports['A']['total'][c]['reference']==reports['AMD']['total'][c]['reference'] for c in range(7)]
        assert all(reference_diff)
        component={key:vector(snapshots['A_'+key].reshape(-1),snapshots['AMD_'+key].reshape(-1)) for key in ['norm_add_gin1']}
        da=snapshots['A_den_sub_gin1'].reshape(-1);dbias=torch.zeros_like(da);dbias[6]=snapshots['AMD_den_sub_gin1'].reshape(-1)[0]
        component['denorm_mapped']=vector(da,dbias)
        write_new(session/'bias-contribution-comparison.json',component)
        result.update(status='complete_CUDA_bias_reference_conditions_Passed',references=reports,component_comparison=component,
            formula='sum_BT(g_norm) - sum_BHC_mapped(g_denorm*detached_stdev/(weight+eps^2)); both actual bias uses included',
            reference_inputs='Actual CUDA AddBackward/SubBackward cotangents and saved stdev/divisor; no CPU reduction replay',
            reference_precision='Exact Fraction sums, integer float32 RN for each product and final accumulation; decimals for reporting only',
            assumptions='Normal values; sum_abs excludes overflow and common quantum excludes subnormal partial sums; saved divisor1 gives exact division/negation',
            scope='Different broadcast reduction organizations observed from actual CUDA nodes; conservative derived bounds plus per-term/formula/state checks, not GPU instruction tracing or full-training proof')
    except BaseException:
        result.update(status='diagnostic_failed',traceback=traceback.format_exc())
    finally:
        for h in handles:h.remove()
        stop.set();thread.join(timeout=2);signal.alarm(0)
        try:INSTANCE.sample(torch)
        except BaseException as exc:result['final_resource_error']=str(exc)
        result['budget']=read(config['budget_file']);result['actual']={k:result['budget']['counts'][k]-CARRIED[k] for k in CARRIED}
        result['nonfinite_snapshots']=nonfinite;result['monitor_errors']=monitor_errors
        result['denials']=[x for x in (json.loads(l) for l in Path(config['audit_log']).read_text().splitlines()) if x['event']=='denied']
        verify_seal(read(config['seal_file']));write_new(config['report_file'],result)
        print(json.dumps({k:result[k] for k in ['status','actual','monitor_errors']},ensure_ascii=False),flush=True)
    return 0 if result['status']=='complete_CUDA_bias_reference_conditions_Passed' and not result['denials'] and not monitor_errors else 1


def launch(repo,evidence):
    from acceptance_driver import validate_inputs,file_seal,verify_seal
    from run_restricted import VERSION,TOOL_ROOT
    from resource_budget import initialize
    import current_policy
    repo=Path(repo).resolve();evidence=Path(evidence).resolve();validate_inputs(repo)
    previous=evidence.parent/'m4612-revin-roundoff-up8puund';prior=previous/'acceptance-conclusion.json';weight=previous/'analysis-conclusion.json'
    assert hashlib.sha256(prior.read_bytes()).hexdigest()==PRIOR_SHA and read(prior)['cumulative']==CARRIED
    assert hashlib.sha256(weight.read_bytes()).hexdigest()==WEIGHT_REFERENCE_SHA
    assert read(evidence/'authorization.json')['cumulative_limits']==dict(forward=512,backward=96,adam=16)
    session=evidence/'diagnostic-01';session.mkdir(exist_ok=False);(session/'fixtures').mkdir()
    limits=dict(forward=173,backward=52,adam=0,seconds=180,method_seconds=180,rss=8*1024**3,reserved=4*1024**3,output=1024**3)
    budget=initialize(session/'budget.json','new_cuda/bias_diagnostic',limits)
    with budget.state() as s:s['counts']=dict(CARRIED);s['carried_counts']=dict(CARRIED);s['carried_source']=dict(path=str(prior),sha256=PRIOR_SHA)
    seal=file_seal(repo)
    for p in [prior,weight,evidence/'authorization.json']:seal[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    write_new(session/'sealed-inputs.json',seal)
    config=dict(version=VERSION,revision=REVISION,repo=str(repo),tool_root=str(TOOL_ROOT),session_root=str(session),
        audit_log=str(session/'audit.jsonl'),forbidden_roots=[str(repo/'data'),str(repo/'artifacts')],stage='new_cuda',
        access_policy='synthetic_regression',restriction_policy_id=current_policy.POLICY_ID,approval_sha256=current_policy.APPROVAL_SHA,
        budget_file=str(session/'budget.json'),limits=limits,seal_file=str(session/'sealed-inputs.json'),report_file=str(session/'report.json'),
        owner_pid=os.getpid(),business_bootstrap=True)
    path=session/'guard-config.json';write_new(path,config)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker-config',str(path)];write_new(session/'command.json',command)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GIT_OPTIONAL_LOCKS='0',PYTHONPATH=os.pathsep.join([str(TOOL_ROOT),str(repo/'tests'),str(repo)]),
        AMD_RR_CONFIG=str(path),AMD_RR_CONFIG_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),TMPDIR=str(session/'fixtures'),CUDA_VISIBLE_DEVICES='0')
    start=time.monotonic()
    with (session/'execution.log').open('x') as log:
        child=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_new(session/'process.json',dict(pid=child.pid,command=command,started_monotonic=start))
        try:code=child.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=124
    verify_seal(seal);record=dict(exit_code=code,elapsed_seconds=time.monotonic()-start,budget=read(session/'budget.json'))
    write_new(session/'exit.json',record);print(json.dumps({k:v for k,v in record.items() if k!='budget'}));return code


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repo');parser.add_argument('--evidence');parser.add_argument('--worker-config')
    args=parser.parse_args()
    raise SystemExit(worker(read(args.worker_config)) if args.worker_config else launch(args.repo,args.evidence))
