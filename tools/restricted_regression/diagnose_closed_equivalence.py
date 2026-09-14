"""Bounded CPU T512 diagnosis; no policy change, optimizer or acceptance rerun."""
import argparse
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

DIAGNOSTIC_ID = 'thls-closed-equivalence-gradient-diagnostic-v1.1'
CARRIED = dict(forward=93, backward=16, adam=0)
MAX_DIAGNOSTIC = dict(forward=8, backward=8, adam=0)
PRIOR_SHA = '7a28d6cffb275fc4014bd5155e18915bdec89b775923197d9f88d7006c34d843'
HEAD_SOURCE_SHA = 'd9195fe1459b1a77455d1455c852ec4555e9b22595395bfcc2e65004ddcafe96'


def json_compatible(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': 'NaN' if math.isnan(value) else ('+Infinity' if value > 0 else '-Infinity')}
    if isinstance(value, dict):
        return {k:json_compatible(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(v) for v in value]
    return value


def write_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(json_compatible(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def read(path):
    return json.loads(Path(path).read_text())


def worker(config):
    from restricted_io_guard import require_installed
    from current_policy import require_scope
    from acceptance_driver import verify_seal
    from resource_budget import INSTANCE, attach_analysis_apis
    import torch
    import numpy as np
    import types
    import main as runner
    from models.tsAMD import AMD
    from models.tsAMD_enhanced import AMDEnhanced
    from test_tsAMD_enhanced import AMDEnhancedTHLSTests

    state = require_installed()
    require_scope(state, 'new_cuda')
    assert config['diagnostic_id'] == DIAGNOSTIC_ID
    assert config['access_policy'] == 'synthetic_regression'
    verify_seal(read(config['seal_file']))
    attach_analysis_apis()
    torch.set_num_threads(1)  # Original class setUpClass, not a new thread policy.
    source = Path(config['head_source']).read_bytes()
    assert hashlib.sha256(source).hexdigest() == HEAD_SOURCE_SHA
    legacy = types.ModuleType('models._thls_diagnostic_head_enhanced')
    legacy.__file__ = config['head_source']
    legacy.__package__ = 'models'
    sys.modules[legacy.__name__] = legacy
    exec(compile(source, config['head_source'], 'exec'), legacy.__dict__)

    def eq(left, right):
        if torch.is_tensor(left):
            return torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
        if isinstance(left, np.ndarray):
            return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
        if isinstance(left, dict):
            return isinstance(right, dict) and left.keys() == right.keys() and all(eq(left[k], right[k]) for k in left)
        if isinstance(left, (tuple, list)):
            return type(left) is type(right) and len(left) == len(right) and all(eq(a, b) for a, b in zip(left, right))
        return left == right

    def digest_tensor(value):
        value = value.detach().cpu().contiguous()
        payload = (str(value.dtype) + str(tuple(value.shape))).encode() + value.numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()

    def tensor_info(value):
        if value is None:
            return dict(present=False)
        d = value.detach()
        return dict(present=True, shape=list(d.shape), dtype=str(d.dtype), device=str(d.device),
                    stride=list(d.stride()), finite=bool(torch.isfinite(d).all()),
                    l2=float(d.double().norm()), max_abs=float(d.abs().max()) if d.numel() else None,
                    sha256=digest_tensor(d))

    def compare(left, right):
        row = dict(left=tensor_info(left), right=tensor_info(right), none_pattern_equal=(left is None) == (right is None))
        if left is None or right is None:
            row['bitwise_equal'] = left is None and right is None
            return row
        row['shape_dtype_equal'] = left.shape == right.shape and left.dtype == right.dtype
        if not row['shape_dtype_equal']:
            row['bitwise_equal'] = False
            return row
        a, b = left.detach().double(), right.detach().double()
        delta = a - b
        mismatch = left.detach() != right.detach()
        row.update(bitwise_equal=torch.equal(left, right), different_elements=int(mismatch.sum()), elements=left.numel(),
                   max_abs=float(delta.abs().max()) if delta.numel() else None,
                   difference_l2=float(delta.norm()),
                   relative_l2_to_right=float(delta.norm()/b.norm()) if float(b.norm()) != 0 else None)
        if delta.numel():
            flat = int(delta.abs().reshape(-1).argmax())
            index = tuple(int(v) for v in np.unravel_index(flat, tuple(delta.shape)))
            row.update(worst_index=list(index), left_worst=float(a[index]), right_worst=float(b[index]))
        return row

    def compare_params(left, right):
        lp, rp = dict(left.named_parameters()), dict(right.named_parameters())
        assert lp.keys() == rp.keys()
        rows = {k:compare(lp[k].grad, rp[k].grad) for k in lp}
        different = [k for k, v in rows.items() if not v['bitwise_equal']]
        candidates = [k for k, v in rows.items() if 'max_abs' in v and v['max_abs'] is not None]
        maximum = max(candidates, key=lambda k:rows[k]['max_abs']) if candidates else None
        return dict(parameter_keys=len(rows), none_patterns_equal=all(v['none_pattern_equal'] for v in rows.values()),
                    all_bitwise_equal=not different, differing_keys=different,
                    first_differing_key=different[0] if different else None,
                    maximum_difference_key=maximum, maximum_abs=rows[maximum]['max_abs'] if maximum else None,
                    by_key=rows)

    def observe(model, x, name):
        records, gradients, handles, boundaries = {}, {}, [], {}
        before_rng = runner.capture_rng_state()
        before_grad_enabled = torch.is_grad_enabled()
        before_counts = dict(read(config['budget_file'])['counts'])
        original = x.detach().clone()
        def capture(key, tensor):
            records[key] = tensor.detach().clone()
            if tensor.requires_grad:
                def receive(grad):
                    gradients[key] = grad.detach().clone()
                    return None  # Observe, never replace a gradient.
                handles.append(tensor.register_hook(receive))
        def pre(module, args):
            boundaries['pre_rng'] = runner.capture_rng_state()
            boundaries['pre_grad_enabled'] = torch.is_grad_enabled()
            boundaries['same_input_object'] = args[0] is x
        def post(module, args, output):
            boundaries['post_rng'] = runner.capture_rng_state()
            boundaries['post_grad_enabled'] = torch.is_grad_enabled()
            boundaries['output'] = output
        def revin(module, args, output):
            key = 'revin_' + args[1]
            capture(key + '_input', args[0])
            capture(key + '_output', output)
            if args[1] == 'denorm':
                boundaries['denorm_target_slice'] = repr(args[2])
        handles += [model.register_forward_pre_hook(pre), model.register_forward_hook(post),
                    model.rev_norm.register_forward_hook(revin),
                    model.pastmixing.register_forward_hook(lambda m, a, o:capture('mdm_output', o)),
                    model.fc_blocks[-1].register_forward_hook(lambda m, a, o:capture('ddi_output', o)),
                    model.moe.register_forward_hook(lambda m, a, o:capture('ams_prediction', o[0]))]
        output = model(x)
        after_rng = runner.capture_rng_state()
        after_counts = read(config['budget_file'])['counts']
        checks = dict(name=name, counter_forward_increment=after_counts['forward']-before_counts['forward'],
                      input_identity_preserved=boundaries['same_input_object'], input_values_unchanged=torch.equal(original,x),
                      output_identity_preserved=output is boundaries['output'],
                      wrapper_pre_rng_unchanged=eq(before_rng, boundaries['pre_rng']),
                      wrapper_post_rng_unchanged=eq(boundaries['post_rng'], after_rng),
                      forward_body_rng_unchanged=eq(boundaries['pre_rng'], boundaries['post_rng']),
                      grad_enabled_preserved=before_grad_enabled == boundaries['pre_grad_enabled'] == boundaries['post_grad_enabled'] == torch.is_grad_enabled(),
                      denorm_target_slice=boundaries['denorm_target_slice'])
        assert checks['counter_forward_increment'] == 1
        assert all(checks[k] for k in ('input_identity_preserved','input_values_unchanged','output_identity_preserved',
                                      'wrapper_pre_rng_unchanged','wrapper_post_rng_unchanged','grad_enabled_preserved'))
        return output, records, gradients, handles, checks

    stop = threading.Event()
    monitor_errors = []
    def monitor():
        while not stop.wait(.5):
            try:
                INSTANCE.sample(torch)
                if sum(p.stat().st_size for p in Path(config['session_root']).rglob('*') if p.is_file()) > config['limits']['output']:
                    raise RuntimeError('diagnostic output limit')
            except BaseException as exc:
                monitor_errors.append(str(exc)); INSTANCE.stop(exc)
                os.kill(os.getpid(), signal.SIGALRM)
                return
    def deadline(signum, frame):
        raise TimeoutError('diagnostic time/resource stop')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(180)
    monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
    result = dict(diagnostic_id=DIAGNOSTIC_ID, cases=[], acceptance_tests_executed=0, optimizer_steps=0,
                  model_forward_device='cpu', torch_version=torch.__version__, threads=torch.get_num_threads(),
                  grad_enabled_initial=torch.is_grad_enabled(), policy_id=config['restriction_policy_id'])
    reference = None
    all_handles = []
    try:
        for case in ('current_vs_frozen', 'head_vs_frozen'):
            INSTANCE.current('diagnostic/' + case)
            common, options = AMDEnhancedTHLSTests._kwargs(512)
            runner.set_seed(2024)
            if case == 'current_vs_frozen':
                model = AMDEnhanced(**common, **options)
                assert model.target_history_local_shape is None
                assert model.use_target_history_local_shape is False
            else:
                old_options = {k:v for k,v in options.items() if k not in ('local_shape_contract_declared','use_target_history_local_shape')}
                model = legacy.AMDEnhanced(**common, **old_options)
            runner.set_seed(2024)
            frozen = AMD(**common, target_slice=slice(options['target_idx'],options['target_idx']+1))
            assert eq(model.state_dict(), frozen.state_dict())
            model.to('cpu').eval(); frozen.to('cpu').eval()
            initial_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
            x=torch.randn(2,512,7,device='cpu',requires_grad=True)
            xf=x.detach().clone().requires_grad_()
            initial_input = dict(equal=torch.equal(x,xf), both_leaf=x.is_leaf and xf.is_leaf,
                                 requires_grad=x.requires_grad and xf.requires_grad, no_grad_accumulation=x.grad is None and xf.grad is None,
                                 distinct_storage=x.data_ptr()!=xf.data_ptr(), left=tensor_info(x), right=tensor_info(xf))
            assert all(initial_input[k] for k in ('equal','both_leaf','requires_grad','no_grad_accumulation','distinct_storage'))
            assert initial_input['left']['stride'] == initial_input['right']['stride']
            assert all(p.grad is None for p in model.parameters()) and all(p.grad is None for p in frozen.parameters())
            out, layers, grads, handles, wrapped = observe(model,x,'enhanced');all_handles+=handles
            f_out, f_layers, f_grads, f_handles, f_wrapped = observe(frozen,xf,'frozen');all_handles+=f_handles
            prediction,auxiliary=out;f_prediction,f_auxiliary=f_out
            loss=prediction.square().mean()+auxiliary;f_loss=f_prediction.square().mean()+f_auxiliary
            aligned = dict(prediction=compare(prediction,f_prediction),auxiliary=compare(auxiliary,f_auxiliary),loss=compare(loss,f_loss))
            assert all(v['bitwise_equal'] for v in aligned.values())
            before_backward=read(config['budget_file'])['counts']['backward']
            loss.backward();f_loss.backward()
            assert read(config['budget_file'])['counts']['backward']-before_backward==2
            layer_rows={}
            for k in layers:
                a,b=layers[k],f_layers[k]
                ga,gb=grads.get(k),f_grads.get(k)
                if k=='revin_denorm_output':
                    a,b=a[:,:,6:7],b[:,:,6:7]
                    ga=None if ga is None else ga[:,:,6:7];gb=None if gb is None else gb[:,:,6:7]
                layer_rows[k]=dict(values=compare(a,b),gradient=compare(ga,gb))
            row=dict(case=case,input=initial_input,parameters_buffers_initial_aligned=True,mode='eval',
                     dtype='torch.float32',shape=[2,512,7],target_index=6,
                     prediction_auxiliary_loss=aligned,input_gradient=compare(x.grad,xf.grad),
                     public_parameter_gradients=compare_params(model,frozen),intermediates=layer_rows,
                     wrapper_checks=[wrapped,f_wrapped],
                     parameters_buffers_unchanged=eq(initial_state,model.state_dict()) and eq(initial_state,frozen.state_dict()),
                     eval_mode_unchanged=all(not m.training for m in model.modules()) and all(not m.training for m in frozen.modules()))
            # Save observed values before a stop assertion; nonfinite values are explicit tagged JSON.
            write_new(Path(config['session_root'])/(case+'.observation.json'),row)
            assert row['parameters_buffers_unchanged'] and row['eval_mode_unchanged']
            assert row['input_gradient']['left']['finite'] and row['input_gradient']['right']['finite']
            assert all(v[side].get('finite',True) for v in row['public_parameter_gradients']['by_key'].values() for side in ('left','right'))
            if reference is None:
                reference=dict(input_digest=digest_tensor(x),gradient=x.grad.detach().clone(),
                               parameter_gradients={k:None if p.grad is None else p.grad.detach().clone() for k,p in model.named_parameters()})
            else:
                row['same_input_as_first_pair']=digest_tensor(x)==reference['input_digest']
                row['HEAD_vs_current_input_gradient']=compare(x.grad,reference['gradient'])
                row['HEAD_vs_current_all_parameter_gradients_bitwise']=all(eq(p.grad,reference['parameter_gradients'][k]) for k,p in model.named_parameters())
                assert row['same_input_as_first_pair']
            result['cases'].append(row)
            write_new(Path(config['session_root'])/(case+'.json'),row)
            for handle in all_handles:handle.remove()
            all_handles=[]
            del model,frozen,x,xf,out,f_out,layers,f_layers,grads,f_grads,initial_state
            if case=='current_vs_frozen' and row['input_gradient']['bitwise_equal']:
                result['stop_reason']='failure_not_reproduced; no automatic acceptance or repeated retries'
                break
        if len(result['cases'])==2:
            old_diff=not result['cases'][1]['input_gradient']['bitwise_equal']
            result['stop_reason']='HEAD_enhanced_also_differs; existing_path_or_numerical_contract_requires_review' if old_diff else 'HEAD_does_not_reproduce; further_cause_not_yet_established'
        result['status']='diagnosis_completed_requires_review'
    except BaseException:
        result['status']='diagnostic_failed'
        result['traceback']=traceback.format_exc()
    finally:
        for handle in all_handles:handle.remove()
        stop.set();monitor_thread.join(timeout=2);signal.alarm(0)
        result['monitor_errors']=monitor_errors
        result['budget']=read(config['budget_file'])
        result['this_diagnostic_counts']={k:result['budget']['counts'][k]-CARRIED[k] for k in CARRIED}
        result['audit_denials']=[v for v in (json.loads(l) for l in Path(config['audit_log']).read_text().splitlines()) if v['event']=='denied']
        result['cuda_model_forward']=0
        result['real_data_reads']=0
        result['checkpoint_loads']=0
        verify_seal(read(config['seal_file']))
        write_new(config['report_file'],result)
        print(json.dumps({k:v for k,v in result.items() if k in ('status','stop_reason','this_diagnostic_counts','monitor_errors')},ensure_ascii=False),flush=True)
    return 2 if result['status']=='diagnosis_completed_requires_review' and not result['audit_denials'] else 1


def launch(repo,evidence):
    from acceptance_driver import validate_inputs,file_seal,verify_seal
    from run_restricted import VERSION,TOOL_ROOT
    import current_policy
    from resource_budget import initialize
    repo=Path(repo).resolve();evidence=Path(evidence).resolve()
    assert (evidence/'starting-state.json').exists()
    validate_inputs(repo)
    prior=evidence.parent/'m4607-thls-v6w8re5_/continuation-result.json'
    assert hashlib.sha256(prior.read_bytes()).hexdigest()==PRIOR_SHA
    assert read(prior)['new_actual_call_counts']['reconciled_workload']==CARRIED
    session=evidence/'diagnostic-01';session.mkdir(exist_ok=False);(session/'fixtures').mkdir()
    limits=dict(forward=101,backward=24,adam=0,seconds=180,method_seconds=180,
                rss=8*1024**3,reserved=4*1024**3,output=1024**3)
    budget=initialize(session/'budget.json','new_cuda/closed_equivalence_diagnosis',limits)
    with budget.state() as s:
        s['counts']=dict(CARRIED);s['carried_counts']=dict(CARRIED)
        s['carried_source']=dict(path=str(prior),sha256=PRIOR_SHA,includes_original_89_plus_4_reconciliation=True)
        s['full_authorized_limits']=dict(forward=512,backward=64,adam=16)
        s['diagnostic_attempt_cap']=dict(MAX_DIAGNOSTIC)
    seal=file_seal(repo);seal[str(evidence/'head_tsAMD_enhanced.py')]=HEAD_SOURCE_SHA
    write_new(session/'sealed-inputs.json',seal)
    config=dict(version=VERSION,repo=str(repo),tool_root=str(TOOL_ROOT),session_root=str(session),
                diagnostic_id=DIAGNOSTIC_ID,head_source=str(evidence/'head_tsAMD_enhanced.py'),
                audit_log=str(session/'audit.jsonl'),forbidden_roots=[str(repo/'data'),str(repo/'artifacts')],
                stage='new_cuda',access_policy='synthetic_regression',restriction_policy_id=current_policy.POLICY_ID,
                approval_sha256=current_policy.APPROVAL_SHA,budget_file=str(session/'budget.json'),limits=limits,
                seal_file=str(session/'sealed-inputs.json'),report_file=str(session/'report.json'),
                owner_pid=os.getpid(),business_bootstrap=True)
    path=session/'guard-config.json';write_new(path,config)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker-config',str(path)]
    write_new(session/'command.json',command)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GIT_OPTIONAL_LOCKS='0',
             PYTHONPATH=os.pathsep.join([str(TOOL_ROOT),str(repo/'tests'),str(repo)]),
             AMD_RR_CONFIG=str(path),AMD_RR_CONFIG_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
             TMPDIR=str(session/'fixtures'),CUDA_VISIBLE_DEVICES='0')
    started=time.monotonic()
    with (session/'execution.log').open('x') as log:
        child=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_new(session/'process.json',dict(pid=child.pid,command=command,started_monotonic=started))
        try:code=child.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=124
    verify_seal(seal)
    result=dict(exit_code=code,elapsed_seconds=time.monotonic()-started,budget=read(session/'budget.json'),
                diagnostic_report_exists=(session/'report.json').exists(),acceptance_or_inherited_or_real_probe_started=False)
    write_new(session/'exit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='budget'},ensure_ascii=False))
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repo');parser.add_argument('--evidence');parser.add_argument('--worker-config')
    args=parser.parse_args()
    if args.worker_config:
        raise SystemExit(worker(read(args.worker_config)))
    raise SystemExit(launch(args.repo,args.evidence))
