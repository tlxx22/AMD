"""Read-only CPU DDI node observations; inherited budget and access policy intact."""
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

from diagnose_closed_equivalence import read, write_new, json_compatible

DIAGNOSTIC_ID = 'thls-ddi-operator-diagnostic-v1'
CARRIED = dict(forward=97, backward=20, adam=0)
TURN_CAP = dict(forward=32, backward=6, adam=0)
PRIOR_SHA = '6a98a844372596b2dc4cadab89304d2d39c14c3e7748172872efd8860fa6e2b5'


def worker(config):
    from restricted_io_guard import require_installed
    from current_policy import require_scope
    from acceptance_driver import verify_seal
    from resource_budget import INSTANCE
    import torch
    import numpy as np
    import main as runner
    from models.tsAMD import AMD
    from models.tsAMD_enhanced import AMDEnhanced
    from test_tsAMD_enhanced import AMDEnhancedTHLSTests

    require_scope(require_installed(), 'new_cuda')
    assert config['diagnostic_id'] == DIAGNOSTIC_ID
    verify_seal(read(config['seal_file']))
    torch.set_num_threads(1)
    root = Path(config['session_root'])
    handles = []
    stop = threading.Event()
    monitor_errors = []
    snapshots = {}
    events = (root / 'node-events.jsonl').open('x')
    event_count = 0

    class ObservedNonfinite(RuntimeError):
        pass

    def eq(a, b):
        if torch.is_tensor(a):
            return torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
        if isinstance(a, np.ndarray):
            return isinstance(b, np.ndarray) and np.array_equal(a, b)
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(eq(a[k], b[k]) for k in a)
        if isinstance(a, (tuple, list)):
            return type(a) is type(b) and len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
        return a == b

    def layout(t):
        return dict(shape=list(t.shape), stride=list(t.stride()), storage_offset=t.storage_offset(),
                    contiguous=t.is_contiguous(), dtype=str(t.dtype), device=str(t.device),
                    storage_ptr=t.untyped_storage().data_ptr(), data_ptr=t.data_ptr(),
                    tensor_version=t._version, requires_grad=t.requires_grad, is_leaf=t.is_leaf,
                    base_storage_ptr=None if t._base is None else t._base.untyped_storage().data_ptr())

    def snapshot(t, key):
        if t is None:
            return dict(present=False)
        original_layout = layout(t)  # Capture before clone changes strides/offset/storage.
        x = t.detach().clone()
        snapshots[key] = x
        finite = torch.isfinite(x)
        a = x.double()
        finite_a = a[torch.isfinite(a)]
        maximum = float(finite_a.abs().max()) if finite_a.numel() else None
        # Reductions use copies in float64; no f32 squared-norm overflow.
        l2 = float(a.norm()) if bool(finite.all()) else None
        return dict(present=True, key=key, layout=original_layout,
                    finite=bool(finite.all()), nan=int(torch.isnan(x).sum()),
                    posinf=int(torch.isposinf(x).sum()), neginf=int(torch.isneginf(x).sum()),
                    finite_max_abs=maximum, l2_float64=l2,
                    sha256=hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest())

    def emit(kind, **row):
        nonlocal event_count
        event_count += 1
        value = dict(event=kind, ordinal=event_count, **row)
        events.write(json.dumps(json_compatible(value), ensure_ascii=False, allow_nan=False) + '\n')
        events.flush()
        return value

    def checked_emit(kind, rows, **data):
        emit(kind, tensors=rows, **data)
        if any(x.get('present') and not x['finite'] for x in rows):
            raise ObservedNonfinite(f'nonfinite {kind}: {data}')

    def compare_keys(ka, kb):
        a, b = snapshots.get(ka), snapshots.get(kb)
        if a is None or b is None:
            return dict(bitwise_equal=a is None and b is None, left_present=a is not None, right_present=b is not None)
        if a.shape != b.shape or a.dtype != b.dtype:
            return dict(bitwise_equal=False, shape_dtype_equal=False)
        d = a.double() - b.double()
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        row = dict(bitwise_equal=torch.equal(a, b), shape_dtype_equal=True,
                   different_elements=int((a != b).sum()), elements=a.numel(), finite=finite)
        if finite:
            ref = float(b.double().norm())
            row.update(max_abs=float(d.abs().max()) if d.numel() else 0.0,
                       difference_l2=float(d.norm()), relative_l2_to_right=float(d.norm()) / ref if ref else None)
            if d.numel():
                index = tuple(int(v) for v in np.unravel_index(int(d.abs().reshape(-1).argmax()), d.shape))
                row.update(worst_index=list(index), left_worst=float(a[index]), right_worst=float(b[index]))
        return row

    def attrs(module):
        result = dict(type=type(module).__module__ + '.' + type(module).__name__, training=module.training)
        for key in ('eps', 'momentum', 'track_running_stats', 'affine', 'normalized_shape', 'num_features',
                    'alpha', 'patch', 'n_history', 'layernorm', 'p', 'inplace', 'approximate'):
            if hasattr(module, key):
                result[key] = getattr(module, key)
        return result

    class Observer:
        def __init__(self, model, x, side):
            self.model, self.x, self.side = model, x, side
            self.calls, self.call_count, self.active, self.node_aliases = {}, {}, {}, {}
            self.nodes, self.node_rows, self.backward, self.output, self.input = [], {}, {}, None, None
            self.forward_rng = None
            ddi = model.fc_blocks[0]
            self.ddi = ddi
            self.attributes = {name: attrs(module) for name, module in ddi.named_modules()}
            for name, module in ddi.named_modules():
                def pre(m, args, name=name):
                    number = self.call_count.get(name, 0)
                    self.call_count[name] = number + 1
                    key = (name or 'DDI') + '/' + str(number)
                    self.active[name] = key
                    self.calls[key] = dict(module=name, call_index=number,
                        patch_start=None if name in ('', 'norm') else 16 * (number + 1),
                        input=snapshot(args[0], f'{side}/call/{key}/input'))
                    if name == '':
                        self.input = args[0]
                        self.calls[key]['input_grad_fn'] = type(args[0].grad_fn).__name__
                    emit('module_pre', side=side, call=key, record=self.calls[key])
                def post(m, args, output, name=name):
                    key = self.active.pop(name)
                    self.calls[key]['output'] = snapshot(output, f'{side}/call/{key}/output')
                    self.calls[key]['input_layout_after_call'] = layout(args[0])
                    if output.grad_fn is not None:
                        self.node_aliases.setdefault(output.grad_fn, []).append(key)
                    checked_emit('module_post', [self.calls[key]['output']], side=side, call=key)
                    if name == '':
                        self.output = output
                handles.extend([module.register_forward_pre_hook(pre), module.register_forward_hook(post)])
            def mdm_post(m, args, output):
                self.mdm = output
                checked_emit('mdm_forward', [snapshot(output, f'{side}/mdm/output')], side=side)
                def receive(g):
                    checked_emit('mdm_total_gradient', [snapshot(g, f'{side}/mdm/total_gradient')], side=side)
                    return None
                handles.append(output.register_hook(receive))
            handles.append(model.pastmixing.register_forward_hook(mdm_post))

        def saved(self, node, node_id, phase):
            result = {}
            for name in sorted(n for n in dir(node) if n.startswith('_saved_')):
                # Read actual saved state without saved_tensors_hooks, clones or graph replacements.
                value = getattr(node, name)
                if torch.is_tensor(value):
                    result[name] = snapshot(value, f'{self.side}/node/{node_id}/{phase}/{name}')
                    result[name]['used_status'] = ('eval_backward_uses_running_statistics_not_saved_training_statistics'
                        if 'BatchNorm' in type(node).__name__ and name in ('_saved_result1', '_saved_result2')
                        and getattr(node, '_saved_training', None) is False else 'saved_state_read_only; usage_reported_with_operator')
                elif isinstance(value, (bool, int, float, str, tuple, list)) or value is None:
                    result[name] = value
                else:
                    result[name] = repr(value)
            return result

        def bind_nodes(self):
            boundary = self.input.grad_fn
            def visit(node):
                if node is None or node is boundary or node in self.node_rows:
                    return
                node_id = f'n{len(self.nodes):04d}:{type(node).__name__}'
                self.nodes.append(node)
                self.node_rows[node] = dict(id=node_id, name=node.name(), aliases=self.node_aliases.get(node, []))
                for nxt, _ in node.next_functions:
                    visit(nxt)
            visit(self.output.grad_fn)
            self.order = []
            for node in self.nodes:
                row = self.node_rows[node]
                node_id = row['id']
                row['next'] = [dict(id='DDI_INPUT_BOUNDARY' if n is boundary else self.node_rows[n]['id'] if n in self.node_rows else None,
                                    output_number=index) for n, index in node.next_functions]
                row['saved_after_forward'] = self.saved(node, node_id, 'after_forward')
                if hasattr(node, 'variable'):
                    p = node.variable
                    row['accumulated_parameter'] = next((n for n, v in self.model.named_parameters() if v is p), None)
                emit('node_bound', side=self.side, node=row)
                def before(gout, node=node, node_id=node_id):
                    saved = self.saved(node, node_id, 'before_backward')
                    rows = [snapshot(g, f'{self.side}/node/{node_id}/grad_output/{i}') for i, g in enumerate(gout)]
                    self.backward[node_id] = dict(saved_before_backward=saved, grad_output=rows)
                    checked_emit('node_backward_pre', rows, side=self.side, node_id=node_id,
                                 aliases=self.node_rows[node]['aliases'], saved=saved)
                    return None
                def after(gin, gout, node=node, node_id=node_id):
                    rows = [snapshot(g, f'{self.side}/node/{node_id}/grad_input/{i}') for i, g in enumerate(gin)]
                    self.backward[node_id]['grad_input'] = rows
                    self.order.append(node_id)
                    checked_emit('node_backward_post', rows, side=self.side, node_id=node_id,
                                 aliases=self.node_rows[node]['aliases'])
                    return None
                handles.extend([node.register_prehook(before), node.register_hook(after)])
            write_new(root / f'{self.side}-forward.json', dict(attributes=self.attributes, calls=self.calls,
                nodes=[self.node_rows[n] for n in self.nodes], counts=self.call_count,
                ddi_boundary=layout(self.input), output_layout=layout(self.output)))

    def monitor():
        while not stop.wait(.5):
            try:
                INSTANCE.sample(torch)
                size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
                if size > config['limits']['output']:
                    raise RuntimeError('diagnostic output limit')
            except BaseException as exc:
                monitor_errors.append(str(exc)); os.kill(os.getpid(), signal.SIGALRM); return

    def timeout(signum, frame):
        raise TimeoutError('diagnostic time/resource stop')

    signal.signal(signal.SIGALRM, timeout); signal.alarm(180)
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    result = dict(diagnostic_id=DIAGNOSTIC_ID, dtype='float32', device='cpu', threads=torch.get_num_threads(),
                  torch_version=torch.__version__, inherited_tests=0, real_probe=0,
                  checkpoint_reads=0, real_data_reads=0, cuda_model_forward=0, model_instances=0)
    observers = []
    try:
        INSTANCE.current('diagnostic/current_A_vs_frozen/DDI_nodes')
        common, options = AMDEnhancedTHLSTests._kwargs(512)
        runner.set_seed(2024); a = AMDEnhanced(**common, **options); result['model_instances'] += 1
        runner.set_seed(2024); f = AMD(**common, target_slice=slice(6, 7)); result['model_instances'] += 1
        a.eval(); f.eval()
        assert a.target_history_local_shape is None and a.use_target_history_local_shape is False
        assert eq(a.state_dict(), f.state_dict())
        assert type(a.fc_blocks[0]) is type(f.fc_blocks[0])
        assert a.fc_blocks[0].forward.__func__ is f.fc_blocks[0].forward.__func__
        initial = {k: v.detach().clone() for k, v in a.state_dict().items()}
        x = torch.randn(2, 512, 7, requires_grad=True); xf = x.detach().clone().requires_grad_()
        result['input'] = dict(A=snapshot(x, 'A/input'), frozen=snapshot(xf, 'frozen/input'), equal=torch.equal(x, xf))
        result['common_config'] = common
        result['options'] = {k: repr(v) for k, v in options.items()}
        ao, fo = Observer(a, x, 'A'), Observer(f, xf, 'frozen'); observers = [ao, fo]
        before = runner.capture_rng_state(); counts = dict(read(config['budget_file'])['counts'])
        grad_mode = torch.is_grad_enabled()
        pa, aa = a(x)
        result['A_forward_counter_delta'] = read(config['budget_file'])['counts']['forward'] - counts['forward']
        result['A_forward_rng_unchanged'] = eq(before, runner.capture_rng_state())
        ao.bind_nodes()
        counts = dict(read(config['budget_file'])['counts']); before = runner.capture_rng_state()
        pf, af = f(xf)
        result['frozen_forward_counter_delta'] = read(config['budget_file'])['counts']['forward'] - counts['forward']
        result['frozen_forward_rng_unchanged'] = eq(before, runner.capture_rng_state())
        fo.bind_nodes()
        result['grad_enabled_unchanged'] = grad_mode == torch.is_grad_enabled()
        result['module_attributes_equal'] = eq(ao.attributes, fo.attributes)
        la = pa.square().mean() + aa; lf = pf.square().mean() + af
        result['prediction_equal'] = torch.equal(pa, pf)
        result['aux_equal'] = torch.equal(aa, af); result['loss_equal'] = torch.equal(la, lf)
        result['forward_values'] = {side: [snapshot(p, side+'/prediction'), snapshot(aux, side+'/aux'), snapshot(loss, side+'/loss')]
                                   for side, p, aux, loss in [('A', pa, aa, la), ('frozen', pf, af, lf)]}
        write_new(root / 'pair-before-backward.json', result)
        assert all(result[k] for k in ('module_attributes_equal', 'prediction_equal', 'aux_equal', 'loss_equal',
                                       'grad_enabled_unchanged', 'A_forward_rng_unchanged', 'frozen_forward_rng_unchanged'))
        assert result['A_forward_counter_delta'] == result['frozen_forward_counter_delta'] == 1
        for side, loss, model, input_tensor in [('A', la, a, x), ('frozen', lf, f, xf)]:
            emit('backward_start', side=side)
            loss.backward()
            # Persist each side before checking, never discard a nonfinite row.
            rows = [snapshot(input_tensor.grad, side+'/input_grad')]
            rows += [snapshot(p.grad, side+'/parameter_grad/'+name) for name, p in model.named_parameters()]
            checked_emit('backward_complete', rows, side=side)
        result['parameters_buffers_unchanged'] = eq(initial, a.state_dict()) and eq(initial, f.state_dict())
        result['input_gradient'] = compare_keys('A/input_grad', 'frozen/input_grad')
        result['parameter_gradients'] = {name: compare_keys('A/parameter_grad/'+name, 'frozen/parameter_grad/'+name)
                                        for name, _ in a.named_parameters()}
        left_nodes = [ao.node_rows[n] for n in ao.nodes]; right_nodes = [fo.node_rows[n] for n in fo.nodes]
        result['graph_structure_equal'] = [(r['id'], r['next'], r['aliases']) for r in left_nodes] == [(r['id'], r['next'], r['aliases']) for r in right_nodes]
        assert result['graph_structure_equal']
        comparisons = []
        for node_id in ao.order:
            left, right = ao.backward[node_id], fo.backward[node_id]
            row = dict(node_id=node_id, aliases=next(r['aliases'] for r in left_nodes if r['id'] == node_id))
            for direction in ('grad_output', 'grad_input'):
                row[direction] = [compare_keys(v.get('key'), w.get('key')) for v, w in zip(left[direction], right[direction])]
            row['saved'] = {}
            for key, value in left['saved_before_backward'].items():
                other = right['saved_before_backward'][key]
                row['saved'][key] = compare_keys(value['key'], other['key']) if isinstance(value, dict) and 'key' in value else dict(equal=value==other)
            comparisons.append(row)
        write_new(root/'paired-nodes.json', comparisons)
        result['first_backward_difference'] = next((r for r in comparisons if not all(v['bitwise_equal'] for v in r['grad_input'])), None)
        result['status'] = 'pair_observed; requires_evidence_review_before_any_further_call'
    except BaseException:
        result['status'] = 'stopped_after_nonfinite' if isinstance(sys.exc_info()[1], ObservedNonfinite) else 'diagnostic_error'
        result['traceback'] = traceback.format_exc()
    finally:
        for observer in observers:
            write_new(root/f'{observer.side}-backward.json', dict(order=getattr(observer, 'order', []), records=observer.backward))
        for handle in handles:
            handle.remove()
        stop.set(); thread.join(timeout=2); signal.alarm(0)
        events.close()
        result['event_count'] = event_count
        result['monitor_errors'] = monitor_errors
        result['budget'] = read(config['budget_file'])
        result['this_turn_counts'] = {k: result['budget']['counts'][k]-CARRIED[k] for k in CARRIED}
        audit = Path(config['audit_log'])
        result['guard_denials'] = [r for r in (json.loads(l) for l in audit.read_text().splitlines()) if r['event']=='denied']
        verify_seal(read(config['seal_file']))
        write_new(config['report_file'], result)
        print(json.dumps({k:result[k] for k in ('status','this_turn_counts','event_count','monitor_errors')},ensure_ascii=False),flush=True)
    return 2 if result['status'].startswith('pair_observed') else 1



def operator_worker(config):
    """No observation hooks: isolate the observed GELU layout with fixed finite values."""
    from restricted_io_guard import require_installed
    from current_policy import require_scope
    from acceptance_driver import verify_seal
    from resource_budget import INSTANCE
    import math
    import torch
    require_scope(require_installed(), 'new_cuda')
    verify_seal(read(config['seal_file']))
    torch.set_num_threads(1)
    root = Path(config['session_root'])
    prior = read(config['operator_evidence'])
    assert prior['status'].startswith('pair_observed')
    node = prior['first_backward_difference']
    assert node['node_id'] == 'n1178:GeluBackward0'
    observation = read(Path(config['operator_evidence']).parent/'A-backward.json')['records'][node['node_id']]
    x_bound = observation['saved_before_backward']['_saved_self']['finite_max_abs']
    g_bound = observation['grad_output'][0]['finite_max_abs']
    result = dict(diagnostic_id=DIAGNOSTIC_ID + '/isolated_gelu_layout', models_constructed=0,
        actual_DDI_inputs_reused=False,
        input_definition='deterministic linspace within observed saved-input/cotangent ranges; observed shape/strides',
        observation_hooks=0, dtype='float32', device='cpu', threads=torch.get_num_threads(),
        approximate='none', cases=[], old_checkpoints=0, real_data=0, optimizer=0)
    def write_stats(t):
        q=t.detach().clone(); d=q.double()
        return dict(shape=list(t.shape), stride=list(t.stride()), storage_offset=t.storage_offset(),
            contiguous=t.is_contiguous(), dtype=str(t.dtype), finite=bool(torch.isfinite(q).all()),
            nan=int(torch.isnan(q).sum()), posinf=int(torch.isposinf(q).sum()), neginf=int(torch.isneginf(q).sum()),
            l2_float64=float(d.norm()) if bool(torch.isfinite(d).all()) else None,
            values=q.tolist(), sha256=hashlib.sha256(q.contiguous().numpy().tobytes()).hexdigest())
    def deadline(signum, frame):
        raise TimeoutError('isolated operator time/resource stop')
    signal.signal(signal.SIGALRM, deadline); signal.alarm(180)
    try:
        INSTANCE.current('diagnostic/no_hook_GELU_VJP_and_analytic_reference')
        base=torch.linspace(-x_bound,x_bound,224,dtype=torch.float32).reshape(2,16,7)
        cotangent=torch.linspace(-g_bound,g_bound,224,dtype=torch.float32).reshape(2,7,16).transpose(1,2)
        # Explicitly charge the independent analytic operator reference, not a free alternative API.
        INSTANCE.charge('forward', 'independent_analytic_GELU_VJP_reference')
        xs=base.flatten().tolist(); gs=cotangent.reshape(-1).tolist()
        exact=[g*(.5*(1+math.erf(x/math.sqrt(2))) + x*math.exp(-x*x/2)/math.sqrt(2*math.pi)) for x,g in zip(xs,gs)]
        reference=torch.tensor(exact,dtype=torch.float64).reshape(2,16,7)
        derivative_bound=.5*(1+math.erf(1)) + math.exp(-1)/math.sqrt(math.pi)
        result['analytic_global_abs_derivative_bound']=derivative_bound
        result['observed_original_gelu_gradient_abs_upper_bound']=g_bound*derivative_bound
        result['reference']=write_stats(reference)
        write_new(root/'operator-inputs-reference.json',dict(input=write_stats(base),cotangent=write_stats(cotangent),result=result))
        for name in ('observed_noncontiguous_cotangent', 'same_values_contiguous_cotangent_control'):
            INSTANCE.sample(torch)
            x=base.detach().clone().requires_grad_()
            g=cotangent.clone() if name.startswith('observed') else cotangent.contiguous()
            assert torch.equal(g,cotangent)
            assert list(x.stride())==[112,7,1]
            assert list(g.stride())==([112,1,16] if name.startswith('observed') else [112,7,1])
            rng=torch.get_rng_state().clone(); grad_mode=torch.is_grad_enabled()
            # Only the real existing nn.GELU and autograd VJP, without tensor/node/module observation hooks.
            module=torch.nn.GELU(approximate='none').eval()
            output=module(x)
            row=dict(case=name,input=write_stats(x),grad_output=write_stats(g),forward=write_stats(output))
            write_new(root/(name+'-before-backward.json'),row)
            if not row['forward']['finite']:
                result['cases'].append(row);result['status']='stopped_after_nonfinite_forward';break
            (gradient,)=torch.autograd.grad(output,x,grad_outputs=g)
            row['grad_input']=write_stats(gradient)
            row['rng_unchanged']=torch.equal(rng,torch.get_rng_state())
            row['grad_enabled_unchanged']=grad_mode==torch.is_grad_enabled()
            if row['grad_input']['finite']:
                delta=gradient.detach().double()-reference
                row['max_abs_vs_analytic']=float(delta.abs().max())
                row['gradient_max_abs']=float(gradient.detach().abs().max())
                row['analytic_max_abs']=float(reference.abs().max())
                row['within_analytic_global_bound']=row['gradient_max_abs'] <= g_bound*derivative_bound+1e-10
                row['analytic_close_atol_1e_10_rtol_1e_5']=bool(torch.allclose(gradient.detach().double(),reference,atol=1e-10,rtol=1e-5))
            result['cases'].append(row)
            write_new(root/(name+'-after-backward.json'),row)
            if not row['grad_input']['finite']:
                result['status']='stopped_after_nonfinite_gradient';break
            assert row['rng_unchanged'] and row['grad_enabled_unchanged']
        else:
            result['status']='isolated_operator_controls_observed_not_acceptance'
    except BaseException:
        result['status']='operator_diagnostic_error';result['traceback']=traceback.format_exc()
    finally:
        signal.alarm(0);INSTANCE.sample(torch)
        result['budget']=read(config['budget_file'])
        result['this_session_counts']={k:result['budget']['counts'][k]-config['carried_counts'][k] for k in CARRIED}
        result['this_turn_counts']={k:result['budget']['counts'][k]-CARRIED[k] for k in CARRIED}
        result['guard_denials']=[r for r in (json.loads(l) for l in Path(config['audit_log']).read_text().splitlines()) if r['event']=='denied']
        verify_seal(read(config['seal_file']))
        write_new(config['report_file'],result)
        print(json.dumps({k:result[k] for k in ('status','this_session_counts','this_turn_counts')}),flush=True)
    return 2 if result['status']=='isolated_operator_controls_observed_not_acceptance' else 1


def launch(repo, evidence, gelu_check=False):
    from acceptance_driver import validate_inputs, file_seal, verify_seal
    from run_restricted import VERSION, TOOL_ROOT
    import current_policy
    from resource_budget import initialize
    repo = Path(repo).resolve(); evidence = Path(evidence).resolve()
    assert (evidence/'starting-state.json').exists()
    validate_inputs(repo)
    prior = evidence.parent/'m4608-closed-gradient-4ifzf7n2/diagnosis-conclusion.json'
    assert hashlib.sha256(prior.read_bytes()).hexdigest() == PRIOR_SHA
    assert read(prior)['cumulative_new_workload'] == dict(forward_API=97, backward=20, Adam_step=0)
    carried = dict(CARRIED)
    if gelu_check:
        previous = read(evidence/'operator-pair-01/report.json')
        assert previous['this_turn_counts'] == dict(forward=2, backward=2, adam=0)
        carried = dict(previous['budget']['counts'])
    session = evidence/('isolated-gelu-02' if gelu_check else 'operator-pair-01')
    session.mkdir(exist_ok=False); (session/'fixtures').mkdir()
    limits = dict(forward=129, backward=26, adam=0, seconds=180, method_seconds=180,
                  rss=8*1024**3, reserved=4*1024**3, output=1024**3)
    budget = initialize(session/'budget.json', 'new_cuda/DDI_operator_diagnostic', limits)
    with budget.state() as state:
        state['counts'] = dict(carried); state['carried_counts'] = dict(carried)
        state['carried_source'] = dict(path=str(prior), sha256=PRIOR_SHA)
        state['full_authorized_limits'] = dict(forward=512, backward=64, adam=16)
        state['turn_cap'] = TURN_CAP
        if gelu_check:
            source=evidence/'operator-pair-01/report.json'
            state['additional_carried_source']=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        state['reserved_final_new18'] = dict(forward=193, backward=38, adam=16)
    seal = file_seal(repo); write_new(session/'sealed-inputs.json', seal)
    config = dict(version=VERSION, repo=str(repo), tool_root=str(TOOL_ROOT), session_root=str(session),
        diagnostic_id=DIAGNOSTIC_ID, stage='new_cuda', access_policy='synthetic_regression',
        gelu_check=gelu_check, carried_counts=carried, operator_evidence=str(evidence/'operator-pair-01/report.json'),
        restriction_policy_id=current_policy.POLICY_ID, approval_sha256=current_policy.APPROVAL_SHA,
        audit_log=str(session/'audit.jsonl'), forbidden_roots=[str(repo/'data'),str(repo/'artifacts')],
        budget_file=str(session/'budget.json'), limits=limits, seal_file=str(session/'sealed-inputs.json'),
        report_file=str(session/'report.json'), owner_pid=os.getpid(), business_bootstrap=True)
    path = session/'guard-config.json'; write_new(path, config)
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker-config', str(path)]
    write_new(session/'command.json', command)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', GIT_OPTIONAL_LOCKS='0',
        PYTHONPATH=os.pathsep.join([str(TOOL_ROOT),str(repo/'tests'),str(repo)]),
        AMD_RR_CONFIG=str(path), AMD_RR_CONFIG_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
        TMPDIR=str(session/'fixtures'), CUDA_VISIBLE_DEVICES='0')
    started = time.monotonic()
    with (session/'execution.log').open('x') as log:
        process = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_new(session/'process.json', dict(pid=process.pid, command=command))
        try:
            code = process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
            code = 124
    verify_seal(seal)
    result = dict(exit_code=code, elapsed_seconds=time.monotonic()-started,
                  budget=read(session/'budget.json'), report_exists=(session/'report.json').exists())
    write_new(session/'exit.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='budget'},ensure_ascii=False))
    return code


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--repo'); parser.add_argument('--evidence'); parser.add_argument('--worker-config'); parser.add_argument('--gelu-check',action='store_true')
    args=parser.parse_args()
    if args.worker_config:
        config=read(args.worker_config)
        raise SystemExit(operator_worker(config) if config.get('gelu_check') else worker(config))
    raise SystemExit(launch(args.repo,args.evidence,args.gelu_check))
