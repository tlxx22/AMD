"""Shared parent/child attempt counters; guards do not replace model behavior."""
from contextlib import contextmanager
import fcntl
import functools
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

CLOCK = time.monotonic
LOCAL = threading.local()
INSTANCE = None


class BudgetExceeded(RuntimeError):
    pass


class SharedBudget:
    def __init__(self, path, limits):
        self.path, self.limits = Path(path), dict(limits)

    @contextmanager
    def state(self):
        with self.path.open('r+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            value = json.load(handle)
            try:
                yield value
            finally:
                handle.seek(0); json.dump(value, handle, sort_keys=True); handle.truncate(); handle.flush()
                fcntl.flock(handle, fcntl.LOCK_UN)

    def charge(self, kind, detail=None):
        error = None
        with self.state() as s:
            if s.get('stopped'):
                error = s['stopped']
            elif self.limits.get(kind) is not None and s['counts'][kind] >= self.limits[kind]:
                error = f'{kind} limit reached before next attempted operation'
                s['stopped'] = error
                s.setdefault('refused_operations', []).append({'kind':kind, 'pid':os.getpid()})
            else:
                s['counts'][kind] += 1
                method = s.get('current_test', 'not_set')
                row = s.setdefault('by_test', {}).setdefault(method, {})
                row[kind] = row.get(kind, 0) + 1
                pid = str(os.getpid())
                row = s.setdefault('by_pid', {}).setdefault(pid, {})
                row[kind] = row.get(kind, 0) + 1
                if detail:
                    s.setdefault('details', {}).setdefault(kind, {})[detail] = s.setdefault('details', {}).setdefault(kind, {}).get(detail, 0) + 1
        if error:
            raise BudgetExceeded(error)

    def stop(self, reason):
        with self.state() as s:
            s['stopped'] = s.get('stopped') or str(reason)

    def current(self, test_id):
        with self.state() as s:
            s['current_test'] = test_id
            s['method_started'] = CLOCK()

    def sample(self, torch_module=None):
        with self.state() as s:
            if s.get('stopped'):
                raise BudgetExceeded(s['stopped'])
            if self.limits.get('seconds') is not None and CLOCK() - s['started'] > self.limits['seconds']:
                s['stopped'] = 'stage wall-clock limit'; raise BudgetExceeded(s['stopped'])
            fields = {}
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith(('VmRSS:', 'VmHWM:')):
                    fields[line.split(':')[0]] = int(line.split()[1]) * 1024
            rss = max(fields.values(), default=0)
            reserved = 0
            if torch_module is not None and torch_module.cuda.is_initialized():
                reserved = torch_module.cuda.max_memory_reserved()
            s['rss_peak'] = max(s.get('rss_peak', 0), rss)
            s['cuda_reserved_peak'] = max(s.get('cuda_reserved_peak', 0), reserved)
            if rss > self.limits.get('rss', float('inf')) or reserved > self.limits.get('reserved', float('inf')):
                s['stopped'] = 'memory limit'; raise BudgetExceeded(s['stopped'])


def initialize(path, stage, limits):
    value = dict(stage=stage, started=CLOCK(), counts=dict(forward=0, backward=0, adam=0),
                 by_test={}, by_pid={}, stopped=None, rss_peak=0, cuda_reserved_peak=0)
    with Path(path).open('x') as f:
        json.dump(value, f)
    return SharedBudget(path, limits)


def install(config):
    global INSTANCE
    if INSTANCE is not None:
        return INSTANCE
    INSTANCE = SharedBudget(config['budget_file'], config['limits'])
    return INSTANCE


def counted(kind, detail, function):
    @functools.wraps(function)
    def call(*args, **kwargs):
        depth = getattr(LOCAL, kind, 0)
        if depth == 0:
            INSTANCE.sample(sys.modules.get('torch'))
            INSTANCE.charge(kind, detail)
        setattr(LOCAL, kind, depth + 1)
        try:
            return function(*args, **kwargs)
        finally:
            setattr(LOCAL, kind, depth)
    return call


def instrument_module_calls(module_type):
    # PyTorch 2.0 binds __call__ = _call_impl at class definition time.
    # Patching only _call_impl leaves the existing call alias uncounted.
    original = module_type.__call__
    if getattr(original, '_amd_rr_counted', False):
        return
    wrapped = counted('forward', 'root_module_call', original)
    wrapped._amd_rr_counted = True
    module_type.__call__ = wrapped


def install_torch_hooks():
    import torch
    if getattr(torch, '_amd_rr_counted', False):
        return
    instrument_module_calls(torch.nn.Module)
    torch.autograd.backward = counted('backward', 'autograd_backward', torch.autograd.backward)
    torch.autograd.grad = counted('backward', 'autograd_grad', torch.autograd.grad)
    torch.optim.Adam.step = counted('adam', 'Adam.step', torch.optim.Adam.step)
    original = torch.nn.functional.conv1d
    def reference_conv(*args, **kwargs):
        frame = sys._getframe(1)
        # The first direct conv is the start of the independent whole-branch
        # reference in this unchanged test. All later convs are the same view.
        if (getattr(LOCAL, 'forward', 0) == 0
                and frame.f_code.co_name == 'test_reference_residual_and_components_single_dropout'
                and Path(frame.f_code.co_filename).name == 'test_target_history_local_shape_residual.py'
                and frame.f_lineno == reference_line()):
            INSTANCE.charge('forward', 'independent_functional_reference')
        return original(*args, **kwargs)
    torch.nn.functional.conv1d = reference_conv
    torch._amd_rr_counted = True


def reference_line():
    global _REFERENCE_LINE
    if '_REFERENCE_LINE' not in globals():
        import ast
        p = Path(__file__).resolve().parents[2] / 'tests/test_target_history_local_shape_residual.py'
        tree = ast.parse(p.read_bytes())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == 'test_reference_residual_and_components_single_dropout')
        _REFERENCE_LINE = min(n.lineno for n in ast.walk(method) if isinstance(n, ast.Call)
                              and isinstance(n.func, ast.Attribute) and n.func.attr == 'conv1d')
    return _REFERENCE_LINE


def attach_analysis_apis():
    for name in ('models.modules.target_history_local_shape_residual',
                 'models.modules.local_change_gated_pmcr', 'models.modules.modern_conv_refinement'):
        module = sys.modules.get(name)
        if module is None:
            continue
        for cls in vars(module).values():
            if not isinstance(cls, type) or cls.__module__ != name:
                continue
            for method in ('compute_features', 'compute_delta', 'compute_components', 'compute_gate', 'compute_base_delta'):
                if method in cls.__dict__ and not getattr(cls.__dict__[method], '_amd_counted', False):
                    raw = cls.__dict__[method]
                    is_static = isinstance(raw, staticmethod)
                    fn = counted('forward', cls.__name__ + '.' + method, raw.__func__ if is_static else raw)
                    fn._amd_counted = True
                    setattr(cls, method, staticmethod(fn) if is_static else fn)

    # The new local analytic reference is one explicitly budgeted forward/API.
    module = sys.modules.get('test_ddi_gelu_backward_layout')
    if module is not None:
        name = '_independent_gelu_vjp_reference'
        function = getattr(module, name)
        if not getattr(function, '_amd_counted', False):
            wrapped = counted('forward', 'independent_GELU_VJP_reference', function)
            wrapped._amd_counted = True
            setattr(module, name, wrapped)
