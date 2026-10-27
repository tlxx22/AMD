"""Reuse exact reviewed v9 native/V/head execution; only select U by P."""
from types import SimpleNamespace
import torch
from moderntcn_dev_v9.model import ModernTCNUV as _V9Execution
from .source import load_author
from .modules import SplineTargetReadout, HorizonCoupledSplineReadout, ConditionalTemporalFilter


class ModernTCNUV(_V9Execution):
    def __init__(self, spec):
        v = spec.value
        load_author().Model.__init__(self, SimpleNamespace(**v['native_options']))
        self.spec = spec
        self.use_u, self.use_v = v['use_u'], v['use_v']
        for enabled, offset, name, ctor in [
            (self.use_u, 17011, 'u', lambda: (SplineTargetReadout if v['pred_len']==1 else HorizonCoupledSplineReadout)(v['D']*v['N'], v['pred_len'])),
            (self.use_v, 17021, 'v', lambda: ConditionalTemporalFilter(v['L'], v['D'], v['N'], v['native_options']['large_size'][0], v['C'], v['target_idx'])),
        ]:
            if enabled:
                saved = torch.random.get_rng_state()
                try:
                    torch.random.default_generator.manual_seed(v['module_init_seed'] + offset)
                    setattr(self, name, ctor())
                finally:
                    torch.random.set_rng_state(saved)

