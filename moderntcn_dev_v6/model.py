"""FFN1 once -> unchanged V3 -> shared C rotation -> native FFN2 unchanged."""
import torch
from types import SimpleNamespace
from moderntcn_dev_v1.model import ModernTCNUV as _NativeWiring
from .source import load_author
from .modules import ConditionalVariableRotation, ConditionalTemporalFilter, stats


class ModernTCNUV(_NativeWiring):
    def __init__(self, spec):
        v = spec.value
        load_author().Model.__init__(self, SimpleNamespace(**v['native_options']))
        self.spec = spec
        self.use_u, self.use_v = v['use_u'], v['use_v']
        options = v['native_options']
        block = self.model.stages[0].blocks[0]
        if block.ffn2pw1.groups != v['D'] or block.ffn2pw1.in_channels != v['D'] * v['C'] or block.ffn2pw1.out_channels != v['D'] * v['C']:
            raise ValueError('shared rotation requires native ratio1 grouped C mixing')
        for enabled, offset, name, ctor in [
            (self.use_u, 17011, 'u', lambda: ConditionalVariableRotation(v['L'], v['C'], v['D'], v['target_idx'], v['aux_idx'])),
            (self.use_v, 17021, 'v', lambda: ConditionalTemporalFilter(v['L'], v['D'], v['N'], options['large_size'][0], v['C'], v['target_idx'])),
        ]:
            if enabled:
                saved = torch.random.get_rng_state()
                try:
                    torch.random.default_generator.manual_seed(v['module_init_seed'] + offset)
                    setattr(self, name, ctor())
                finally:
                    torch.random.set_rng_state(saved)

    def _block(self, H, X, Z, diagnostics):
        block = self.model.stages[0].blocks[0]
        B, C, D, N = H.shape
        x = block.dw(H.reshape(B, C * D, N)).reshape(B, C, D, N)
        x = block.norm(x.reshape(B * C, D, N)).reshape(B, C * D, N)
        x = block.ffn1drop1(block.ffn1pw1(x))
        x = block.ffn1act(x)
        h = block.ffn1drop2(block.ffn1pw2(x)).reshape(B, C, D, N)
        dv = self.v(h, Z, X, diagnostics) if self.use_v else None
        z = h + dv if self.use_v else h
        rotated = self.u(z, X, diagnostics) if self.use_u else z
        flat = rotated.permute(0, 2, 1, 3).reshape(B, D * C, N)
        a = block.ffn2pw1(flat)
        if diagnostics is not None:
            diagnostics['shared_rotation_native_mixing'] = {
                'actual_pre_activation': stats(a),
                'U_insertion': 'post_V_pre_native_ffn2pw1',
                'V_input': 'original_post_ffn1_h',
            }
        x = block.ffn2drop1(a)
        x = block.ffn2act(x)
        x = block.ffn2drop2(block.ffn2pw2(x))
        return H + x.reshape(B, D, C, N).permute(0, 2, 1, 3)
