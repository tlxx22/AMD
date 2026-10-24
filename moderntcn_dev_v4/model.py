"""Native FFN1 once -> V3(h,Z,X) -> auxiliary-content U(h+deltaV,X)."""
import torch
from types import SimpleNamespace
from moderntcn_dev_v1.model import ModernTCNUV as _NativeWiring
from .source import load_author
from .modules import AuxiliaryPatchSelection, ConditionalTemporalFilter, stats


class ModernTCNUV(_NativeWiring):
    def __init__(self, spec):
        v = spec.value
        load_author().Model.__init__(self, SimpleNamespace(**v['native_options']))
        self.spec = spec
        self.use_u, self.use_v = v['use_u'], v['use_v']
        options = v['native_options']
        for enabled, offset, name, ctor in [
            (self.use_u, 17011, 'u', lambda: AuxiliaryPatchSelection(v['L'], v['C'], v['D'], v['N'], options['patch_size'], options['patch_stride'], v['target_idx'], v['aux_idx'])),
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
        x = block.dw(H.reshape(B, C*D, N)).reshape(B, C, D, N)
        x = block.norm(x.reshape(B*C, D, N)).reshape(B, C*D, N)
        x = block.ffn1drop1(block.ffn1pw1(x)); x = block.ffn1act(x)
        h = block.ffn1drop2(block.ffn1pw2(x)).reshape(B, C, D, N)
        delta_v = self.v(h, Z, X, diagnostics) if self.use_v else None
        h_v = h + delta_v if self.use_v else h
        s = self.u(h_v, X, diagnostics) if self.use_u else h_v
        if diagnostics is not None:
            diagnostics['v_then_auxiliary_content'] = {
                'axes': 'B,C,D,N', 'h': stats(h), 'h_v': stats(h_v),
                'delta_V': stats(delta_v) if delta_v is not None else None,
                'merged': stats(s), 'merged_delta': stats(s-h),
                'V_filter_input': 'post_ffn1_h', 'V_router_input': 'post_RevIN_Z_and_pre_RevIN_target_state',
                'U_context': 'post_V_target_slot', 'U_values': 'pre_RevIN_ordered_aux_history_patches',
            }
        x = s.permute(0, 2, 1, 3).reshape(B, D*C, N)
        x = block.ffn2drop1(block.ffn2pw1(x)); x = block.ffn2act(x)
        x = block.ffn2drop2(block.ffn2pw2(x))
        return H + x.reshape(B, D, C, N).permute(0, 2, 1, 3)
