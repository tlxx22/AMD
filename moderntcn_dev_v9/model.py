"""Unchanged native/V3 block; spline target readout after native head, before denorm."""
from types import SimpleNamespace
import torch
from moderntcn_dev_v1.model import ModernTCNUV as _NativeWiring
from .source import load_author
from .modules import SplineTargetReadout, ConditionalTemporalFilter


class ModernTCNUV(_NativeWiring):
    def __init__(self, spec):
        v = spec.value
        load_author().Model.__init__(self, SimpleNamespace(**v['native_options']))
        self.spec = spec
        self.use_u, self.use_v = v['use_u'], v['use_v']
        for enabled, offset, name, ctor in [
            (self.use_u, 17011, 'u', lambda: SplineTargetReadout(v['D']*v['N'], v['pred_len'])),
            (self.use_v, 17021, 'v', lambda: ConditionalTemporalFilter(v['L'], v['D'], v['N'], v['native_options']['large_size'][0], v['C'], v['target_idx'])),
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
        z = h + self.v(h, Z, X, diagnostics) if self.use_v else h
        s = z
        x = s.permute(0, 2, 1, 3).reshape(B, D * C, N)
        x = block.ffn2drop1(block.ffn2pw1(x))
        x = block.ffn2act(x)
        x = block.ffn2drop2(block.ffn2pw2(x))
        return H + x.reshape(B, D, C, N).permute(0, 2, 1, 3)

    def forward(self, x, te=None, return_diagnostics=False):
        if not self.use_u:
            return super().forward(x, te, return_diagnostics)
        if te is not None:
            raise ValueError('ModernTCN does not consume marks')
        native = self.model
        B, L, C = x.shape
        v = self.spec.value
        if (L, C) != (v['L'], v['C']):
            raise ValueError('input L/C mismatch')
        diagnostics = {} if return_diagnostics else None
        Z = native.revin_layer(x, 'norm').permute(0, 2, 1)
        stem = Z.reshape(B*C, 1, L)
        patch, stride = v['native_options']['patch_size'], v['native_options']['patch_stride']
        if patch != stride:
            stem = torch.cat([stem, stem[:, :, -1:].repeat(1, 1, patch-stride)], dim=-1)
        stem = native.downsample_layers[0](stem)
        H = stem.reshape(B, C, v['D'], v['N'])
        H = self._block(H, x, Z, diagnostics)
        y0 = native.head(H).permute(0, 2, 1)
        a = H[:, v['target_idx'], :, :].reshape(B, v['D']*v['N'])
        delta = self.u(a, diagnostics)
        y = y0.clone()
        y[:, :, v['target_idx']] = y0[:, :, v['target_idx']] + delta
        y = native.revin_layer(y, 'denorm')
        return (y, diagnostics) if return_diagnostics else y
