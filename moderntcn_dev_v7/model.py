"""One RevIN; target backcast subtraction and tied forecast before denorm."""
from types import SimpleNamespace
import torch
from moderntcn_dev_v1.model import ModernTCNUV as _NativeWiring
from .source import load_author
from .modules import TiedHistoryComponent, ConditionalTemporalFilter, stats


class ModernTCNUV(_NativeWiring):
    def __init__(self, spec):
        v = spec.value
        load_author().Model.__init__(self, SimpleNamespace(**v['native_options']))
        self.spec = spec
        self.use_u, self.use_v = v['use_u'], v['use_v']
        for enabled, offset, name, ctor in [
            (self.use_u, 17011, 'u', lambda: TiedHistoryComponent(v['L'], v['C'], v['pred_len'], v['target_idx'], v['aux_idx'], v['label_horizon'], v['dataset'] == 'UrbanEV')),
            (self.use_v, 17021, 'v', lambda: ConditionalTemporalFilter(v['L'], v['D'], v['N'], v['native_options']['large_size'][0], v['C'], v['target_idx'])),
        ]:
            if enabled:
                saved = torch.random.get_rng_state()
                try:
                    torch.random.default_generator.manual_seed(v['module_init_seed'] + offset)
                    setattr(self, name, ctor())
                finally:
                    torch.random.set_rng_state(saved)

    def _block(self, H, X, Z0, diagnostics):
        block = self.model.stages[0].blocks[0]
        B, C, D, N = H.shape
        x = block.dw(H.reshape(B, C * D, N)).reshape(B, C, D, N)
        x = block.norm(x.reshape(B * C, D, N)).reshape(B, C * D, N)
        x = block.ffn1drop1(block.ffn1pw1(x))
        x = block.ffn1act(x)
        h = block.ffn1drop2(block.ffn1pw2(x)).reshape(B, C, D, N)
        dv = self.v(h, Z0, X, diagnostics) if self.use_v else None
        z = h + dv if self.use_v else h
        x = z.permute(0, 2, 1, 3).reshape(B, D * C, N)
        x = block.ffn2drop1(block.ffn2pw1(x))
        x = block.ffn2act(x)
        x = block.ffn2drop2(block.ffn2pw2(x))
        return H + x.reshape(B, D, C, N).permute(0, 2, 1, 3)

    def forward(self, x, te=None, return_diagnostics=False):
        if not self.use_u:
            return _NativeWiring.forward(self, x, te, return_diagnostics)
        if te is not None:
            raise ValueError('native second-round ModernTCN uses no extra marks')
        B, L, C = x.shape
        v = self.spec.value
        if (L, C) != (v['L'], v['C']):
            raise ValueError('history shape')
        native = self.model
        diagnostics = {} if return_diagnostics else None
        Z0 = native.revin_layer(x, 'norm').permute(0, 2, 1)
        backcast, forecast = self.u(x, Z0, diagnostics)
        Zres = Z0.clone()
        Zres[:, v['target_idx'], :] = Z0[:, v['target_idx'], :] - backcast
        stem = Zres.reshape(B * C, 1, L)
        if native.patch_size != native.patch_stride:
            stem = torch.cat((stem, stem[:, :, -1:].repeat(1, 1, native.patch_size - native.patch_stride)), -1)
        stem = native.downsample_layers[0](stem)
        D, N = stem.shape[1:]
        H = stem.reshape(B, C, D, N)
        H = self._block(H, x, Z0, diagnostics)
        y = native.head(H).permute(0, 2, 1)
        combined = y.clone()
        combined[:, :, v['target_idx']] = y[:, :, v['target_idx']] + forecast
        output = native.revin_layer(combined, 'denorm')
        if diagnostics is not None:
            diagnostics['residual_history'] = dict(target=stats(Zres[:, v['target_idx']]), V_router='original_Z0_and_X', forecast_added='normalized_target_before_single_denorm')
        return (output, diagnostics) if return_diagnostics else output
