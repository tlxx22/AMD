"""Author TimeMixer once, then parallel bounded output corrections."""
from types import SimpleNamespace
import torch
from .source import load_author
from .spec import validate_revision
from .modules import SelectiveOutputResidual, PrototypeOutputCalibration, receiver_indices

AuthorModel = load_author().Model


class TimeMixerUV(AuthorModel):
    def __init__(self, spec):
        v = validate_revision(spec.value)
        super().__init__(SimpleNamespace(**v['native_options']))
        self.spec = spec
        self.use_u, self.use_v = v['use_u'], v['use_v']
        self.receivers = receiver_indices(v['C'], v['task'], v['target_idx'], v['aux_idx'])
        # CPU-only child substreams preserve the backbone and external RNG state.
        if self.use_u:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(v['module_init_seed'])
                self.u = SelectiveOutputResidual(v['L'], v['pred_len'], v['C'], v['task'], v['target_idx'], v['aux_idx'])
        if self.use_v:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(v['module_init_seed'])
                self.v = PrototypeOutputCalibration(v['L'], v['pred_len'], v['C'], v['time_mark']['K'], v['task'], v['target_idx'], v['aux_idx'])

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, *, return_diagnostics=False):
        # This delegates the whole bound author computation, including one denorm.
        # Never inherit the historical v1/v2 enhanced forecast.
        y0 = super().forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if not self.use_u and not self.use_v:
            return (y0, dict(Y0=y0)) if return_diagnostics else y0
        result = y0.clone()
        du = dv = None
        if self.use_u:
            delta_u, du = self.u(x_enc, return_diagnostics=True) if return_diagnostics else (self.u(x_enc), None)
            result[:, :, self.receivers] = result[:, :, self.receivers] + delta_u.transpose(1, 2)
        if self.use_v:
            delta_v, dv = self.v(x_enc, x_mark_enc, y0, return_diagnostics=True) if return_diagnostics else (self.v(x_enc, x_mark_enc, y0), None)
            result[:, :, self.receivers] = result[:, :, self.receivers] + delta_v.transpose(1, 2)
        return (result, dict(Y0=y0, U=du, V=dv)) if return_diagnostics else result
