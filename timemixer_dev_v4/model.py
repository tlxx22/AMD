"""Author TimeMixer once, then null-anchored competitive output adoption."""
from types import SimpleNamespace
import torch
from .source import load_author
from .spec import validate_revision
from .modules import SelectiveOutputResidual, PrototypeOutputCalibration, receiver_indices, competitive_fusion

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
        du = dv = delta_u = delta_v = ell_u = ell_v = u_valid = None
        if self.use_u:
            delta_u, du = self.u(x_enc, return_diagnostics=True)
            u_valid = self.u.valid_receivers(x_enc.device)
            if bool(u_valid.any()):
                ell_u = self.u.adoption_logit(x_enc, y0, delta_u, du)
            else:
                # An invalid U has no place in the denominator, even if V exists.
                delta_u = None
        if self.use_v:
            delta_v, dv = self.v(x_enc, x_mark_enc, y0, return_diagnostics=True)
            ell_v = self.v.adoption_logit(x_enc, y0, delta_v, dv)
        selected, alpha, correction = competitive_fusion(
            y0[:, :, self.receivers].transpose(1, 2),
            u=delta_u, v=delta_v, ell_u=ell_u, ell_v=ell_v, u_valid=u_valid)
        result = y0.clone()
        result[:, :, self.receivers] = selected.transpose(1, 2)
        if return_diagnostics:
            return result, dict(Y0=y0, U=du, V=dv, ell_U=ell_u, ell_V=ell_v,
                                adoption_alpha=alpha, correction=correction, U_valid=u_valid)
        return result
