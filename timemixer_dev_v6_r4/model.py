"""Versioned author forecast: final PDM V, then finest native-head U gate.

Author TimeMixer Model at e2461058 is unchanged. The off/off path delegates
straight to that author; the enabled copy preserves its CI1 marks repeat and S1 limitation.
"""
from types import SimpleNamespace
import torch
from .source import load_author
from .spec import validate_revision
from .modules import TargetConditionedAuxBasisGate, TemporalQKRawValueOperator, gated_native_projection
AuthorModel = load_author().Model

class TimeMixerUV(AuthorModel):
    def __init__(self, spec):
        v = validate_revision(spec.value)
        super().__init__(SimpleNamespace(**v['native_options']))
        self.spec = spec
        self.use_u, self.use_v = v['use_u'], v['use_v']
        if self.use_u:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(v['module_init_seed'])
                self.u = TargetConditionedAuxBasisGate(v['L'], v['native_options']['d_model'], v['C'], v['task'], v['target_idx'], v['aux_idx'])
        if self.use_v:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(v['module_init_seed'])
                self.v = TemporalQKRawValueOperator(v['native_options']['d_model'])

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, *, return_diagnostics=False):
        if not self.use_u and not self.use_v:
            result = super().forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return (result, dict(original_path=True)) if return_diagnostics else result
        raw_x = x_enc
        diagnostics = {}

        if self.use_future_temporal_feature:
            if self.channel_independence == 1:
                B, T, N = x_enc.size()
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
            else:
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)

        x_enc, x_mark_enc = self._Model__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # V refines only current last-PDM finest, before original temporal prediction.
        if self.use_v:
            value = self.v(enc_out_list[0], return_diagnostics=return_diagnostics)
            enc_out_list[0], details = value if return_diagnostics else (value, None)
            if return_diagnostics: diagnostics['V'] = details

        # Future Multipredictor Mixing as decoder for future
        if self.use_u and self.u.has_aux:
            value = self.u(raw_x, return_diagnostics=return_diagnostics, return_logits=True)
            gate, details = value if return_diagnostics else (value, None)
            if return_diagnostics: diagnostics['U'] = details
            dec_out_list = self._future_with_native_gate(B, enc_out_list, x_list, gate, diagnostics if return_diagnostics else None)
        else:
            dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)
        if return_diagnostics: diagnostics['scale_predictions'] = dec_out_list

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return (dec_out, diagnostics) if return_diagnostics else dec_out


    def _future_with_native_gate(self, batch_size, enc_out_list, x_list, gate, diagnostics=None):
        """Author future mixing; gate argument is raw controller logits for stable r4 interface.

        All coarse scales and the native trend bypass keep their author operators.
        Bias is never multiplied by the gate. No separate learned output head.
        """
        result = []
        if self.channel_independence == 1:
            season = x_list[0]
            for i, enc in zip(range(len(season)), enc_out_list):
                carrier = self.predict_layers[i](enc.permute(0, 2, 1)).permute(0, 2, 1)
                if self.use_future_temporal_feature:
                    carrier = carrier + self.x_mark_dec
                if i == 0:
                    value = gated_native_projection(carrier, self.projection_layer, gate,
                        self.u.receivers, batch_size=batch_size, channels=self.spec.value['C'],
                        channel_independence=1, diagnostics=(diagnostics.setdefault('U_interface', []) if diagnostics is not None else None))
                else:
                    value = self.projection_layer(carrier)
                value = value.reshape(batch_size, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                result.append(value)
        else:
            for i, enc, trend in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                carrier = self.predict_layers[i](enc.permute(0, 2, 1)).permute(0, 2, 1)
                if i == 0:
                    value = gated_native_projection(carrier, self.projection_layer, gate,
                        self.u.receivers, batch_size=batch_size, channels=self.spec.value['C'],
                        channel_independence=0, diagnostics=(diagnostics.setdefault('U_interface', []) if diagnostics is not None else None))
                    residual = self.out_res_layers[i](trend.permute(0, 2, 1))
                    residual = self.regression_layers[i](residual).permute(0, 2, 1)
                    value = value + residual
                else:
                    value = self.out_projection(carrier, i, trend)
                result.append(value)
        return result
