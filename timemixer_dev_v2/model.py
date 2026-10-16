"""Real bound author Model subclass; two localized forecast insertions only."""
from types import SimpleNamespace
import torch
from .modules import SelectiveVariableInteraction, HistoryConditionedScaleAdjustment
from .source import load_author
from .spec import validate_revision

AuthorModel = load_author().Model


class TimeMixerUV(AuthorModel):
    def __init__(self, spec):
        validate_revision(spec.value)
        v = spec.value
        super().__init__(SimpleNamespace(**v['native_options']))
        self.spec = spec
        self.use_u, self.use_v = v['use_u'], v['use_v']
        # Each optional module has its own identical CPU substream, restored on exit.
        if self.use_u:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(2024)
                self.u = SelectiveVariableInteraction(v['L'], v['C'], v['task'], v['target_idx'], v['aux_idx'])
        if self.use_v:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(2024)
                self.v = HistoryConditionedScaleAdjustment(self.configs.d_model, self.configs.down_sampling_layers + 1,
                    v['pred_len'], v['C'], v['task'], v['target_idx'], v['aux_idx'], self.channel_independence)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if not self.use_u and not self.use_v:
            return super().forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # Aligned against e246105...::Model.forecast. Future-feature branch is
        # unreachable by the structural contract (same as current task profiles).
        x_enc, x_mark_enc = self._Model__multi_scale_process_inputs(x_enc, x_mark_enc)
        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                original = x
                x = self.normalize_layers[i](x, 'norm')
                if i == 0 and self.use_u:
                    x = self.u(x, original)
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                original = x
                x = self.normalize_layers[i](x, 'norm')
                if i == 0 and self.use_u:
                    x = self.u(x, original)
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)
                enc_out_list.append(enc_out)
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)
        if self.use_v:
            dec_out = self.v(enc_out_list, dec_out_list,
                             trends=x_list[1] if self.channel_independence == 0 else None)
        else:
            dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out
