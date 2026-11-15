"""Isolated split execution of frozen v10, final-H spatial update, shared head/U readout."""
from types import SimpleNamespace
import json
from pathlib import Path
import torch
from torch import nn
from moderntcn_dev_v10.model import ModernTCNUV
from .spatial import SpatialResponse


def temporal_spec(channels, pred_len, target_idx=0, seed=2024):
    if channels < 1 or pred_len not in (1, 12, 24) or not 0 <= target_idx < channels:
        raise ValueError('explicit C/target and P1/P12/P24 required')
    profiles = json.loads((Path(__file__).resolve().parents[1] / 'configs/ch3_baseline_ms_u96_oc01_v3.json').read_text())['resolved_profiles']
    base = next(p for p in profiles.values() if p['model'] == 'ModernTCN' and p['dataset'] == 'UrbanEV' and p['H'] == 3)
    options = dict(base['structure'])
    options.update(task_name='long_term_forecast', seq_len=12, pred_len=pred_len,
                   enc_in=channels, dec_in=channels, c_out=channels, label_len=0, features='M',
                   num_blocks=[1], dims=[64]*4, ffn_ratio=1,
                   patch_size=1, patch_stride=1, large_size=[7], small_size=[3],
                   small_kernel_merged=False, dropout=.1, head_dropout=0.,
                   revin=1, affine=0, subtract_last=0, use_multi_scale=False,
                   decomposition=0, individual=0,
                   kernel_size=25, stem_ratio=6)
    return SimpleNamespace(value=dict(L=12, C=channels, D=64, N=12, pred_len=pred_len,
                                      target_idx=target_idx, use_u=True, use_v=True,
                                      module_init_seed=seed, native_options=options))


class Chapter4Model(nn.Module):
    def __init__(self, channels, pred_len, nodes, target_idx=0, spatial=True, seed=2024):
        super().__init__()
        self.temporal = ModernTCNUV(temporal_spec(channels, pred_len, target_idx, seed))
        self.spatial = SpatialResponse(nodes, seed) if spatial else None
        self.nodes = nodes
        self.target_idx = target_idx

    def forward(self, x, geo=None, sem=None, available=None, node_indices=None,
                spatial_disabled=False, return_diagnostics=False):
        if x.ndim != 4:
            raise ValueError('expected synchronous [B,L,N,C]')
        b, l, n, c = x.shape
        v = self.temporal.spec.value
        if (l, c, n) != (v['L'], v['C'], self.nodes):
            raise ValueError('task shape changed')
        flat = x.permute(0, 2, 1, 3).reshape(b*n, l, c)
        native = self.temporal.model
        diagnostics = {} if return_diagnostics else None
        z = native.revin_layer(flat, 'norm').permute(0, 2, 1)
        stem = native.downsample_layers[0](z.reshape(b*n*c, 1, l))
        h = stem.reshape(b*n, c, 64, 12)
        h = self.temporal._block(h, flat, z, diagnostics)
        before = h
        valid = None
        if self.spatial is not None and not spatial_disabled:
            if geo is None or sem is None:
                raise ValueError('bound geographic and semantic supports required')
            if available is None:
                raise ValueError('explicit target availability required')
            if available.shape != (b, l, n) or available.dtype != torch.bool:
                raise ValueError('availability must be bool [B,L,N]')
            target = flat[:, :, self.target_idx].float()
            mean = target.mean(1)
            var = (target - mean[:, None]).square().mean(1)
            state = torch.stack((mean, torch.log1p(var)), -1).reshape(b, n, 2)
            ht = h[:, self.target_idx].reshape(b, n, 64, 12).permute(0, 3, 1, 2)
            delta, valid = self.spatial(ht, state, geo, sem, available, node_indices)
            h = h.clone()
            h[:, self.target_idx] = (ht + delta).permute(0, 2, 3, 1).reshape(b*n, 64, 12)
        y0 = native.head(h).permute(0, 2, 1)
        increment = self.temporal.u(h[:, self.target_idx].reshape(b*n, 64*12), diagnostics)
        y = y0.clone()
        y[:, :, self.target_idx] = y0[:, :, self.target_idx] + increment
        y = native.revin_layer(y, 'denorm')
        prediction = y[:, :, self.target_idx].reshape(b, n, v['pred_len']).permute(0, 2, 1)
        if return_diagnostics:
            diagnostics.update(H_before=before, H_after=h, spatial_valid=valid)
            return prediction, diagnostics
        return prediction
