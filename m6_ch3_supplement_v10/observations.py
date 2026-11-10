"""Deterministic case selection and streaming metrics, without model imports."""
import numpy as np
import pandas as pd


def select_nodes(train_volume, node_order):
    x = np.asarray(train_volume, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(node_order) or len(set(node_order)) != len(node_order):
        raise ValueError('time/node shape or duplicate node identity')
    mu, std = x.mean(0), x.std(0)
    valid = np.isfinite(mu) & np.isfinite(std) & (mu > 0)
    if valid.sum() < 2:
        raise ValueError('insufficient eligible nodes')
    low, high = np.quantile(mu[valid], [.25, .75])
    candidates = np.flatnonzero(valid & (mu >= low) & (mu <= high))
    if len(candidates) < 2:
        raise ValueError('insufficient middle-demand nodes')
    cv = std[candidates] / mu[candidates]
    targets = np.quantile(cv, [.25, .75])
    selected = []
    for target in targets:
        available = [i for i in range(len(candidates)) if int(candidates[i]) not in selected]
        i = min(available, key=lambda j: (abs(cv[j]-target), int(candidates[j])))
        selected.append(int(candidates[i]))
    return dict(indices=selected, nodes=[node_order[i] for i in selected],
                train_mean=mu.tolist(), train_std=std.tolist(), mean_bounds=[float(low), float(high)],
                eligible=candidates.tolist(), cv_targets=targets.tolist(), rule='train-only, population std, original node_order ties')


def case_times(common_target_times, H=6):
    times = pd.DatetimeIndex(common_target_times)
    if H != 6 or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError('fixed H6 ordered unique target times required')
    candidates = [i for i, t in enumerate(times) if t.dayofweek == 0 and t.hour == 0 and t.minute == 0]
    def complete(i):
        return i+168 <= len(times) and times[i:i+168].equals(pd.date_range(times[i], periods=168, freq='h'))
    monday = next((i for i in candidates if complete(i)), None)
    start = monday if monday is not None else next((i for i in range(len(times)) if complete(i)), None)
    if start is None:
        raise ValueError('no complete common 168-hour validation window')
    target = times[start:start+168]
    return dict(target_times=target.astype(str).tolist(), origins=(target-pd.Timedelta(hours=H)).astype(str).tolist(),
                reason='first full Monday week' if monday is not None else 'earliest complete common week; no full Monday week', H=H)


class PositionErrors:
    def __init__(self, P):
        self.P = P
        self.sse = np.zeros(P, dtype=np.float64)
        self.sae = np.zeros(P, dtype=np.float64)
        self.signed = np.zeros(P, dtype=np.float64)
        self.count = 0

    def add(self, prediction, truth):
        a, b = np.asarray(prediction), np.asarray(truth)
        if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (self.P, 1):
            raise ValueError('exact [B,P,1] prediction/target shape required')
        error = (a-b).astype(np.float64)
        if not np.isfinite(error).all():
            raise ValueError('nonfinite errors')
        self.sse += np.square(error).sum((0, 2))
        self.sae += np.abs(error).sum((0, 2))
        self.signed += error.sum((0, 2))
        self.count += len(a)

    def result(self):
        if not self.count:
            raise ValueError('empty coverage')
        return dict(count_per_position=self.count, sse=self.sse.tolist(), sae=self.sae.tolist(),
                    mse=(self.sse/self.count).tolist(), mae=(self.sae/self.count).tolist(),
                    signed_bias=(self.signed/self.count).tolist(),
                    overall_mse=float(self.sse.sum()/(self.count*self.P)),
                    overall_mae=float(self.sae.sum()/(self.count*self.P)))
