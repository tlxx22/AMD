"""Explicit split/window/scaling contracts. No data is loaded at import."""
import numpy as np
import torch
from torch.utils.data import Dataset


def batch_size(domain):
    from .protocol import dataset_protocol
    return dataset_protocol(domain)['batch']


def fit_scaler(train, valid, shared_features=()):
    if train.ndim != 3 or valid.shape != train.shape:
        raise ValueError('expected [T,N,C] and independent mask')
    means, scales = [], []
    for c in range(train.shape[-1]):
        x, mask = train[:, :, c], valid[:, :, c]
        if c in shared_features:
            if not np.array_equal(mask, np.broadcast_to(mask[:, :1], mask.shape)) or not np.allclose(x, x[:, :1], equal_nan=True):
                raise ValueError('shared weather must be identical across nodes')
            x, mask = x[:, 0], mask[:, 0]
        values = x[mask]
        if not len(values) or not np.isfinite(values).all():
            raise ValueError('undefined/nonfinite training statistics')
        mean, std = values.mean(dtype=np.float64), values.std(dtype=np.float64, ddof=0)
        means.append(mean); scales.append(std if std > 0 else 1.)
    return np.asarray(means), np.asarray(scales)


def causal_fill(values, valid, train_mean):
    out = np.array(values, dtype=np.float64, copy=True)
    if valid.shape != out.shape:
        raise ValueError('independent feature mask required')
    previous = np.broadcast_to(train_mean, out.shape[1:]).copy()
    for t in range(len(out)):
        previous = np.where(valid[t], out[t], previous)
        out[t] = previous
    if not np.isfinite(out).all():
        raise ValueError('causal fill did not yield finite input')
    return out


def calendar(times):
    times = np.asarray(times, dtype='datetime64[ns]')
    days = times.astype('datetime64[D]').astype(np.int64)
    seconds = (times - times.astype('datetime64[D]')).astype('timedelta64[s]').astype(np.int64)
    return np.stack((seconds / 86400., (days + 3) % 7), -1).astype(np.float32)


class GraphWindows(Dataset):
    def __init__(self, x, target, target_valid, timestamps, start, stop, length=12,
                 prediction=1, offset=1, history_within_split=False):
        self.x, self.target, self.valid = x, target, target_valid
        self.times = np.asarray(timestamps, dtype='datetime64[ns]')
        self.length, self.prediction, self.offset = length, prediction, offset
        from utils.ch3_time_marks import marks
        frequency='t' if len(self.times)>1 and self.times[1]-self.times[0]==np.timedelta64(5,'m') else 'h'
        import pandas as pd
        self.marks=marks(pd.DatetimeIndex(self.times),frequency)
        # origin is the first time after the history window. Urban uses split-local windows.
        first = start + length if history_within_split else max(length, start - offset + 1)
        last = stop - offset - prediction + 1
        self.origins = np.arange(first, max(first, last + 1), dtype=np.int64)
        if len(self.times) < stop or len(x) < stop:
            raise ValueError('requested split exceeds accessible prefix')
        if x.shape[:2] != target.shape or target_valid.shape != target.shape:
            raise ValueError('synchronous node/target axes')

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        o = int(self.origins[index]); a = o + self.offset - 1; z = a + self.prediction
        return dict(x=torch.from_numpy(np.array(self.x[o-self.length:o], dtype=np.float32,copy=True)),
                    y=torch.from_numpy(np.array(self.target[a:z], dtype=np.float32,copy=True)),
                    mask=torch.from_numpy(np.array(self.valid[a:z], dtype=bool,copy=True)),
                    available=torch.from_numpy(np.array(self.valid[o-self.length:o], dtype=bool,copy=True)),
                    history_calendar=torch.from_numpy(calendar(self.times[o-self.length:o])),
                    time_marks=torch.from_numpy(self.marks[o-self.length:o]),
                    future_calendar=torch.from_numpy(calendar(self.times[a:z])), origin=o)


def masked_mse(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != target.shape or mask.dtype != torch.bool:
        raise ValueError('prediction/target/mask shape or dtype')
    if not mask.any():
        raise ValueError('batch has no valid supervision')
    # Select before arithmetic so missing labels do not inject NaN gradients.
    error = prediction[mask] - target[mask]
    if not torch.isfinite(error).all():
        raise ValueError('nonfinite valid prediction or target')
    return error.square().mean()


class MetricTotals:
    def __init__(self):
        self.sse = 0.; self.sae = 0.; self.count = 0

    def update(self, prediction, target, mask):
        error = (prediction[mask].double() - target[mask].double())
        if not torch.isfinite(error).all():
            raise ValueError('nonfinite metric values')
        self.sse += float(error.square().sum()); self.sae += float(error.abs().sum()); self.count += error.numel()

    def result(self):
        if not self.count:
            raise ValueError('empty evaluation')
        return dict(mse=self.sse/self.count, mae=self.sae/self.count, sse=self.sse, sae=self.sae, count=self.count)
