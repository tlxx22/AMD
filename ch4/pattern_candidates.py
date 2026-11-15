"""PDFormer source preprocessing: deterministic train-time coverage adaptation B."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .imports import sha

RULE = 'train-legal-uniform-floor-v1'


def candidate_indices(legal, budget):
    if legal < 1 or budget < 1:
        raise ValueError('positive legal window count and candidate budget required')
    count = min(legal, budget)
    # Integer arithmetic implements floor exactly, independently of RNG/float precision.
    return np.array([k * (legal - 1) // (count - 1) for k in range(count)]
                    if count > 1 else [0], dtype=np.int64)


def build_candidates(target, indices):
    target = np.asarray(target)
    indices = np.asarray(indices)
    if target.ndim != 2 or indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu':
        raise ValueError('candidate input shape/index type')
    if indices[0] < 0 or indices[-1] + 3 > len(target) or np.any(np.diff(indices) <= 0):
        raise ValueError('candidate history bounds/order')
    # Preserve source order: time origin, node, three historical target values.
    return np.stack([target[i:i+3].T for i in indices]).reshape(-1, 3, 1)


def candidate_spec(domain, metadata, times):
    end = metadata['endpoints'][0]
    horizon, output = (12, 1) if domain == 'UrbanEV' else (1, 12) if domain.startswith('PEMS') else (1, 1)
    legal = end - 12 - horizon - output + 2
    periods = 288 if domain.startswith('PEMS') else 24
    days = 21 if domain == 'PEMS08' else 14
    indices = candidate_indices(legal, days * periods)
    train_times = np.asarray(times[:end], dtype='datetime64[ns]')
    if len(train_times) != end or np.isnat(train_times).any() or not np.all(np.diff(train_times) == np.timedelta64(5 if periods == 288 else 60, 'm')):
        raise ValueError('train time axis must have the approved continuous frequency')
    last = int(indices[-1])
    if last + 12 + horizon + output - 2 >= end:
        raise ValueError('full historical window and approved labels must stay in train')
    digest = lambda value: hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    return dict(rule=RULE, adaptation='PDFormer来源预处理的时间覆盖适配',
        formula='K=min(days*steps_per_day,L_legal); i_k=k*(L_legal-1)//(K-1); K=1 -> [0]',
        domain=domain, train_rows=end, legal_train_windows=legal, candidate_windows=len(indices),
        candidate_sequences=len(indices)*metadata['nodes'], cand_key_days=days, steps_per_day=periods,
        history_length=12, common_horizon=horizon, output_length=output, pattern_length=3,
        indices=indices.tolist(), indices_sha256=digest(indices.tolist()),
        node_order_sha256=digest(metadata['node_order']), nodes=metadata['nodes'],
        start_first=str(train_times[0]), start_last=str(train_times[last]),
        pattern_last=str(train_times[last+2]), history_last=str(train_times[last+11]),
        latest_label=str(train_times[last+12+horizon+output-2]), train_last=str(train_times[-1]),
        seed=2024, index_selection_uses_rng=False, train_only=True,
        candidate_order='time_origin,node,history_position; no filtering or extra normalization')


def candidate_identity(domain, metadata_path):
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    times_path = metadata_path.parent/'times.npy'
    if sha(times_path) != metadata['files']['times.npy']:
        raise ValueError('candidate time source changed')
    # mmap only the train slice; later split values are not decoded for fitting.
    spec = candidate_spec(domain, metadata, np.load(times_path, mmap_mode='r', allow_pickle=False))
    return dict(selection=spec, metadata=dict(path=str(metadata_path), sha256=sha(metadata_path)),
        inputs={key:dict(path=str(metadata_path.parent/(key+'.npy')), sha256=metadata['files'][key+'.npy'])
                for key in ('y', 'valid', 'times')}, producer_sha256=sha(__file__))


def validate_candidate_identity(reference, domain, metadata_path):
    path = Path(reference['path'])
    if sha(path) != reference['sha256']:
        raise ValueError('candidate identity bytes changed')
    actual = json.loads(path.read_text())
    if actual != candidate_identity(domain, metadata_path):
        raise ValueError('candidate identity contract/source changed')
    return actual['selection']
