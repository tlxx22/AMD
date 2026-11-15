"""Train-only supports; padding is -1, never an additional node."""
import numpy as np


def neighbor_table(rows):
    width = max(map(len, rows))
    out = np.full((len(rows), width), -1, dtype=np.int64)
    for i, row in enumerate(rows):
        out[i, :len(row)] = sorted(row)
    return out


def geographic_neighbors(distance, limit=8):
    distance = np.asarray(distance, dtype=np.float64)
    n = len(distance)
    if distance.shape != (n, n) or np.isnan(distance).any() or (distance < 0).any():
        raise ValueError('nonnegative square distances (infinity allowed) required')
    rows = [{i} for i in range(n)]
    for i in range(n):
        candidates = [j for j in range(n) if j != i and np.isfinite(distance[i, j])]
        for j in sorted(candidates, key=lambda j: (distance[i, j], j))[:limit]:
            rows[i].add(j)
            rows[j].add(i)
    return neighbor_table(rows)


def physical_shortest_paths(nodes, edges):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    lengths = {}
    for i, j, d in edges:
        i, j = int(i), int(j)
        if not 0 <= i < nodes or not 0 <= j < nodes or not np.isfinite(d) or d < 0:
            raise ValueError('invalid physical road edge')
        if i != j:
            key = (min(i, j), max(i, j))
            lengths[key] = min(lengths.get(key, np.inf), float(d))
    rows, cols, vals = [], [], []
    for (i, j), d in lengths.items():
        rows.extend((i, j)); cols.extend((j, i)); vals.extend((d, d))
    return shortest_path(csr_matrix((vals, (rows, cols)), shape=(nodes, nodes)), directed=False)


def semantic_neighbors(target, valid, timestamps, limit=8):
    """Only train observations. Subhourly observations first averaged per clock-hour."""
    target = np.asarray(target, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    timestamps = np.asarray(timestamps, dtype='datetime64[ns]')
    if target.ndim != 2 or valid.shape != target.shape or len(timestamps) != len(target):
        raise ValueError('train target/mask/time axes')
    if not np.isfinite(target[valid]).all():
        raise ValueError('valid target values must be finite')
    hours = timestamps.astype('datetime64[h]')
    unique, inverse = np.unique(hours, return_inverse=True)
    sums = np.zeros((len(unique), target.shape[1])); counts = np.zeros_like(sums)
    np.add.at(sums, inverse, np.where(valid, target, 0))
    np.add.at(counts, inverse, valid.astype(float))
    hourly = sums / np.maximum(counts, 1)
    profile = np.zeros((target.shape[1], 24))
    missing = []
    hour_of_day = unique.astype(np.int64) % 24
    for h in range(24):
        ok = counts[hour_of_day == h] > 0
        count = ok.sum(0)
        if np.any(count == 0):
            missing.extend((int(i), h) for i in np.flatnonzero(count == 0))
        profile[:, h] = np.where(ok, hourly[hour_of_day == h], 0).sum(0) / np.maximum(count, 1)
    if missing:
        raise ValueError('train-only semantic hourly coverage missing: ' + repr(missing))
    centered = profile - profile.mean(1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1)
    constant = norm <= 1e-12 * np.maximum(1, np.linalg.norm(profile, axis=1))
    normalized = centered / np.where(constant, 1, norm)[:, None]
    rows = []
    for i in range(len(profile)):
        sim = normalized @ normalized[i]
        candidates = [] if constant[i] else [j for j in range(len(profile)) if j != i and not constant[j] and sim[j] > 0]
        rows.append({i, *sorted(candidates, key=lambda j: (-sim[j], j))[:limit]})
    return neighbor_table(rows), dict(profile=profile, constant=constant)
