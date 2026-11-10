"""Frozen producer-aware validation inference, entered only by reviewed runtime."""
import copy
from pathlib import Path
import numpy as np
import torch
from .contract import read, load_ref, sha, write_new
from .observations import select_nodes, case_times, PositionErrors
from .benchmark import measure
from .gpu_identity import check_gpu_identity, measure_with_identity
from utils.ch3_time_marks import batch_parts


def source(config, job):
    entries = load_ref(config['ledger_ref'])['entries']
    def match(e):
        t = e['task']
        if t['dataset'] != job['dataset'] or t['h'] != (6 if job['dataset'] == 'UrbanEV' else 24):
            return False
        if job['model'] == 'UV':
            return e['kind'] != 'baseline' and t.get('arm') == 'UV'
        return e['kind'] == 'baseline' and t['model'] == job['model']
    found = [e for e in entries if match(e)]
    if len(found) != 1:
        raise ValueError('missing/duplicate frozen source')
    return found[0]


def datasets(row):
    from utils.ch3_data import load
    task, p = row['task'], row['profile']
    c = load_ref(row['protocol_ref']) if row['kind'] == 'baseline' else task['data_config']
    sets, metadata = load(c, task, resolved_profile=p)
    for k in ('scaler', 'window_counts', 'node_order', 'prefixes', 'record_prefix_sha256'):
        if k in row['data'] and metadata.get(k) != row['data'][k]:
            raise ValueError('frozen data/scaler identity changed: ' + k)
    if 'test' in sets or metadata.get('test_observations_accessed'):
        raise ValueError('read-only validation cannot read test')
    return sets, metadata


def prepare_case(config, folder):
    """Must precede every model prediction in this attempt."""
    row = source(config, dict(dataset='UrbanEV', model='ModernTCN'))
    sets, meta = datasets(row)
    scale = np.asarray(meta['scaler']['volume_scale'])
    mean = np.asarray(meta['scaler']['volume_mean'])
    train = sets['train']
    raw = train.x[:train.end, :, 0] * scale + mean
    selection = select_nodes(raw, meta['node_order'])
    validation = sets['validation']
    import pandas as pd
    if set(meta['interval_counts']) != {'0 days 01:00:00'}:
        raise ValueError('case requires verified hourly clock')
    offset = validation.start + validation.T + validation.label_horizon - 1
    times = pd.date_range(pd.Timestamp(meta['time_first'])+pd.Timedelta(hours=offset), periods=validation.count, freq='h')
    selection.update(case_times(times))
    lookup = {str(t): i for i, t in enumerate(times)}
    selection['dataset_indices'] = [lookup[t]*validation.nodes+n for n in selection['indices'] for t in selection['target_times']]
    selection['scaler'] = meta['scaler']
    selection['source_manifest'] = row['manifest_ref']
    from .contract import verify_case_selection
    verify_case_selection(config, selection)
    write_new(Path(folder)/'case-selection.json', selection)
    if config.get('technical_continuation'):
        if sha(Path(folder)/'case-selection.json') != config['technical_continuation']['case_selection_ref']['sha256']:
            raise ValueError('rebuilt case selection bytes differ from sealed failed attempt')
    return selection


def restore(row, device):
    if sha(row['weights_ref']['path']) != row['weights_ref']['sha256']:
        raise ValueError('strict-best weight checksum')
    state = torch.load(row['weights_ref']['path'], map_location='cpu')
    if state['identity'] != row['identity']:
        raise ValueError('producer checkpoint identity mismatch')
    p, t = row['profile'], row['task']
    if row['kind'] == 'baseline':
        from models.ch3_adapter import build, target_prediction
        c = load_ref(row['protocol_ref'])
        from utils.ch3_contract import profile
        if profile(c, t) != p:
            raise ValueError('baseline source profile changed')
        model = build(c, t)
        predict = lambda x, marks: target_prediction(model, x, p, marks)[0]
    else:
        # Keep actual producer v9 for P1 references, v10 for fresh EPF.
        if row['identity']['candidate_id'] == 'moderntcn-uv-dev-v9':
            from moderntcn_dev_v9 import adapter
        elif row['identity']['candidate_id'] == 'moderntcn-uv-dev-v10':
            from moderntcn_dev_v10 import adapter
        else:
            raise ValueError('unexpected full UV producer')
        model = adapter.build(p, use_u=True, use_v=True, task='MS', fold=t['fold'])
        predict = lambda x, marks: adapter.prediction(model, x, marks)
    expected = model.state_dict()
    if set(expected) != set(state['model']):
        raise ValueError('producer tensor keys')
    for k, value in state['model'].items():
        if expected[k].shape != value.shape or expected[k].dtype != value.dtype or not bool(torch.isfinite(value).all()):
            raise ValueError('producer tensor shape/dtype/finite')
    if row['kind'] != 'baseline':
        from moderntcn_dev_v10.modules import validate_fixed_state
        validate_fixed_state(model, state['model'])
    model.load_state_dict(state['model'], strict=True)
    model.to(device).eval()
    return model, predict


def run(config, job, folder, stop):
    out = Path(folder)/job['id']
    out.mkdir(exist_ok=False)
    evidence_dir = out/'gpu-identity'
    if job['phase'] == 'E':
        check_gpu_identity(config['gpu_uuid'], evidence_dir, 'before-model-load', False)
    row = source(config, job)
    sets, meta = datasets(row)
    p = row['profile']
    torch.set_num_threads(config['cpu_threads'])
    model, predict = restore(row, 'cuda:0')
    original_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    from torch.utils.data import DataLoader, Subset
    with torch.inference_mode():
        if job['phase'] == 'E':
            batch = next(iter(DataLoader(sets['validation'], batch_size=job['batch'], shuffle=False, generator=torch.Generator().manual_seed(2024))))
            x, _, marks = batch_parts(batch)
            x = x.cuda()
            marks = marks.cuda() if marks is not None else None
            input_copy = x.cpu().clone()
            mark_copy = marks.cpu().clone() if marks is not None else None
            def forward():
                return predict(x, marks)
            def memory():
                return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                            max_allocated=torch.cuda.max_memory_allocated(), max_reserved=torch.cuda.max_memory_reserved())
            result = measure_with_identity(
                lambda: measure(forward, torch.cuda.synchronize, torch.cuda.reset_peak_memory_stats,
                                memory, job['batch'], checkpoint=stop),
                config['gpu_uuid'], evidence_dir)
            if not torch.equal(x.cpu(), input_copy) or (marks is not None and not torch.equal(marks.cpu(), mark_copy)):
                raise ValueError('measurement forward changed its fixed input/marks')
            result.update(parameters=sum(v.numel() for v in model.parameters()), cpu_threads=torch.get_num_threads(),
                          cpu_interop_threads=torch.get_num_interop_threads(),
                          torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                          gpu=torch.cuda.get_device_name(), matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                          cudnn_tf32=torch.backends.cudnn.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark,
                          deterministic=torch.backends.cudnn.deterministic, compile=False, precision='float32')
        else:
            selected = read(Path(folder)/'case-selection.json') if job['phase'] == 'C' else None
            data = Subset(sets['validation'], selected['dataset_indices']) if selected else sets['validation']
            meter = PositionErrors(p['pred_len'])
            predictions, truths = [], []
            calls = 0
            for batch in DataLoader(data, batch_size=128, shuffle=False, drop_last=False, generator=torch.Generator().manual_seed(2024)):
                stop()
                x, y, marks = batch_parts(batch)
                pred = predict(x.cuda(), marks.cuda() if marks is not None else None).cpu().numpy()
                truth = y.numpy()
                meter.add(pred, truth)
                predictions.append(pred)
                truths.append(truth)
                calls += 1
            prediction, truth = np.concatenate(predictions), np.concatenate(truths)
            if selected:
                if len(prediction) != 336:
                    raise ValueError('case exact sample count')
                nodes = np.repeat(selected['indices'], 168)
                scales = np.asarray(meta['scaler']['volume_scale'])[nodes, None, None]
                means = np.asarray(meta['scaler']['volume_mean'])[nodes, None, None]
                physical_prediction = prediction*scales+means
                physical_truth = truth*scales+means
                np.savez(out/'predictions.npz', prediction=prediction, truth=truth, physical_prediction=physical_prediction, physical_truth=physical_truth, indices=selected['dataset_indices'])
            else:
                np.savez(out/'predictions.npz', prediction=prediction, truth=truth, indices=np.arange(len(data)))
            result = dict(metrics=meter.result(), forward_calls=calls, presentations=len(data), split='validation', test=0)
            if job['phase'] == 'P':
                original = load_ref(row['metrics_ref'])
                epoch = original['best_epoch']
                if sha(row['history_ref']['path']) != row['history_ref']['sha256']:
                    raise ValueError('saved validation history changed')
                history = [__import__('json').loads(line) for line in Path(row['history_ref']['path']).read_text().splitlines()]
                saved = next(h['validation'] for h in history if h['epoch'] == epoch)
                comparisons = {}
                for metric in ('mse', 'mae'):
                    actual = result['metrics']['overall_'+metric]
                    error = abs(actual-saved[metric])
                    comparisons[metric] = dict(original=saved[metric], measured=actual, absolute_error=error,
                                               atol=2e-6, rtol=2e-5, passed=error <= 2e-6+2e-5*abs(saved[metric]))
                result['original_validation_comparison'] = comparisons
    if not torch.equal(rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
        raise ValueError('readonly inference changed RNG')
    if any(not torch.equal(v, model.state_dict()[k].cpu()) for k, v in original_state.items()):
        raise ValueError('readonly inference changed persistent model state')
    result.update(job=job, producer_manifest=row['manifest_ref'], strict_best=row['weights_ref'],
                  note='new measurement, never overwrites original scientific metrics; RevIN temporary cache excluded')
    write_new(out/'result.json', result)
    if job['phase'] == 'P' and not all(c['passed'] for c in result['original_validation_comparison'].values()):
        raise ValueError('full validation differs beyond preregistered policy; saved outputs retained')
    return result
