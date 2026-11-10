"""Validation-only sensitivity worker. No test loader capability or test claim."""
import random
from pathlib import Path
import numpy as np
import torch
from .contract import write_new, sha, load_ref
from .model import build
from moderntcn_dev_v10 import adapter
from moderntcn_dev_v10.guards import loss_guard, gradients_collected, parameters_collected
from utils.ch3_contract import BestState
from utils.ch3_onecycle import Schedule
from utils.ch3_data import load, batches
from utils.ch3_time_marks import batch_parts
from timemixer_dev.training_state import snapshot, save
from moderntcn_input_ablation_v10.worker import evaluate, append


def run(config, job, folder, stop, gpu_uuid):
    task = config['sensitivity_prototypes'][job['dataset']]
    p = task['structural_source_profile']
    tr = p['training']
    if (job['G'], job['R']) not in ((3, 4), (7, 4), (5, 2), (5, 6)):
        raise ValueError('default center is reference only')
    sets, metadata = load(task['data_config'], task, resolved_profile=p)
    if 'test' in sets or metadata.get('test_observations_accessed'):
        raise ValueError('validation-only data boundary')
    if metadata.get('prefixes', {}) != task['baseline_prefix_metadata']:
        raise ValueError('data record prefix mismatch')
    if any(metadata.get(k) != v for k, v in task['baseline_record_prefix'].items()):
        raise ValueError('data prefix identity mismatch')
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    torch.set_num_threads(tr['threads'])
    model = build(p, job['G'], job['R'], task='MS', fold=task['fold']).cuda()
    identity = dict(scope=config['scope'], purpose='validation_only_sensitivity', job=job,
                    analysis=model.analysis_identity, profile=p, data=metadata,
                    base_structural_spec=model.spec.value, source_binding=config['source_binding'],
                    execution_device=dict(device='cuda:0', gpu_uuid=gpu_uuid))
    out = Path(folder)/job['id']
    return fit(model, p, task, sets, identity, out, stop, 'cuda:0')


def fit(model, p, task, sets, identity, out, stop, device):
    """Shared training core; CPU acceptance supplies only synthetic tensors."""
    if set(sets) != {'train', 'validation'}:
        raise ValueError('only train/validation datasets may enter this worker')
    tr = p['training']
    job = identity['job']
    out = Path(out)
    opt = torch.optim.Adam(model.parameters(), lr=tr['lr'], betas=tuple(tr['betas']), eps=tr['eps'], weight_decay=tr['weight_decay'])
    scheduler = Schedule(opt, tr['scheduler'])
    generator = torch.Generator().manual_seed(2024)
    out.mkdir(exist_ok=False)
    write_new(out/'identity.json', identity)
    for name in ('history.jsonl', 'updates.jsonl'):
        (out/name).touch(exist_ok=False)
    best = BestState(tr['patience'])
    steps = 0
    route = dict(train=0, validation=0)
    selected = None
    for epoch in range(1, tr['epochs']+1):
        if best.stopped:
            break
        model.train()
        start = steps
        for batch_index, batch in enumerate(batches(sets['train'], p, 'train', generator)):
            stop()
            x, y, marks = batch_parts(batch)
            opt.zero_grad(set_to_none=True)
            pred = adapter.prediction(model, x.to(device), marks.to(device) if marks is not None else None)
            loss = adapter.loss(pred, y.to(device))
            loss_guard(loss)
            loss.backward()
            named = list(model.named_parameters())
            gradients_collected(named)
            opt.step()
            parameters_collected(named)
            scheduler.after_successful_update()
            steps += 1
            route['train'] += 1
            append(out/'updates.jsonl', dict(step=steps, epoch=epoch, batch=batch_index, loss=float(loss.detach()), lr=opt.param_groups[0]['lr'], beta1=opt.param_groups[0]['betas'][0]))
        if steps-start != task['arithmetic']['train_batches']:
            raise ValueError('training arithmetic mismatch')
        stop()
        metrics = evaluate(model, sets['validation'], p, device, adapter.prediction, route, 'validation')
        improved = best.update(metrics['mse'], epoch)
        append(out/'history.jsonl', dict(epoch=epoch, steps=steps, validation=metrics, best_epoch=best.epoch, bad=best.bad, improved=improved))
        state = snapshot(model, opt, scheduler, best, epoch, steps, generator, identity,
                         journals=dict(history_sha256=sha(out/'history.jsonl'), updates_sha256=sha(out/'updates.jsonl')))
        if improved:
            selected = metrics
            save(out/'best.pt', state)
        save(out/'last.pt', state)
    result = dict(status='validation_only_completed', job=job, best_epoch=best.epoch,
                  validation=selected, epochs=epoch if not best.stopped else len((out/'history.jsonl').read_text().splitlines()),
                  updates=steps, routes=route, test_calls=0, purpose='sensitivity_not_main_table')
    write_new(out/'result.json', result)
    write_new(out/'seal.json', dict(files={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}, identity=identity))
    return result
