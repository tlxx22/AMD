"""Profile -> model -> prediction -> exact-shape loss; no training entry point."""
import copy
from types import SimpleNamespace
import torch
from .source import load_author, verify_author, implementation_binding
from .spec import StructuralSpec, canonical, validate_revision


def build(profile, *, use_u, use_v, task=None, fold=None):
    from .model import ModernTCNUV
    spec = StructuralSpec.from_profile(profile, use_u=use_u, use_v=use_v, task=task, fold=fold)
    verify_author()
    torch.random.default_generator.manual_seed(2024)
    return ModernTCNUV(spec)


def build_original(profile):
    spec = StructuralSpec.from_profile(profile, use_u=False, use_v=False)
    torch.random.default_generator.manual_seed(2024)
    return load_author().Model(SimpleNamespace(**spec.value['native_options']))


def build_from_task(config, task, *, use_u, use_v):
    """Resolve existing metadata only; task ID/fold remain explicit identity."""
    from utils.ch3_contract import profile
    if task.get('model') != 'ModernTCN':
        raise ValueError('select a ModernTCN baseline profile as structural source')
    p = profile(config, task)
    return build(p, use_u=use_u, use_v=use_v, task=task.get('task', 'MS'), fold=task['fold'])


def prediction(model, x, x_mark_enc=None):
    v = model.spec.value
    validate_revision(v)
    if tuple(x.shape[1:]) != (v['L'], v['C']):
        raise ValueError('candidate history shape mismatch')
    policy = v['time_mark']
    if policy is not None:
        if x_mark_enc is None or tuple(x_mark_enc.shape) != (x.shape[0], v['L'], policy['K']):
            raise ValueError('registered historical native marks required')
    elif x_mark_enc is not None:
        raise ValueError('unregistered native marks')
    y = model(x, x_mark_enc)
    if tuple(y.shape) != (x.shape[0], v['pred_len'], v['C']):
        raise ValueError('native output must retain all C channels')
    return y if v['task'] == 'M' else y[:, :, v['target_idx']:v['target_idx'] + 1]


def loss(predicted, target):
    if predicted.shape != target.shape:
        raise ValueError('prediction/target shape mismatch; broadcasting forbidden')
    return torch.nn.functional.mse_loss(predicted, target)


def synthetic_checkpoint(model):
    return dict(format='moderntcn-uv-synthetic-v5', spec=model.spec.value,
                state_dict=copy.deepcopy(model.state_dict()))


def load_synthetic_checkpoint(model, checkpoint):
    # Identity and tensor schema checked before the first parameter is written.
    validate_revision(model.spec.value)
    if checkpoint.get('format') != 'moderntcn-uv-synthetic-v5' or canonical(checkpoint.get('spec')) != model.spec.encoded:
        raise ValueError('checkpoint structural/task/source identity mismatch')
    if implementation_binding() != model.spec.value['implementation']:
        raise ValueError('implementation bytes changed since construction')
    verify_author()
    expected, state = model.state_dict(), checkpoint['state_dict']
    if set(expected) != set(state) or any(expected[k].shape != state[k].shape or expected[k].dtype != state[k].dtype for k in expected):
        raise ValueError('checkpoint tensor schema mismatch')
    return model.load_state_dict(state, strict=True)
