"""Read-only reuse of the reviewed epoch checkpoint validation and restore."""
from timemixer_dev.training_state import (SCHEMA, parameter_names, rng_devices,
    validate_cuda_rng, file_sha, atomic_json, snapshot, save, read,
    static_groups)
from timemixer_dev import training_state as _reviewed
from .spec import CANDIDATE_ID, validate_revision


def _require_model_identity(model, identity):
    if identity.get('candidate_id') != CANDIDATE_ID:
        raise ValueError('ModernTCN v10 checkpoint candidate identity required')
    spec = validate_revision(identity['structural_spec'])
    if hasattr(model, 'spec'):
        if model.spec.value != spec:
            raise ValueError('live ModernTCN v10 model/task/arm/output structural identity mismatch')
    elif not (identity.get('reference_original') is True and identity.get('purpose') == 'synthetic'):
        raise ValueError('only explicit synthetic original-equivalence reference lacks candidate spec')


def validate(state, model, opt, scheduler, identity):
    _require_model_identity(model, identity)
    from .modules import validate_fixed_state
    validate_fixed_state(model, state['model'])
    return _reviewed.validate(state, model, opt, scheduler, identity)


def restore(path, model, opt, scheduler, generator, identity):
    _require_model_identity(model, identity)
    state = validate(read(path), model, opt, scheduler, identity)
    # Same reviewed restoration sequence, after the additional fixed-buffer check.
    model.load_state_dict(state['model'], strict=True); opt.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'], state['steps'])
    best = _reviewed.BestState(); best.__dict__.update(state['best'])
    _reviewed.random.setstate(state['rng']['python']); _reviewed.np.random.set_state(state['rng']['numpy']); _reviewed.torch.set_rng_state(state['rng']['torch'])
    if state['rng']['cuda']: _reviewed.torch.cuda.set_rng_state_all(state['rng']['cuda'])
    generator.set_state(state['rng']['generator'])
    return best, state['epoch'], state['steps']
