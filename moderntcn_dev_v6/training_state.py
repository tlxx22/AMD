"""Read-only reuse of the reviewed epoch checkpoint validation and restore."""
from timemixer_dev.training_state import (SCHEMA, parameter_names, rng_devices,
    validate_cuda_rng, file_sha, atomic_json, snapshot, save, read,
    static_groups)
from timemixer_dev import training_state as _reviewed
from .spec import CANDIDATE_ID, validate_revision


def _require_model_identity(model, identity):
    if identity.get('candidate_id') != CANDIDATE_ID:
        raise ValueError('ModernTCN v6 checkpoint candidate identity required')
    spec = validate_revision(identity['structural_spec'])
    if hasattr(model, 'spec'):
        if model.spec.value != spec:
            raise ValueError('live ModernTCN v6 model/task/arm/output structural identity mismatch')
    elif not (identity.get('reference_original') is True and identity.get('purpose') == 'synthetic'):
        raise ValueError('only explicit synthetic original-equivalence reference lacks candidate spec')


def validate(state, model, opt, scheduler, identity):
    _require_model_identity(model, identity)
    return _reviewed.validate(state, model, opt, scheduler, identity)


def restore(path, model, opt, scheduler, generator, identity):
    _require_model_identity(model, identity)
    return _reviewed.restore(path, model, opt, scheduler, generator, identity)
