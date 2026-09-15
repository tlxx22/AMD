"""Current THLS authorization is distinct from incomplete historical recovery."""
import hashlib
import json
from pathlib import Path

POLICY_ID = 'thls-engineering-exact-restrictions-current-v1'
APPROVAL_SHA = '0ea8fb954f43caa2d9e9e5bd1e4e0609a0568bbae54ff6840a8ca19d4791a121'
CURRENT_IDS = tuple('test_urbanev_data_contract.UrbanEVDataContractTests.' + name for name in (
    'test_actual_source_dimensions_time_axis_and_fingerprints',
    'test_actual_source_hashes_and_first_version_file_scope',
    'test_weather_central_mapping_calendar_and_exclusions',
    'test_actual_canonical_schema_is_eleven_channels_and_ordered'))
CPU_ID = 'test_runner.PMCRMSInterfaceTests.test_cuda_amd_equivalence_body_rng_generator_first_batch'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate(document, inventory):
    current = document['current_policy']
    if document['recovery_status'] != 'incomplete' or current['historical_recovery_status'] != 'incomplete':
        raise ValueError('historical evidence must remain incomplete')
    if current['id'] != POLICY_ID:
        raise ValueError('unknown current policy')
    approval = dict(current['approval']); claimed = approval.pop('sha256')
    if digest(approval) != claimed or claimed != APPROVAL_SHA or set(approval['ids']) != set(CURRENT_IDS):
        raise ValueError('current explicit approval missing or altered')
    inherited = set(inventory['inherited']); new = set(inventory['new'])
    rows = current['entries']; ids = [r['id'] for r in rows]
    legacy = {r['id']:r for r in document['restrictions']}
    if len(legacy) != 17 or len(ids) != 21 or len(set(ids)) != 21:
        raise ValueError('exact restriction cardinality mismatch')
    if set(ids) != set(legacy) | set(CURRENT_IDS) or not set(ids) <= inherited or set(ids) & new:
        raise ValueError('extra, missing, foreign or new-test restriction')
    for row in rows:
        if row['id'] in CURRENT_IDS:
            if (row['basis'] != 'current_user_authorization' or row['current_approval'] != APPROVAL_SHA
                    or row['historical_evidence'] != 'Missing/Not verified' or row['mode'] != 'all'):
                raise ValueError('current approval cannot be described as historical recovery')
        else:
            original = legacy[row['id']]
            if (row['basis'] != 'historical_semantic_record' or row['historical_evidence'] != 'B'
                    or any(row[k] != original[k] for k in ('reason', 'mode', 'evidence'))):
                raise ValueError('historical semantic restriction changed')
        if (row['id'] == CPU_ID) != (row['mode'] == 'cpu_only'):
            raise ValueError('CPU-only restriction applied to wrong method')
    return rows


def restrictions_for(document, inventory, stage):
    rows = validate(document, inventory)
    if stage not in {'new_cuda', 'inherited_cpu', 'real_prefix'}:
        raise ValueError('unapproved execution mode')
    if stage != 'inherited_cpu':
        return {}
    return {r['id']:r['reason'] for r in rows}


def require_scope(config, stage):
    wanted = 'real_prefix_probe' if stage == 'real_prefix' else 'synthetic_regression'
    if config.get('access_policy') == 'ettm1_thls_development_smoke_v1':
        if (stage != 'real_prefix' or config.get('task_scope') not in {'M4_63_ETTm1_16_4_0', 'M4_67_ETTm1_NSJ_48_16_0'}
                or config.get('approved_real_file') != str(Path(config['repo'])/'data/ETTm1.csv')
                or config.get('approved_data_sha256') !=
                '6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e'):
            raise ValueError('ETTm1 smoke task/file identity mismatch')
        wanted = 'ettm1_thls_development_smoke_v1'
    if (config.get('stage') != stage or config.get('access_policy') != wanted
            or config.get('restriction_policy_id') != POLICY_ID):
        raise ValueError('stage/access policy mismatch; real access cannot be borrowed')
    if config.get('approval_sha256') != APPROVAL_SHA:
        raise ValueError('execution is not bound to current approval')
