"""Only the registered AMD ETT declaration failure after sealed base287."""
import ast
import hashlib
import hmac
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from utils.ch3_contract import ROOT, digest, profile, step_arithmetic, BestState
from utils.ch3_native_recovery_records import bound, exclusive, ref, sha
from utils import ch3_moderntcn_etth1_recovery as numeric

PACKAGE = numeric.POLICY_PACKAGE / 'amend-ett-identity-repair-v1'
PRODUCER = '760b9dd7162d200c11b8836a7c9ece41822dfcf1'
OLD_RESULT = numeric.RESULT
RESULT = OLD_RESULT.with_name('baseline-unified-v3-ms-seal-m128-recovery1-amend-ett-identity-r1')
ATTEMPT = 'M_AMEND-AMD-ETT-identity-r1'
SESSION = 'ch3-m6-amend-ett-identity-r1'
LOG = PACKAGE / 'followup-launcher.log'
ENTRY = ROOT / 'm6_amend_ett_identity_recovery_entry.py'
WRAPPER = ROOT / 'scripts/ch3/start_amend_ett_identity_recovery.sh'
SOURCE_REF = dict(path=str(PACKAGE / 'source-anchors.json'), sha256='8fe23863733d7a8f59733b5f1018ca3907d611fba2c72222510f7a783f47bf36')
REUSE_REF = dict(path=str(PACKAGE / 'prefix-verification.json'), sha256='4550f45447919a8d12c2a65c1d13851ef35d7b6a606fdabebc1396d56008c4b0')
ALL_M_POLICY_REF = numeric.ALL_M_POLICY_REF
COMPLETED_PREFIX = True
ACTIVE = False


def delta_ref():
    return ref(PACKAGE / 'producer-delta-proof.json')


def code_binding():
    return {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), ENTRY, WRAPPER,
        ROOT / 'tests/test_m6_amend_ett_identity_recovery.py')}


def activate():
    global ACTIVE, DELTA_REF
    if ACTIVE:
        return
    numeric.activate()
    from utils import ch3_type1_tasks as s, ch3_type1_chain as q, ch3_ms_seal_recovery as ms, ch3_round2_amendment as amend
    ACTIVE = True
    DELTA_REF = delta_ref()
    s.RESULT = ms.RESULT = RESULT
    amend.RESULT = RESULT / 'round2-amendment'
    original_context = s.context
    s.context = lambda c: dict(original_context(c), fixture=PACKAGE / c['baseline_unified']['stage'] / 'fixtures')
    q.CONTROL = RESULT / 'queue/controller'
    q.LOG, q.SESSION, q.ENTRY, q.WRAPPER = LOG, SESSION, ENTRY, WRAPPER
    q.PROBE_RECOVERY = sys.modules[__name__]
    q.QUEUE_LOCK = numeric.PACKAGE / 'queue.lock'


def anchors():
    return dict(recovery='specific_AMD_new_ETT_declaration_after_completed_base287',
        execution_attempt=ATTEMPT, parent_technical_failure_ref=SOURCE_REF,
        source_anchors_ref=SOURCE_REF, completed_prefix_ref=REUSE_REF,
        producer_delta_ref=delta_ref(), candidate_config_refs=numeric.CONFIG_REFS,
        all_m_policy_ref=ALL_M_POLICY_REF, retained_producer_commit=PRODUCER,
        expected_runs=dict(MS=203, M=84, total=287), completed_new_formal=84,
        expected_remaining_formal=343, approved_total_new_formal=427)


def amd_guard_delta(old, new):
    """An exact AST replacement of one dataset tuple, never a function whitelist."""
    tree = ast.parse(old)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_amd_declaration')
    candidates = [n for n in ast.walk(function.body[0]) if isinstance(n, ast.Tuple) and all(isinstance(x, ast.Constant) for x in n.elts)
                  and ast.literal_eval(n) == ('ETTh1', 'Weather', 'Exchange')]
    if len(candidates) != 1:
        raise ValueError('exact old AMD M dataset guard required')
    candidates[0].elts.extend(ast.Constant(x) for x in ('ETTh2', 'ETTm1', 'ETTm2'))
    if ast.dump(tree, include_attributes=False) != ast.dump(ast.parse(new), include_attributes=False):
        raise ValueError('only the approved AMD M dataset tuple may change')


def verify_production_inheritance():
    proof = bound(delta_ref())
    if proof.get('purpose') != 'exact_base287_producer_delta_v1' or proof.get('producer_commit') != PRODUCER:
        raise ValueError('registered producer delta proof')
    producer = bound(bound(SOURCE_REF)['refs']['M_boundary'])['code']
    allowed = {'utils/ch3_contract.py', 'utils/ch3_type1_chain.py', 'utils/ch3_round2_amendment.py',
               'utils/ch3_probe_schema_recovery.py', 'utils/ch3_type1_execution.py', 'm6_type1_followup_entry.py'}
    actual = {name for name, expected in producer.items() if sha(ROOT / name) != expected}
    if actual != set(proof['changes']) or not actual <= allowed:
        raise ValueError('unregistered completed producer delta')
    for name, value in proof['changes'].items():
        old = subprocess.check_output(['git', '-C', str(ROOT), 'show', PRODUCER + ':' + name])
        if hashlib.sha256(old).hexdigest() != producer[name] or value != dict(before_sha256=producer[name], after_sha256=sha(ROOT / name)):
            raise ValueError('exact reviewed producer before/after bytes: ' + name)
        if name == 'utils/ch3_contract.py':
            amd_guard_delta(old.decode(), (ROOT / name).read_text())
    # New entry code is exact-bound as well; no computation-producing file outside
    # this finite repair delta can change under an existing precise start approval.
    if proof['new_code'] != code_binding():
        raise ValueError('new adoption entry byte binding')
    from tools.restricted_regression.run_restricted import verify_bundle
    return dict(changes=sorted(actual), bundle_sha=verify_bundle())


def verify_source_light():
    source = bound(SOURCE_REF)
    if source['producer_commit'] != PRODUCER or source['old_result'] != str(OLD_RESULT) or source['config_refs'] != numeric.CONFIG_REFS:
        raise ValueError('exact old r2 producer and five immutable configurations')
    rows = {k: bound(v) for k, v in source['refs'].items()}
    for value in numeric.CONFIG_REFS.values():bound(value)
    if rows['controller']['authorization'] != source['refs']['authorization'] or rows['authorization']['closure_commit'] != PRODUCER:
        raise ValueError('old lifecycle authorization')
    if rows['failure']['error'] != "RuntimeError('owned child technical failure: M_AMEND-probe')" or rows['probe_failure']['error'] != "RuntimeError('serial technical gate failed; no fallback')":
        raise ValueError('only the registered AMD declaration failure')
    if sha(source['failed_worker_log']['path']) != source['failed_worker_log']['sha256'] or 'M AMD native all-channel identity' not in Path(source['failed_worker_log']['path']).read_text():
        raise ValueError('registered actual construction failure')
    if rows['failed_budget']['counts'] != dict(adam=0, backward=0, forward=0) or rows['failed_process']['returncodes'] != [1] or rows['failed_process']['failure_kind'] != 'business':
        raise ValueError('failed first construction, not a resource fallback')
    amend = bound(numeric.CONFIG_REFS['M_AMEND'])
    failed = source['failed_task']
    if failed != 'AMD-ETTh2-M-oc01-v3-amend1-m128-recovery1-f1-h96-s2024' or failed not in {t['id'] for t in amend['tasks']} or rows['failed_worker_config']['task'] != failed or rows['failed_runtime']['task'] != failed or rows['probe_permit']['commit'] != PRODUCER or rows['probe_permit']['protocol_sha'] != digest(amend):
        raise ValueError('exact first AMD ETT task/permit/runtime identity')
    ms, verification = rows['MS_boundary'], rows['MS_verification']
    base, m = rows['base'], rows['M_boundary']
    if base['commit'] != PRODUCER or base['counts'] != dict(MS=203, M=84, total=287) or base['technical_complete'] is not True or base['boundaries'] != dict(MS=source['refs']['MS_boundary'], M=source['refs']['M_boundary']) or base['upstream_ref'] != source['refs']['upstream']:
        raise ValueError('exact sealed base287 prefix')
    if rows['upstream']['boundaries']['MS'] != source['refs']['MS_boundary'] or ms['source_verification_ref'] != source['refs']['MS_verification'] or ms['training_commit'] != 'df6a16403e10d51097db8c88829909c533d15652' or ms['task_ids'] != verification['task_ids'] or ms['receipts'] != verification['receipts'] or len(set(ms['task_ids'])) != 203:
        raise ValueError('retained MS203 original source')
    c = bound(numeric.CONFIG_REF)
    if m['commit'] != PRODUCER or m['protocol_sha'] != digest(c) or m['task_ids'] != [t['id'] for t in c['tasks']] or m['technical_complete'] is not True or set(m['receipts']) != {'AMD','DLinear','PatchTST','iTransformer','TimeMixer','ModernTCN','TimeXer'}:
        raise ValueError('exact completed M84 producer/task coverage')
    complete = rows['M_probe_complete']
    from utils import ch3_type1_tasks as s
    groups = s.probe_groups(c)
    if complete['execution_complete'] is not True or complete['plan'] != s.plan(c) or complete['protocol_sha'] != digest(c) or set(complete['decisions']) != {g['id'] for g in groups} or any(complete['decisions'][g['id']]['status'] != 'Passed' or complete['decisions'][g['id']]['coverage'] != g['coverage'] for g in groups) or rows['M_permit']['commit'] != PRODUCER or rows['M_permit']['protocol_sha'] != digest(c) or any(rows['M_permit'][k] != m[k] for k in ('code','environment','hardware','source_states')):
        raise ValueError('retained complete M_BASE admission')
    from utils.ch3_type1_upstream import assert_owned_exited
    assert_owned_exited(source['exit_instances'])
    if (OLD_RESULT / 'queue/controller/STOP').exists() or (OLD_RESULT / 'round2-amendment/probe/M_AMEND/STOP').exists():
        raise ValueError('STOP is not the registered failure')
    return rows


def verify_prefix():
    """One startup metadata integrity pass; no checkpoint bytes or probe replay."""
    rows = verify_source_light()
    evidence = bound(REUSE_REF)
    c = bound(numeric.CONFIG_REF)
    data = bound(c['baseline_unified']['data_ref'])
    if evidence['source_ref'] != SOURCE_REF or evidence['producer_commit'] != PRODUCER or evidence['counts'] != dict(MS=203, M=84, total=287) or evidence['M_receipts'] != rows['M_boundary']['receipts'] or set(evidence['M_tasks']) != {t['id'] for t in c['tasks']}:
        raise ValueError('review-bound complete prefix evidence')
    seen = set()
    for model, value in evidence['M_receipts'].items():
        group = bound(value)
        tasks = [t for t in c['tasks'] if t['model'] == model]
        if group['technical_complete'] is not True or group['model'] != model or group['task_ids'] != [t['id'] for t in tasks] or set(group['artifacts']) != set(group['task_ids']):
            raise ValueError('complete exact twelve-task model receipt')
        for t in tasks:
            run = t['id']; p = profile(c, t); files = group['artifacts'][run]
            if run in seen or evidence['M_tasks'][run]['artifacts'] != files or set(files) != {'result.json','manifest.json','history.jsonl','budget.json','runtime.json','best.pt','last.pt'}:
                raise ValueError('missing/duplicate/sealed task references')
            seen.add(run)
            root = OLD_RESULT / 'M_BASE' / ('formal-' + model) / run
            if any(v['path'] != str(root / k) for k, v in files.items()):
                raise ValueError('exact completed producer artifact root')
            r, m, budget, runtime = [bound(files[k]) for k in ('result.json','manifest.json','budget.json','runtime.json')]
            if r['id'] != run or r['commit'] != PRODUCER or r['protocol_sha'] != digest(c) or r['profile_sha'] != digest(p) or r['scientific_protocol'] != c['baseline_unified']['id'] or m['task'] != t or m['profile'] != p or m['identity']['commit'] != PRODUCER or m['identity']['profile_sha'] != digest(p) or m['identity']['protocol_sha'] != digest(c) or m['identity']['data_sha'] != data['data_bindings'][t['dataset']][run]:
                raise ValueError('retained actual result/manifest/profile/version')
            h = files['history.jsonl']
            if ref(h['path']) != h:
                raise ValueError('retained history checksum')
            history = [json.loads(line) for line in Path(h['path']).read_text().splitlines()]
            best = BestState(p['training']['patience']); arithmetic = step_arithmetic(c, t)
            for epoch, row in enumerate(history, 1):
                if row['epoch'] != epoch or row['steps'] != epoch * arithmetic['train_batches']:
                    raise ValueError('retained epoch/update history')
                val = row['validation']
                elements = arithmetic['validation_windows'] * p['pred_len'] * p['C']
                if val['elements'] != elements or any(not math.isfinite(val[k]) for k in ('mse','mae','sse','sae')) or val['mse'] != val['sse']/elements or val['mae'] != val['sae']/elements:
                    raise ValueError('retained finite all-channel weighted validation')
                best.update(row['validation']['mse'], epoch)
                if row['best_epoch'] != best.epoch:
                    raise ValueError('retained validation-only selection')
            forward = len(history)*(arithmetic['train_batches']+math.ceil(arithmetic['validation_windows']/p['training']['eval_batch']))+math.ceil(arithmetic['test_windows_arithmetic_only']/p['training']['eval_batch'])
            counts = dict(adam=arithmetic['max_optimizer_steps'], backward=arithmetic['max_optimizer_steps'], forward=forward)
            if len(history) != p['training']['epochs'] or r.get('final_test') != dict(calls=1, selected='best.pt', sha256=files['best.pt']['sha256'], epoch=best.epoch) or r['best_epoch'] != best.epoch or runtime.get('error') is not None or runtime['task'] != run or budget['counts'] != counts or budget.get('by_pid') != {str(runtime['pid']):counts} or r['scheduler_updates'] != counts['adam'] or r['scheduler_sha'] != digest(p['training']['scheduler']) or r['elements'] != arithmetic['test_windows_arithmetic_only'] * p['pred_len'] * p['C'] or r['metric_scope'] != 'all_channels':
                raise ValueError('completed history/test-once/runtime/budget')
            for name in ('best.pt','last.pt'):
                path = Path(files[name]['path'])
                if path.is_symlink() or not path.is_file() or dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns) != evidence['M_tasks'][run]['checkpoints'][name]:
                    raise ValueError('retained checkpoint stat differs; no automatic re-test')
    return rows


def status():
    verify_source_light()
    try:verify_production_inheritance()
    except (ImportError,RuntimeError) as exc:
        raise ValueError('completed-prefix producer/bundle verification failed: '+str(exc)[:300]) from exc
    return dict(state='COMPLETED_BASE287_ADOPTION_AWAITING_START', READY_FOR_GPU_EXECUTION=False,
                anchors=anchors(), remaining_formal_runs=343, result_review='pending')


def sign(body):
    from utils import ch3_type1_chain as q
    secret = os.environ.get(q.SECRET)
    if not secret:
        raise PermissionError('new controlled adoption lifecycle required')
    return dict(body, mac=hmac.new(secret.encode(), digest(body).encode(), hashlib.sha256).hexdigest())


def adopt_prefix():
    from utils import ch3_type1_chain as q
    q.stop_check(); verify_production_inheritance(); rows = verify_prefix(); q.stop_check()
    source = bound(SOURCE_REF); commit = q.closure()
    upstream = dict(state='COMPLETED_BASE287_OLD_SOURCES_RELEASED', READY_FOR_GPU_EXECUTION=True,
        anchors=anchors(), boundaries=dict(MS=source['refs']['MS_boundary']), handoff_scope=q.s.ID,
        successor_owner=q.owner(), adoption_commit=commit, adopted_base287_ref=source['refs']['base'],
        prefix_verification_ref=REUSE_REF, result_review='pending')
    upstream_ref = exclusive(q.CONTROL / 'upstream-technical-boundary.json', sign(upstream))
    # Keep the old training commit/code and receipts. Only the adoption has a new
    # commit, owner and MAC; the old producer MAC is never reused as permission.
    body = dict(rows['M_boundary'], adopted_source_ref=source['refs']['M_boundary'],
                prefix_verification_ref=REUSE_REF, adoption_commit=commit, adoption_owner=q.owner())
    m_ref = exclusive(q.s.context(q.configs()['M_BASE'])['control'] / 'technical-boundary.json', sign(body))
    base_ref = q.seal_base287_boundary(m_ref)
    return {'VERIFY_IMPORT_MS203_AND_SEAL': upstream_ref, 'SEAL_M128_BOUNDARY': m_ref,
            'SEAL_BASE_287_BOUNDARY': base_ref}


def validate_adopted_boundary(value):
    from utils import ch3_type1_chain as q
    body = bound(value); content = {k: v for k, v in body.items() if k != 'mac'}
    controller = json.loads((q.CONTROL / 'controller.json').read_text())
    if os.environ.get(q.SECRET):
        if not hmac.compare_digest(sign(content)['mac'], body.get('mac', '')) or not q.same(controller['owner']):
            raise PermissionError('current controlled adoption MAC/owner required')
    if body.get('adoption_owner') != controller['owner'] or body.get('adoption_commit') != q.closure() or body.get('prefix_verification_ref') != REUSE_REF or body.get('adopted_source_ref') != bound(SOURCE_REF)['refs']['M_boundary']:
        raise PermissionError('completed prefix is not adopted by this exact lifecycle')
    old = bound(body['adopted_source_ref'])
    if {k: body[k] for k in old} != old:
        raise ValueError('adoption must retain every original producer field')


def validate_permit_link(c, a, probe):
    if c['baseline_unified']['stage'] == 'M_BASE':
        raise PermissionError('completed M_BASE cannot be dispatched again')
    if a.get('execution_attempt') != ATTEMPT or a.get('probe_recovery_ref') != REUSE_REF:
        raise PermissionError('exact completed-prefix attempt/evidence binding')
    numeric.validate_revision(c)


def retained_refs(report):
    if report.get('probe_recovery_ref') not in (None, REUSE_REF):
        raise PermissionError('unregistered completed-prefix reference')
    return {}


def retained_payload(c, point): return False
def location(report, key, default): return Path(default)
def producer(report, key, current): return current
def self_review(report, path): return False
def group_review(report, group): return None
def extra_actual(report): return dict(adam=0, backward=0, forward=0)
def load_seed(c, a): raise PermissionError('completed M_BASE must never be replayed')
