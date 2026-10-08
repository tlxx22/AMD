"""Pure record, provenance and accounting rules for native v4 recovery1."""
import hashlib
import json
import math
from pathlib import Path
from utils.ch3_contract import digest, profile, step_arithmetic, BestState

BASE = '0734f3e91f15854f75907c30d10d43e942f69267'
ID = 'm6-native-tmark-chain-v4-recovery1'
MODE = 'preauthorized_machine_gate'
KINDS = ('adam', 'forward', 'backward')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def ref(path):
    return dict(path=str(Path(path)), sha256=sha(path))


def bound(value):
    if set(value) != {'path', 'sha256'}:
        raise ValueError('exact path/SHA reference required')
    p = Path(value['path'])
    if p.is_symlink() or p.resolve() != p or sha(p) != value['sha256']:
        raise ValueError('bound record path/SHA changed')
    result = json.loads(p.read_text())
    if sha(p) != value['sha256']:
        raise ValueError('record changed during read')
    return result


def exclusive(path, value):
    p = Path(path)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n').encode()
    # Never follow a retained link, even when its target happens to match.
    if p.is_symlink():
        raise FileExistsError('retained link')
    if p.exists():
        if p.read_bytes() != raw:
            raise FileExistsError('retained record differs; no overwrite')
        return ref(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as handle:
        handle.write(raw)
    return ref(p)


def compact_decisions(report):
    result = {}
    for key, value in report['decisions'].items():
        if value.get('status') != 'Passed' or type(value.get('concurrency')) is not int or value['concurrency'] not in (1, 2, 4):
            raise ValueError('exact successful measured concurrency required')
        result[key] = dict(status='Passed', concurrency=value['concurrency'],
                           coverage=value.get('coverage'),
                           representatives=value.get('representatives'),
                           source_decision_sha=digest(value))
    return result


def validate_whole_card_receipt(value,memory,expected,task_ids):
    """Saved card samples plus owned control lifecycle, never GPU attribution."""
    from tools.restricted_regression.m5_formal_entry import resource_assessment
    if expected.get('resource_mode')!='exclusive_gpu_whole_card_v1' or any(value.get(k)!=expected.get(k)for k in('resource_mode','resource_contract_ref')):raise ValueError('exact resource receipt/producer binding')
    bound(expected['resource_contract_ref'])
    if (value.get('failure')is not None or value.get('returncodes')!=[0]*len(task_ids) or not task_ids or value.get('resource_admission')is not True
        or value.get('owned_workers_exited')is not True or any(value.get(k)is not None for k in('process_peaks','cpu_peaks','process_attribution','external_occupancy_known','exit_observation','exit_transitions_resolved'))):raise ValueError('whole-card resource/lifecycle receipt')
    workers=value.get('worker_lifecycles',[])
    if ([w.get('task_id')for w in workers]!=task_ids or len({w.get('pid')for w in workers})!=len(task_ids)
        or any(set(w)!={'task_id','pid','start_ticks','returncode'}or type(w['pid'])is not int or not str(w['start_ticks']).isdigit()or w['returncode']!=0 for w in workers)):raise ValueError('owned control lifecycle/task identity')
    baseline=value['baseline']
    fields={'time','uuid','device','total','used','free','driver_reserved','resource_mode','query_timeout','query_elapsed'}
    if set(baseline)!=fields or not resource_assessment(baseline,[],resource_mode=expected['resource_mode'])['admission']:raise ValueError('whole-card baseline safety')
    gpu=expected.get('hardware',{}).get('gpu')
    if not isinstance(gpu,str)or not gpu.strip()or baseline['uuid']!=gpu.splitlines()[0].split(',')[0].strip():raise ValueError('resource card differs from fixed producer GPU')
    last=baseline['time'];last_row=None;count=0;peak=None
    with Path(memory).open()as handle:
        for line in handle:
            row=json.loads(line);assessment=resource_assessment(row,[],baseline,resource_mode=expected['resource_mode'])
            if set(row)!=fields|{'assessment'} or row.get('assessment')!=assessment or not assessment['admission']or row['time']<last:raise ValueError('whole-card sample integrity/safety')
            last=row['time'];last_row=row;count+=1;peak=row['used']if peak is None else max(peak,row['used'])
    if not count or value.get('sample_count')!=count:raise ValueError('complete whole-card samples required')
    observed=value.get('lifecycle_exit_observed_at')
    if (type(observed)not in(int,float)or not math.isfinite(observed)or observed<baseline['time']or observed>last
        or value.get('fresh_post_exit_sample')!=last_row or value.get('whole_card_peak')!=peak):raise ValueError('fresh post-exit whole-card proof required')
    return {str(w['pid']):dict(pid=w['pid'],start_ticks=str(w['start_ticks']),task_id=w['task_id'])for w in workers}


def project_summary(report, complete_ref, admission_ref, permit_ref):
    """A deterministic projection, never a hand-authored Passed assertion."""
    if report.get('execution_complete') is not True or report.get('reviewed') is not False:
        raise ValueError('actual original machine probe completion required')
    return dict(purpose='native_recovery_admission_summary_v1', scope=ID,
                science_baseline_commit=BASE, original_probe_execution_commit=report['commit'],
                original_complete_ref=complete_ref, original_admission_ref=admission_ref,
                original_probe_permit_ref=permit_ref, protocol_sha=report['protocol_sha'],
                code=report['code'], environment=report['environment'], hardware=report['hardware'],
                decisions=compact_decisions(report), budget=report['budget'],
                technical_admission=True, reviewed=False, manual_review=False,
                review_mode=MODE, result_review='pending')


def manifest_projection(report, complete_ref, root, extra_refs=()):
    root = Path(root)
    from utils.ch3_probe_schema_recovery import current_recovery
    readonly=current_recovery().retained_refs(report)
    artifacts = dict(report.get('artifacts',{}))
    for value in extra_refs:
        if value['path'] in artifacts and artifacts[value['path']]!=value:raise ValueError('conflicting expected artifact reference')
        artifacts[value['path']]=value
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError('original exact artifact inventory absent')
    rows = []
    for name, expected in sorted(artifacts.items()):
        p = Path(name)
        if p.is_symlink() or p.resolve() != p or (not p.is_relative_to(root) and readonly.get(name)!=expected):
            raise ValueError('probe artifact path/symlink outside exact scope')
        if expected != dict(path=str(p), sha256=expected.get('sha256')):
            raise ValueError('original artifact reference identity')
        actual = sha(p)
        if actual != expected['sha256']:
            raise ValueError('original expected probe SHA mismatch')
        rows.append(dict(path=str(p), expected_sha256=expected['sha256'],
                         verified_current_sha256=actual, size=p.stat().st_size,
                         artifact_role='numeric_payload' if p.suffix == '.bin' else 'probe_evidence'))
    result=dict(purpose='native_recovery_probe_manifest_v1', source_probe_complete_ref=complete_ref,
                rows=rows, artifact_count=len(rows), verified_bytes=sum(r['size'] for r in rows),
                trust_basis='deterministic projection of fixed original complete artifact refs')
    if report.get('probe_recovery_ref'):
        result['readonly_probe_recovery_ref']=report['probe_recovery_ref']
        result['readonly_probe_scope']=report.get('scope','')
        if report.get('single_serial_source'):result['single_serial_source']=report['single_serial_source']
    for name in('resource_mode','resource_contract_ref'):
        if name in report:result[name]=report[name]
    return result


def process_refs(report):
    """Resources and approval are explicit refs outside the raw artifact map."""
    return [v['process'] for v in report['evidence'].values()]+[report['approval']]


def merge_manifests(manifests):
    source=manifests[0]['source_probe_complete_ref'];rows=[];seen=set()
    for m in manifests:
        if m['source_probe_complete_ref']!=source or m['artifact_count']!=len(m['rows']):raise ValueError('same complete source and exact manifest counts')
        for row in m['rows']:
            if row['path'] in seen:raise ValueError('overlapping manifest projection')
            seen.add(row['path']);rows.append(row)
    return dict(purpose='native_recovery_probe_manifest_v1',source_probe_complete_ref=source,
        rows=rows,artifact_count=len(rows),verified_bytes=sum(r['size'] for r in rows))


def scan_manifest(manifest, source_ref, root, on_read=None):
    """One startup integrity scan, no tensor parsing or numeric comparison."""
    if manifest.get('purpose') != 'native_recovery_probe_manifest_v1' or manifest.get('source_probe_complete_ref') != source_ref:
        raise ValueError('manifest original source mismatch')
    root = Path(root); seen = set()
    from utils.ch3_probe_schema_recovery import current_recovery
    selector=dict(probe_recovery_ref=manifest.get('readonly_probe_recovery_ref'),scope=manifest.get('readonly_probe_scope',''))
    if manifest.get('single_serial_source'):selector['single_serial_source']=manifest['single_serial_source']
    readonly=current_recovery().retained_refs(selector) if manifest.get('readonly_probe_recovery_ref') else {}
    for row in manifest['rows']:
        p = Path(row['path'])
        if str(p) in seen or p.is_symlink() or p.resolve() != p or (not p.is_relative_to(root) and readonly.get(str(p))!=dict(path=str(p),sha256=row['expected_sha256'])):
            raise ValueError('manifest duplicate/path/link')
        seen.add(str(p))
        if row['verified_current_sha256'] != row['expected_sha256'] or p.stat().st_size != row['size']:
            raise ValueError('manifest identity/size mismatch')
        if on_read:
            on_read(p)
        if sha(p) != row['expected_sha256']:
            raise ValueError('startup artifact integrity failed')
    if len(seen) != manifest['artifact_count']:
        raise ValueError('manifest incomplete count')
    return dict(files=len(seen), bytes=sum(r['size'] for r in manifest['rows']), integrity_scan_passed=True)


def validate_inventory(c, inventory):
    if inventory.get('purpose') != 'native_v4_recovery_inventory_v1' or inventory.get('science_baseline_commit') != BASE or inventory.get('protocol_sha') != digest(c):
        raise ValueError('recovery inventory science identity')
    if [r['task_id'] for r in inventory['rows']] != [t['id'] for t in c['tasks']]:
        raise ValueError('exact ordered inventory task set')
    if inventory.get('budget_refund') is not False:
        raise ValueError('historical consumption cannot be refunded')
    for t, row in zip(c['tasks'], inventory['rows']):
        kind = row['classification']
        if kind not in 'ABCD' or row.get('errors'):
            raise ValueError('inconsistent recovery task')
        if row['profile_sha'] != digest(profile(c, t)) or row['original_commit'] != BASE:
            raise ValueError('recovery task/profile identity')
        counts = row['actual_charged_counts']
        if set(counts) != set(KINDS) or any(type(v) is not int or v < 0 for v in counts.values()):
            raise ValueError('actual execution accounting')
        if kind == 'B':
            if row.get('test_access_status') != 'not_observed_under_guard' or row.get('checkpoint_committed_epoch', 0) < 1:
                raise ValueError('resume test status/checkpoint')
            committed = row['checkpoint_represented_counts']
            if any(counts[k] < committed[k] or row['charged_work_after_checkpoint'][k] != counts[k]-committed[k] for k in KINDS):
                raise ValueError('charged-after-checkpoint accounting')
        if kind == 'A' and row['test_access_status'] != 'completed_once_bound_to_saved_best':
            raise ValueError('completed test provenance')
        if kind == 'C' and (any(counts.values()) or row['test_access_status'] != 'no_model_work_no_test'):
            raise ValueError('prepared state is not exact zero work')
        if kind == 'D' and Path(row['original_root']).exists():
            raise ValueError('not-started original output now retained')
    return inventory


def future_counts(c, t, row):
    p = profile(c, t); a = step_arithmetic(c, t)
    if row['classification'] == 'A':
        return dict.fromkeys(KINDS, 0)
    epochs = p['training']['epochs']-row.get('checkpoint_committed_epoch', 0)
    if epochs < 0:
        raise ValueError('checkpoint exceeds scientific epochs')
    return dict(adam=epochs*a['train_batches'], backward=epochs*a['train_batches'],
                forward=epochs*(a['train_batches']+math.ceil(a['validation_windows']/p['training']['eval_batch']))+
                math.ceil(a['test_windows_arithmetic_only']/p['training']['eval_batch']))


def budget_projection(c, inventory):
    validate_inventory(c, inventory)
    historical = dict.fromkeys(KINDS, 0); future = dict(historical); repeated = dict(historical)
    maximum = dict(historical); committed = dict(historical)
    for t, row in zip(c['tasks'], inventory['rows']):
        p = profile(c, t); a = step_arithmetic(c, t); cost = future_counts(c, t, row)
        full = dict(adam=a['max_optimizer_steps'], backward=a['max_optimizer_steps'],
                    forward=p['training']['epochs']*(a['train_batches']+math.ceil(a['validation_windows']/p['training']['eval_batch']))+
                    math.ceil(a['test_windows_arithmetic_only']/p['training']['eval_batch']))
        for k in KINDS:
            historical[k] += row['actual_charged_counts'][k]
            committed[k] += row['checkpoint_represented_counts'][k]
            future[k] += cost[k]; maximum[k] += full[k]
            if row['classification'] == 'B':
                repeated[k] += row['charged_work_after_checkpoint'][k]
    total = {k: historical[k]+future[k] for k in KINDS}
    return dict(formula='historical_actual_consumed + recovery_future_worst_case',
                original_formal_max=maximum, historical_actual_consumed=historical,
                checkpoint_committed_work=committed,charged_work_after_checkpoint=repeated,
                authorized_remaining={k:maximum[k]-historical[k] for k in KINDS},
                authorized_optimizer_max=maximum['adam'],
                authorization_fields=dict(adam='max_optimizer_steps hard authorization',backward='derived train arithmetic; no independent original authorization field',forward='derived train/validation/test arithmetic; no independent original authorization field'),
                recovery_future_worst_case=future, required_repeated_work=repeated,
                repeated_work_included_in_future=True, total_execution_worst_case=total,
                deficit_or_headroom={k: maximum[k]-total[k] for k in KINDS}, budget_refund=False)


def effective_map(c, inventory, result_root):
    validate_inventory(c, inventory)
    rows = {}
    for t, r in zip(c['tasks'], inventory['rows']):
        kind = r['classification']
        path = Path(r['original_root']) if kind == 'A' else Path(result_root)/('formal-'+t['model'])/t['id']
        rows[t['id']] = dict(classification=kind, effective_root=str(path), original_root=r['original_root'],
                             origin='original_v4_complete' if kind == 'A' else
                             'original_v4_checkpoint_plus_recovery' if kind == 'B' else 'recovery_fresh',
                             scientific_task_id=t['id'], selection_basis='inventory; never predictive effect')
    return dict(purpose='native_recovery_effective_result_map_v1', scope=ID, rows=rows, result_review='pending')


def pending_waves(waves, inventory):
    complete = {r['task_id'] for r in inventory['rows'] if r['classification'] == 'A'}
    # Filter inside each original wave. Never repack across waves.
    return [(n, [r for r in wave if r not in complete]) for n, wave in enumerate(waves) if any(r not in complete for r in wave)]


def validate_history(c, t, history):
    p = profile(c, t); a = step_arithmetic(c, t); best = BestState(p['training']['patience'])
    for epoch, row in enumerate(history, 1):
        if best.stopped or row['epoch'] != epoch or row['steps'] != epoch*a['train_batches']:
            raise ValueError('history progress/early stopping')
        v = row['validation']; channels = p['C'] if p.get('task') == 'M' else 1
        elements = a['validation_windows']*p['pred_len']*channels
        if v['elements'] != elements or any(not math.isfinite(v[k]) for k in ('mse','mae','sse','sae')) or v['mse'] != v['sse']/elements or v['mae'] != v['sae']/elements:
            raise ValueError('validation identity/finite/arithmetic')
        best.update(v['mse'], epoch)
        if row['best_epoch'] != best.epoch:
            raise ValueError('validation best/tie rule')
    if not history or len(history) > p['training']['epochs']:
        raise ValueError('history scientific epoch limit')
    return best
