"""Closure-bound user retirement; JSON/stat evidence only, never a success report."""
import json
import subprocess
from pathlib import Path
from utils import ch3_m_tasks as source
from utils import ch3_native_tasks as scope
from utils.ch3_contract import digest
from utils.ch3_m_handoff import OLD_QUEUE_START_ANCHOR
from m6_remaining_entry import same

RECEIPT={
    'path':'/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/native-time-mark-chain-v2/retirement/receipt.json',
    'sha256':'973898058035a04ddd0bae72ab6df2df6842cc924b3fbc83e79d420fa8d1f1a1',
}
DECISION='retire remaining N and all S to prioritize successor experiments'
KIND='old_MS_user_authorized_retirement_boundary_v1'


def validate_partition(r,inventory,original):
    """Pure accounting gate used by both real binding and synthetic negative tests."""
    ids=[t['id'] for t in original['tasks']]
    if len(ids)!=552 or r['original_task_ids']!=ids or inventory['original_task_ids']!=ids or set(inventory['runs'])!=set(ids):raise ValueError('exact original 552 retirement scope')
    exact=dict(kind='old_MS_user_authorized_retirement_receipt_v1',status='user_authorized_protocol_retirement',
               user_decision=DECISION,synthetic_fixture=False,scientific_failure=False,technical_complete=False,
               retirement_accepted=True,result_review='pending',budget_refund=False,artifact_deleted=False,
               artifact_overwritten=False,checkpoint_audit_performed=False)
    for k,v in exact.items():
        if r.get(k)!=v or (type(v)is bool and type(r.get(k))is not bool):raise ValueError('retirement field '+k)
    completed=[t for t in ids if inventory['runs'][t]['completed']]
    retired=[t for t in ids if t not in completed]
    n=[t['id'] for t in original['tasks'] if t['model']=='N']
    s=[t['id'] for t in original['tasks'] if t['model']=='S']
    partial=[t for t in n if inventory['runs'][t].get('partial')]
    not_started=[t for t in n if not inventory['runs'][t]['exists']]
    nc=[t for t in n if t in completed];ni=[t for t in n if t not in completed]
    values=dict(completed_task_ids=completed,retired_task_ids=retired,N_completed=nc,N_incomplete=ni,
                N_not_started=not_started,incomplete_partial_N_task_ids=partial,S_not_executed=s)
    for k,v in values.items():
        if r[k]!=v or len(v)!=len(set(v)):raise ValueError('retirement partition '+k)
    if set(retired)!=set(ni+s) or len(n)!=24 or len(s)!=24:raise ValueError('only remaining N and all S can retire')
    if any(inventory['runs'][t]['exists'] or inventory['runs'][t]['completed'] for t in s):raise ValueError('S was executed')
    for t,row in inventory['runs'].items():
        if row.get('completed') and (not row['exists'] or not all(name in row['sources'] for name in ('result.json','manifest.json','history.jsonl','budget.json'))):raise ValueError('partial cannot be Passed')
        if row.get('partial') and (row['completed'] or not row['exists'] or 'result.json' in row['sources']):raise ValueError('partial misclassified')
        if row['exists'] and row['identity'].get('run_id')!=t:raise ValueError('recorded actual run identity')
    counts=dict(original=552,completed_before_stop=len(completed),completed_after_stop=len(completed),
                N_completed=len(nc),N_partial=len(partial),N_not_started=len(not_started),N_retired=len(ni),S_not_executed=len(s),retired=len(retired))
    if r['counts']!=counts or (len(completed),len(nc),len(partial),len(not_started))!=(520,16,4,4):raise ValueError('exact stopped snapshot counts')
    def total(selected):
        rows=[inventory['runs'][t] for t in selected]
        return {k:sum(x.get('counts',{}).get(k,0) for x in rows) for k in ('adam','forward','backward')}|{'recorded_completed_epochs':sum(x.get('recorded_epochs',0) for x in rows)}
    if r['consumption']['recorded_registered_552']!=total(ids) or r['consumption']['recorded_N']!=total(n):raise ValueError('persisted budget not refundable/reproducible')
    return r


def validate_receipt(worker=False):
    # Read exactly the reviewed bytes BEFORE following any controller reference.
    r=source.bound(RECEIPT);original=scope.parent_config();inventory=source.bound(r['run_inventory'])
    validate_partition(r,inventory,original)
    p=r['production']
    if p['commit']!=source.BASE or p['protocol_sha']!=digest(original) or p['config']['sha256']!=source.sha(p['config']['path']):raise ValueError('retired production version changed')
    if worker:return r
    production_root=Path(p['config']['path']).parents[1]
    if subprocess.check_output(['git','-C',str(production_root),'rev-parse','HEAD'],text=True).strip()!=p['commit']:raise ValueError('retired production HEAD changed')
    if any(source.sha(production_root/path)!=value for path,value in p['code'].items()):raise ValueError('retired production source bytes changed')
    from utils.ch3_m6 import source_states
    states=source_states(original)
    for child in p['child_source_bindings'].values():
        approval=source.bound(child['approval'])
        if child['source_states']!=approval['source_states'] or child['data_bindings']!=approval['data_bindings'] or any(states.get(path)!=state for path,state in child['source_states'].items()):raise ValueError('retired data/source binding state changed')
    start=source.bound(OLD_QUEUE_START_ANCHOR);anchor=start['queue']['controller.json'];q=r['old_queue'];root=Path(q['root'])
    if root!=scope.EVIDENCE/'queues/remaining-models-v1' or q['controller']['sha256']!=anchor['sha256'] or source.bound(q['controller'])!=anchor['content']:raise ValueError('fixed old controller anchor changed')
    if (root/'complete.json').exists():raise ValueError('retired original batch is not complete')
    before=source.bound(r['before'])
    if before['completed_task_ids']!=r['completed_task_ids'] or before['progress']['current_model']!='N' or before['controller']!=anchor['content']:raise ValueError('stop before-state changed')
    current=source.bound(q['current']);failure=source.bound(q['failure']);source.bound(q['accounting'])
    if current['model']!='N' or failure.get('current_model')!='N' or failure.get('pending_models')!=['S'] or failure.get('automatic_retry')is not False:raise ValueError('unexpected retirement stop/failure scope')
    if q['STOP']['path']!=str(root/'STOP') or source.sha(q['STOP']['path'])!=q['STOP']['sha256']:raise ValueError('persistent old STOP missing/changed')
    stopped=source.bound(q['process_exit'])
    if stopped['remaining'] or stopped['extra_signals_sent']!=0 or len(stopped['observed'])!=len(stopped['original_processes']):raise ValueError('old process exit evidence incomplete')
    for row in stopped['observed']:
        if row['same_instance_live']is not False or same(row['original']):raise ValueError('retired old original still active')
    if same(anchor['content']) or same(current['child']):raise ValueError('retired old controller/child still active')
    v1=r['retired_successor']
    if v1['scope']!='m6-native-tmark-chain-v1' or source.bound(v1['progress'])['state']!='WAIT_OLD_MS':raise ValueError('v1 did not retire at waiting boundary')
    for key in ('controller','failure','stop_command','process_exit'):source.bound(v1[key])
    if source.sha(v1['STOP']['path'])!=v1['STOP']['sha256'] or source.sha(v1['launcher']['path'])!=v1['launcher']['sha256']:raise ValueError('v1 retired evidence changed')
    exit1=source.bound(v1['process_exit'])
    if exit1['old_MS_signal_sent']is not False or exit1['same_instance_live']is not False or same(source.bound(v1['controller'])):raise ValueError('v1 retirement process boundary')
    for t,row in inventory['runs'].items():
        path=Path(row['path'])
        if path.is_symlink() or not path.resolve().is_relative_to(scope.EVIDENCE) or path.name!=t or path.exists()!=row['exists']:raise ValueError('retired artifact path/state changed')
        if not row['exists']:continue
        for ref in row['sources'].values():
            if Path(ref['path']).parent!=path or source.sha(ref['path'])!=ref['sha256']:raise ValueError('retired JSON/history/budget changed')
        if (path/'result.json').exists()!=row['completed']:raise ValueError('partial retirement changed to completion')
        for entry in row.get('checkpoint_state',{}).values():
            st=Path(entry['ref']['path']).stat()
            if (st.st_size,st.st_mtime_ns)!=(entry['size'],entry['mtime_ns']):raise ValueError('retained N checkpoint state changed')
    return r


def boundary(worker=False):
    r=validate_receipt(worker)
    keys=('original_task_ids','completed_task_ids','retired_task_ids','incomplete_partial_N_task_ids','N_completed','N_incomplete','N_not_started','S_not_executed','counts','consumption','production','old_queue','retired_successor')
    return dict(kind=KIND,retirement_receipt=RECEIPT,technical_complete=False,retirement_accepted=True,
                scientific_failure=False,budget_refund=False,result_review='pending',
                authorization_basis='old MS original batch explicitly retired by user; successor authorized from exact retirement boundary',
                **{k:r[k] for k in keys})
