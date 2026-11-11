"""One explicit import-repair continuation: retain exactly the original eight S jobs."""
from pathlib import Path
from .contract import read, sha, load_ref

FIRST = 'test-reporting-v10-first-r1'
RETRY = 'test-reporting-v10-import-retry1'
FIELDS = ('technical_continuation','execution_jobs','execution_waves','original_wave_indices','continuation_budget')


def execution_jobs(c):return c.get('execution_jobs',c['jobs'])

def execution_waves(c):return c.get('execution_waves',c['waves'])


def validate(c):
    if c['attempt']==FIRST:
        if any(k in c for k in FIELDS):raise ValueError('original scope cannot inherit new continuation fields')
        return
    if c['attempt']!=RETRY or any(k not in c for k in FIELDS):raise ValueError('only explicit import retry1')
    t=c['technical_continuation']
    if t['schema']!='m6-test-import-continuation-v1':raise ValueError('continuation schema')
    old=load_ref(t['config_ref']);runtime=load_ref(t['runtime_ref']);auth=load_ref(t['auth_ref'])
    if old['attempt']!=FIRST or runtime['scope']!=c['scope'] or runtime['config_ref']!=t['config_ref'] or runtime['auth_ref']!=t['auth_ref'] or runtime['commit']!=auth['commit'] or auth['config_sha256']!=t['config_ref']['sha256'] or auth['review_approved'] is not True:raise ValueError('original producer binding')
    if runtime['environment_ref']!=old['environment_ref'] or auth['source_binding']!=old['source_binding']:raise ValueError('original environment/source binding')
    if set(c)!=set(old)|set(FIELDS):raise ValueError('unexpected continuation fields')
    for k in old:
        if k not in ('attempt','source_binding','preparation_head') and c[k]!=old[k]:raise ValueError('scientific or resource field changed: '+k)
    if c['execution_jobs']!=old['jobs'][8:] or c['execution_waves']!=old['waves'][8:] or c['original_wave_indices']!=list(range(9,22)):raise ValueError('exact remaining order')
    budget=dict(inherited_model_jobs=8,execution_model_jobs=12,execution_offline_jobs=1,execution_waves=13,batches=780,presentations=99180,elements=2264400,training=0)
    if c['continuation_budget']!=budget:raise ValueError('remaining budget')
    folder=Path(c['result_root'])/FIRST
    for key,name in [('runtime_ref','runtime.json'),('failure_ref','failure.json'),('case_plan_ref','case-plan.json')]:
        if Path(t[key]['path']).resolve()!=(folder/name).resolve():raise ValueError('original attempt path')
        load_ref(t[key])
    if Path(t['failed_access_ref']['path']).resolve()!=(folder/'PT-NP-ModernTCN/access.jsonl').resolve() or sha(t['failed_access_ref']['path'])!=t['failed_access_ref']['sha256']:raise ValueError('failed test access trace')
    load_ref(t['repair_ref'])
    index=load_ref(t['inherited_ref'])
    if index['schema']!='m6-test-inherited-eight-v1' or index['runtime_ref']!=t['runtime_ref'] or [r['job'] for r in index['rows']]!=old['jobs'][:8]:raise ValueError('exact inherited eight')
    if {p.stem for p in (folder/'receipts').glob('*.json')}!={j['id'] for j in old['jobs'][:8]}:raise ValueError('old receipt set changed')
    if {p.name for p in folder.glob('wave-*-complete.json')}!={f'wave-{i}-complete.json' for i in range(1,9)}:raise ValueError('old completed waves changed')
    from .queue import checked_receipt
    from .contract import evaluation_key
    for i,row in enumerate(index['rows'],1):
        job=row['job'];ref=row['receipt_ref']
        if Path(ref['path']).resolve()!=(folder/'receipts'/(job['id']+'.json')).resolve():raise ValueError('inherited receipt path')
        receipt=load_ref(ref)
        if checked_receipt(folder,job,folder/'runtime.json')!=receipt or receipt['result']['evaluation_key']!=evaluation_key(old,job):raise ValueError('inherited receipt/claim mismatch')
        if read(folder/f'wave-{i}-complete.json')!={'jobs':[job['id']]}:raise ValueError('inherited wave identity')


def location(c, folder, job):
    """Return the real producer folder/runtime; never copy receipts into the new run."""
    if job not in c['jobs'] and job!={'id':'O','phase':'O'}:raise ValueError('unknown result job')
    if c.get('technical_continuation') and job in c['jobs'][:8]:
        t=c['technical_continuation'];index=load_ref(t['inherited_ref']);row=next(r for r in index['rows'] if r['job']==job)
        load_ref(row['receipt_ref']);load_ref(t['runtime_ref'])
        p=Path(t['runtime_ref']['path']);return p.parent,p,True
    folder=Path(folder);return folder,folder/'runtime.json',False


def receipt(c, folder, job):
    from .queue import checked_receipt
    origin,runtime,_=location(c,folder,job)
    return checked_receipt(origin,job,runtime)
