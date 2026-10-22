"""Read-only upstream readiness; one sealed completion verification after exit."""
import math
import os
from pathlib import Path
from .contract import bound,read,sha,digest

class Blocked(RuntimeError):pass

def proc_identity(pid):
    try:
        s=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(start=s[19],state=s[0])
    except FileNotFoundError:return None

def live(owner):
    p=proc_identity(owner['pid'])
    return bool(p and p['start']==str(owner['start']) and p['state'] not in ('Z','X'))

def inspect(binding):
    a=Path(binding['folder'])
    try:
        auth=bound(binding['auth']);runtime=bound(binding['runtime']);config=bound(binding['config'])
        if runtime['authorization_ref']!=binding['auth']or runtime['config_ref']!=binding['config']or auth['task_ids']!=runtime['task_ids']or len(runtime['task_ids'])!=27 or len(set(runtime['task_ids']))!=27 or len(runtime['waves'])!=9:raise Blocked('upstream binding/27/9 mismatch')
        if runtime['task_ids']!=[x for w in runtime['waves']for x in w]or runtime['task_ids']!=[t['id']for t in config['tasks']]:raise Blocked('upstream exact grid/order')
        if any((a/n).exists()for n in ('STOP','failure.json','stopped.json')):raise Blocked('upstream STOP/failure: no automatic continuation')
        owner=read(a/'owner.json')
        if owner['runtime_ref']!=binding['runtime']or owner['pid']!=runtime['owner_pid']or owner['start']!=runtime['owner_start']:raise Blocked('upstream owner/runtime drift')
        children=read(a/'children.json');active=live(owner)or any(live(ch)for ch in children)
        if not (a/'complete.json').exists():
            if not live(owner):raise Blocked('upstream owner exited without technical completion')
            return dict(ready=False,state='WAIT_UPSTREAM_27_COMPLETE',progress=read(a/'progress.json'))
        complete=read(a/'complete.json')
        if complete.get('technical_complete')is not True or complete.get('task_ids')!=runtime['task_ids']or complete.get('waves')!=9:raise Blocked('upstream incomplete exact coverage')
        if active:return dict(ready=False,state='WAIT_UPSTREAM_27_COMPLETE',reason='complete exists; waiting owned processes to exit')
        if list((a/'staging').iterdir()):raise Blocked('upstream retained staging')
        return dict(ready=True,state='VERIFY_UPSTREAM_TECHNICAL_COMPLETION')
    except (KeyError,ValueError,OSError)as exc:raise Blocked('upstream malformed/changed: '+str(exc))from exc

def verify_completion(binding):
    if not inspect(binding)['ready']:raise Blocked('upstream not finished')
    a=Path(binding['folder']);rt=bound(binding['runtime']);cfg=bound(binding['config']);receipts=list((a/'receipts').glob('*.json'))
    if len(receipts)!=27 or len(list((a/'runs').iterdir()))!=27:raise Blocked('upstream run/receipt membership')
    assigned={t['id']:t for t in cfg['tasks']};seen=[];refs=[]
    for rp in receipts:
        r=read(rp);p=Path(r['path']);m=r['manifest'];ident=m['identity'];tid=m['task_id']
        if tid not in assigned or tid in seen or p.parent!=a/'runs'or p.is_symlink():raise Blocked('foreign/duplicate upstream receipt')
        seal=read(p/'seal.json');manifest=read(p/'manifest.json');metrics=read(p/'metrics.json')
        if m!=manifest or metrics!=r['metrics']or sha(p/'seal.json')!=r['seal_sha256']or seal['identity_sha']!=m['identity_sha']or seal['complete']is not True:raise Blocked('upstream receipt/seal mismatch')
        if sha(p/'checksums.sha256')!=seal['checksums_sha256']:raise Blocked('upstream checksum list changed')
        expected={}
        for line in (p/'checksums.sha256').read_text().splitlines():
            h,n=line.split('  ',1)
            if n in expected or Path(n).name!=n or n.startswith('.'):raise Blocked('upstream checksum member')
            expected[n]=h
        actual={f.name for f in p.iterdir()}
        required={'manifest.json','metrics.json','history.jsonl','updates.jsonl','best.pt','last.pt','test-claim.json','test-result.json','config.json','identity.json'}
        if not required <= set(expected):raise Blocked('upstream incomplete sealed payload')
        if actual!=set(expected)|{'seal.json','checksums.sha256'}:raise Blocked('upstream sealed membership')
        # Exactly once, after exit; bytes only, never checkpoint deserialization or history replay.
        for n,h in expected.items():
            if (p/n).is_symlink()or sha(p/n)!=h:raise Blocked('upstream sealed byte mismatch '+n)
        if ident['task']!=assigned[tid]or ident['config_sha']!=digest(cfg)or ident['source']!=rt['source_binding']or ident['producer_commit']!=rt['commit']:raise Blocked('upstream task/source identity')
        if m['launch_provenance']!=dict(runtime_ref=binding['runtime'],authorization_ref=binding['auth'])or m['status']!='completed'or metrics['test_calls']!=1:raise Blocked('upstream provenance/test completion')
        if any(not math.isfinite(metrics[k])for k in ('mse','mae')):raise Blocked('upstream nonfinite saved metric')
        seen.append(tid);refs.append(dict(path=str(rp),sha256=sha(rp)))
    if set(seen)!=set(rt['task_ids']):raise Blocked('upstream coverage')
    original=Path(binding['config']['path']).parents[1]
    for p,h in rt['source_binding'].items():
        if sha(original/p)!=h:raise Blocked('upstream execution byte drift')
    return dict(technical_complete=True,task_ids=rt['task_ids'],waves=9,receipt_refs=refs,complete_ref=dict(path=str(a/'complete.json'),sha256=sha(a/'complete.json')),checkpoint_loads=0,test_forward=0,effect_gate_used=False)
