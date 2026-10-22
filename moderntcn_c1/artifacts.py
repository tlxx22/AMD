"""C.1 flat immutable seals, strict saved-history audit, no torch imports."""
import math
import os
import subprocess
from pathlib import Path
from .contract import read,sha,digest,atomic,profile,step_arithmetic,ID
from utils.ch3_contract import BestState

def audit(path,c,t,commit,full=True,sealed=True):
    p=Path(path);m=read(p/'manifest.json');r=read(p/'result.json');hist=[__import__('json').loads(x)for x in (p/'history.jsonl').read_text().splitlines()];ident=m['identity'];pr=profile(c,t)
    if m['task']!=t or m['profile']!=pr or ident['run_id']!=t['id']or ident['revision']!=ID or ident['commit']!=commit or ident['protocol_sha']!=digest(c)or ident['profile_sha']!=digest(pr)or ident['source']!=c['source_binding']or ident['data_sha']!=c['data_bindings'][t['dataset']][t['id']]:raise ValueError('C1 sealed identity')
    if digest(m['metadata'])!=ident['data_sha']:raise ValueError('C1 saved data binding')
    best=BestState(pr['training']['patience']);ar=step_arithmetic(c,t)
    for n,row in enumerate(hist,1):
        if best.stopped or row['epoch']!=n or row['steps']!=n*ar['train_batches']:raise ValueError('C1 history/count/early stop')
        best.update(row['validation']['mse'],n)
        if row['best_epoch']!=best.epoch:raise ValueError('C1 strict best')
    if not hist or (len(hist)!=pr['training']['epochs']and not best.stopped):raise ValueError('C1 incomplete epochs')
    if r['id']!=t['id']or r['commit']!=commit or r['protocol_sha']!=digest(c)or r['profile_sha']!=digest(pr)or r['best_epoch']!=best.epoch or r['metric_space']!='train-standardized'or r['metric_scope']!=t['metric_scope']:raise ValueError('C1 metric identity')
    if r['final_test']!=dict(calls=1,selected='best.pt',sha256=sha(p/'best.pt'),epoch=best.epoch):raise ValueError('C1 selected best/test once')
    claim=read(p/'test-claim.json')
    if claim!=dict(identity_sha=digest(ident),calls=1,best_epoch=best.epoch):raise ValueError('C1 test claim')
    for k,total in [('mse','sse'),('mae','sae')]:
        if not math.isfinite(r[k])or not math.isfinite(r[total])or r['elements']<=0 or r[k]!=r[total]/r['elements']:raise ValueError('C1 metric arithmetic/finite')
    if sealed and not (p/'seal.json').is_file():raise ValueError('C1 final seal required')
    if (p/'seal.json').exists():
        seal=read(p/'seal.json')
        if seal['schema']!='moderntcn-C1-seal-v1'or seal['task_id']!=t['id']or seal['checksums_sha256']!=sha(p/'checksums.sha256'):raise ValueError('C1 seal')
        expected={}
        for line in (p/'checksums.sha256').read_text().splitlines():
            h,n=line.split('  ',1)
            if n in expected or Path(n).name!=n or n.startswith('.'):raise ValueError('C1 checksum member')
            expected[n]=h
        if {x.name for x in p.iterdir()}!=set(expected)|{'checksums.sha256','seal.json'}:raise ValueError('C1 member set')
        if full:
            if any((p/n).is_symlink()or sha(p/n)!=h for n,h in expected.items()):raise ValueError('C1 checksum')
            subprocess.run(['sha256sum','--check','checksums.sha256'],cwd=p,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    return dict(task_id=t['id'],path=str(p),epochs=len(hist),updates=hist[-1]['steps'],best_epoch=best.epoch,metrics={k:r[k]for k in ('mse','mae','sse','sae','elements')},test_calls=1,identity_sha=digest(ident))

def publish(staging,final,c,t,commit):
    p=Path(staging);final=Path(final)
    if final.exists():raise FileExistsError('successful C1 task never overwritten')
    audit(p,c,t,commit,sealed=False)
    files=sorted(p.iterdir())
    if any(not x.is_file()or x.is_symlink()or '.tmp' in x.name for x in files):raise ValueError('C1 flat final payload')
    hashes={x.name:sha(x)for x in files};out=subprocess.check_output(['sha256sum','--',*hashes],cwd=p,text=True)
    if {n:h for h,n in (l.split('  ',1)for l in out.splitlines())}!=hashes:raise ValueError('system/python checksums')
    (p/'checksums.sha256').write_text(''.join(h+'  '+n+'\n'for n,h in hashes.items()))
    atomic(p/'seal.json',dict(schema='moderntcn-C1-seal-v1',task_id=t['id'],complete=True,checksums_sha256=sha(p/'checksums.sha256')))
    audit(p,c,t,commit);final.parent.mkdir(exist_ok=True);os.rename(p,final)
    row=audit(final,c,t,commit,full=False);row['seal_ref']=dict(path=str(final/'seal.json'),sha256=sha(final/'seal.json'));return row
